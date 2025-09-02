import os
import random
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.optim import Optimizer

import sed_scores_eval
import torch
import torch_geometric.data
import torchmetrics
from codecarbon import OfflineEmissionsTracker
from desed_task.data_augm import mixup
from desed_task.evaluation.evaluation_measures import (
    compute_per_intersection_macro_f1, compute_psds_from_operating_points,
    compute_psds_from_scores)
from desed_task.utils.scaler import TorchScaler
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram

from .utils import batched_decode_preds, batched_node_decode_preds, log_sedeval_metrics


from torch.utils.data import  DataLoader



class RespiraSED_lab(pl.LightningModule):
    """Pytorch lightning module for the SED 2021 baseline
    Args:
        hparams: dict, the dictionary to be used for the current experiment/
        encoder: ManyHotEncoder object, object to encode and decode labels.
        EzhouNet: torch.Module, the student model to be trained. The teacher model will be
        opt: torch.optimizer.Optimizer object, the optimizer to be used
        train_data: torch.utils.data.Dataset subclass object, the training data to be used.
        valid_data: torch.utils.data.Dataset subclass object, the validation data to be used.
        test_data: torch.utils.data.Dataset subclass object, the test data to be used.
        train_sampler: torch.utils.data.Sampler subclass object, the sampler to be used in the training dataloader.
        scheduler: BaseScheduler subclass object, the scheduler to be used.
                   This is used to apply ramp-up during training for example.
        fast_dev_run: bool, whether to launch a run with only one batch for each set, this is for development purpose,
            to test the code runs.
    """

    def __init__(
            self,
            hparams,
            encoder,
            GraphNet,
            opt=None,
            train_flag = False,
            train_data=None,
            valid_data=None,
            test_data=None,
            train_sampler=None,
            scheduler=None,
            fast_dev_run=False,
            evaluation=False,

    ):
        super(RespiraSED_lab, self).__init__()
        self.hparams.update(hparams)

        self.encoder = encoder
        self.sed_net = GraphNet
        self.train_flag = train_flag
        # if sed_teacher is None:
        #     self.sed_teacher = deepcopy(EzhouNet)
        # else:
        #     self.sed_teacher = sed_teacher

        self.ema_model = deepcopy(self.sed_net)
        for para in self.ema_model.parameters():
            para.requires_grad = False


        self.opt = opt
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.train_sampler = train_sampler
        self.scheduler = scheduler
        self.fast_dev_run = fast_dev_run
        self.evaluation = evaluation

        # add by cy, this para  call  by the  forward func:
        # batch:  (48, 10, 156)
        # self.example_input_array = torch.Tensor( 48, 128, 626)

        if self.fast_dev_run:
            self.num_workers =  self.hparams["training"]["debug_num_workers"]
        else:
            self.num_workers = self.hparams["training"]["num_workers"]

        feat_params = self.hparams["feats"]
        self.mel_spec = MelSpectrogram(
            sample_rate=feat_params["sample_rate"],
            n_fft=feat_params["n_window"],
            win_length=feat_params["n_window"],
            hop_length=feat_params["hop_length"],
            f_min=feat_params["f_min"],
            f_max=feat_params["f_max"],
            n_mels=feat_params["n_mels"],
            window_fn=torch.hamming_window,
            wkwargs={"periodic": False},
            power=1,
        )

        # for param in self.sed_teacher.parameters():  # 创建监督损失 supervised_loss
        #     param.detach_()  # 对每个参数调用 param.detach_() 方法，将其从计算图中分离出去。用于停止梯度传播或冻结某些层以防止其被更新

        # instantiating losses
        self.supervised_loss = torch.nn.CrossEntropyLoss()
        if hparams["training"]["record_loss_type"] == "mse":
            self.record_level_loss = torch.nn.MSELoss()
        elif hparams["training"]["record_loss_type"] == "bce":
            self.record_level_loss = torch.nn.BCELoss()
        else:
            raise NotImplementedError

        # Note, for weak labels we simply compute f1 score
        self.record_level_bin_f1_seg_macro = (  # 创建两个多标签宏平均 F1 分数计算对象，分别用于学生模型和教师模型
            torchmetrics.classification.f_beta.MultilabelF1Score(

                len(self.encoder.binary_labels),
                threshold= 0.1 ,
                average="macro"
            )
        )

        #self.scaler = self._init_scaler()  # 该方法根据指定参数初始化用于数据标准化的缩放器。
        # buffer for event based scores which we compute using sed-eval





        self.val_buffer_node_level = {  # 为不同阈值创建多个 DataFrame 缓冲区，用于存储合成数据和测试数据的验证结果。
            k: pd.DataFrame() for k in self.hparams["training"]["val_thresholds"]
        }
        self.val_scores_postprocessed_buffer_node_level = {}



        self.val_buffer_node_level_test = {
            k: pd.DataFrame() for k in self.hparams["training"]["val_thresholds"]
        }
        # 为测试阶段的不同阈值创建多个 DataFrame 缓冲区。
        # 使用字典形式为学生和教师分别创建了不同阈值下的psds缓冲区，
        test_n_thresholds = self.hparams["training"]["n_test_thresholds"]
        # 该段代码首先使用numpy库生成一个数组test_thresholds，
        # 其中包含从1/(test_n_thresholds * 2)到1之间的等差数列，
        # 步长为1/test_n_thresholds。该数组用于定义测试阶段的不同阈值。
        test_thresholds = np.arange(
            1 / (test_n_thresholds * 2), 1, 1 / test_n_thresholds
        )
        self.test_psds_buffer_node_level = {k: pd.DataFrame() for k in
                                         test_thresholds}
        self.decoded_node_level_05_buffer = pd.DataFrame()
        self.test_scores_raw_buffer_node_level = {}
        self.test_scores_postprocessed_buffer_node_level = {}
        # 创建了四个空字典，用于存储学生和教师的原始分数、以及经过后处理的分数。
        # 创建额外的缓冲区用于存储解码后的结果和原始分数。
        # 这些字典可能后续会被用来保存不同测试的数据。


        self.best_val_score = float('-inf')  # Track the best validation score
        self.best_model_weights = None       # Placeholder for the best model weights (if needed)

        # self.val_buffer_teacher_synth = {
        #     k: pd.DataFrame() for k in self.hparams["training"]["val_thresholds"]
        # }
        # self.val_buffer_teacher_test = {
        #     k: pd.DataFrame() for k in self.hparams["training"]["val_thresholds"]
        # }
        # self.val_scores_postprocessed_buffer_teacher_synth = {}
        #
        # self.test_psds_buffer_teacher = {k: pd.DataFrame() for k in test_thresholds}
        # self.decoded_teacher_05_buffer = pd.DataFrame()
        # self.test_scores_raw_buffer_teacher = {}
        # self.test_scores_postprocessed_buffer_teacher = {}  #
        # 通过这些步骤，该初始化方法为后续的训练、验证和测试过程准备了必要的组件和数据结构。


    _exp_dir = None

    @property
    def exp_dir(self):
        if self._exp_dir is None:
            try:
                self._exp_dir = self.logger.log_dir
            except Exception as e:
                self._exp_dir = self.hparams["log_dir"]
        return self._exp_dir

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step()

    def on_train_start(self) -> None:
        # Print device of the model before training starts
        model_device = next(self.sed_net.parameters()).device
        print(f"Model is loaded on device: {model_device}")

        os.makedirs(os.path.join(self.exp_dir, "training_codecarbon"), exist_ok=True)
        self.tracker_train = OfflineEmissionsTracker(
            "Respiratory Sound  SED TRAINING",
            output_dir=os.path.join(self.exp_dir, "training_codecarbon"),
            log_level="warning",
            country_iso_code="FRA",
        )
        self.tracker_train.start()

        # Remove for debugging. Those warnings can be ignored during training otherwise.
        # to_ignore = []
        to_ignore = [
            ".*Trying to infer the `batch_size` from an ambiguous collection.*",
            ".*invalid value encountered in divide*",
            ".*mean of empty slice*",
            ".*self.log*",
        ]
        for message in to_ignore:
            warnings.filterwarnings("ignore", message)

    def update_ema(self, alpha, global_step):
        """Update teacher model parameters

        Args:
            alpha: float, the factor to be used between each updated step.
            global_step: int, the current global step to be used.
            model: torch.Module, student model to use
            ema_model: torch.Module, teacher model to use
        """
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_params, params in zip(self.ema_model.parameters(), self.sed_net.parameters()):
            ema_params.data.mul_(alpha).add_(params.data, alpha=1 - alpha)

    def _init_scaler(self):
        """Scaler inizialization
        #该方法根据指定参数初始化用于数据标准化的缩放器。
        它检查要使用的缩放类型（“实例”或“数据集”）并相应地创建TorchScaler的实例。
        如果提供了保存路径并且该路径中已存在缩放器，则会加载保存的缩放器。
        如果不是，它将缩放器适合来自数据加载器的训练数据，应用转换函数来处理数据。
        最后，如果指定了保存路径，它将保存新安装的缩放器以供将来使用。如
        Raises:
            NotImplementedError: in case of not Implemented scaler

        Returns:
            TorchScaler: returns the scaler
        """
        # 根据 self.hparams["scaler"]["statistic"] 配置选择初始化方式：
        if self.hparams["scaler"]["statistic"] == "instance":
            scaler = TorchScaler(  # 若为 "instance"，直接创建 TorchScaler 实例并返回。
                "instance",
                self.hparams["scaler"]["normtype"],
                self.hparams["scaler"]["dims"],
            )

            return scaler
        elif self.hparams["scaler"]["statistic"] == "dataset":
            # we fit the scaler
            scaler = TorchScaler(
                "dataset",
                self.hparams["scaler"]["normtype"],
                self.hparams["scaler"]["dims"],
            )
        else:
            raise NotImplementedError
        if self.hparams["scaler"]["savepath"] is not None:
            if os.path.exists(self.hparams["scaler"]["savepath"]):  # 若路径存在，则从该路径加载已保存的 TorchScaler 并返回
                scaler = torch.load(self.hparams["scaler"]["savepath"])
                print(
                    "Loaded Scaler from previous checkpoint from {}".format(
                        self.hparams["scaler"]["savepath"]
                    )
                )
                return scaler

        self.train_loader = self.train_dataloader()
        scaler.fit(
            self.train_loader,
            transform_func=lambda x: self.take_log(self.mel_spec(x[0])),
        )

        if self.hparams["scaler"]["savepath"] is not None:
            torch.save(scaler, self.hparams["scaler"]["savepath"])
            print(
                "Saving Scaler from previous checkpoint at {}".format(
                    self.hparams["scaler"]["savepath"]
                )
            )
            return scaler

    def take_log(self, mels):
        """Apply the log transformation to mel spectrograms.
        Args:
            mels: torch.Tensor, mel spectrograms for which to apply log.

        Returns:
            Tensor: logarithmic mel spectrogram of the mel spectrogram given as input
        """
        # mels (bt, nmels=128, n_frams=626), 调用amp_to_db实例处理输入的梅尔频谱图mels，将幅度值转换为分贝值
        amp_to_db = AmplitudeToDB(stype="amplitude")
        amp_to_db.amin = 1e-5  # amin= 1e-5 as in librosa, 使用clamp方法将变换后的结果限制在-50到80之间，防止异常值影响后续处理
        return amp_to_db(mels).clamp(min=-50, max=80)  # clamp to reproduce old code, 将结果限制在-50到80之间。

    # def forward(self, mel_feats, model):
    #     return model(self.scaler(self.take_log(mel_feats)))

    def detect(self, mel_feats, model):
        return model(self.scaler(self.take_log(mel_feats)))

    def training_step(self, batch_data, batch_indx):
        """Apply the training for one batch (a step). Used during trainer.fit

        Args:
            batch_data: dict containing batched data
            batch_idx: int, index of the batch

        Returns:
            torch.Tensor, the loss to take into account.
        """

        # audio, labels, padded_indxs, _ = batch  # audio:(bt, 160k), labels:(bt,10,156),  pad_ind:(48* 1), mask_event:(bt,10)
        # x, edge_index, batch_indices = data.x, data.edge_index, data.batch
        # 此时返回的 batch_data 是list,  其中有两项，
        # batch_data[0]:   batch 个样本构成的图数据， 即将一个batch中多个音频的node节点特征，组合在一起。
        # batch_data[1]：  batch 个样本， 每个样本的音频文件名称。

        # Print device of the input batch to check if it is on GPU
        # data_device = batch_data['spectrograms'].device
        # print(f"\n Training on device: {data_device}")

        # You can also print the model's device by checking one of the parameters
        device = next(self.sed_net.parameters()).device
        print(f"Model is on device: {device}")



        # Move data to device
        # spectrograms = batch_data['spectrograms'].to(device)
        # frame_labels = batch_data['frame_labels'].to(device)
        spectrograms = [torch.tensor(spectrogram).to(device) for spectrogram in batch_data['spectrograms']]
        frame_labels = [torch.tensor(frame_label).to(device) for frame_label in batch_data['frame_labels']]
        c_ex_mixtures = batch_data['c_ex_mixtures']  # List of audio names (metadata)


        # Prepare batch_data dictionary for the model
        batch_data_device = {
            'spectrograms': spectrograms,
            'frame_labels': frame_labels,
            'c_ex_mixtures': c_ex_mixtures,
        }

        # Forward pass through the model
        outputs = self.sed_net(batch_data_device)

        # Unpack model outputs
        node_predictions = outputs['node_predictions']  # Tensor of shape (total_nodes, num_classes)
        record_predictions = outputs['record_predictions']  # Tensor of shape (batch_size, num_classes)
        node_labels = outputs['node_labels']  # Tensor of shape (total_nodes, num_classes)
        batch_indices = outputs['batch_indices']  # Tensor of shape (total_nodes,)

        # Generate record-level labels based on node labels
        record_labels = self.sed_net.generate_record_label(node_labels, batch_indices)
        # Move labels to device if not already
        record_labels = record_labels.to(device)
        node_labels  = node_labels.to(device)

        # Supervised loss
        node_loss = self.supervised_loss(node_predictions, node_labels)
        audio_bin_loss = self.supervised_loss(record_predictions, record_labels)

        # we apply consistency between the predictions, use the scheduler for learning rate (to be changed ?)
        # 保留该权重，用于后续调整两个损失的 系数；
        weight = (
                self.hparams["training"]["const_max"]
                * self.scheduler["scheduler"]._get_scaling_factor()
        )

        # audio_bin_loss = audio_bin_loss * weight
        tot_loss =  node_loss + audio_bin_loss

        self.log("train/node_level_loss", node_loss)
        self.log("train/audio_level_loss", audio_bin_loss)
        self.log("train/total_loss", tot_loss)


        # 记录日志信息：记录多个训练过程中重要的损失值，便于监控训练进度。
        self.log("train/step", self.scheduler["scheduler"].step_num, prog_bar=True)
        self.log("train/loss_ratio_weight", weight)
        self.log("train/lr", self.opt.param_groups[-1]["lr"], prog_bar=True)

        # print(f"\n *******the train step check  out")


        return tot_loss

    # def on_before_zero_grad(self, *args, **kwargs):
    #     # update EMA teacher
    #     self.update_ema(
    #         self.hparams["training"]["ema_factor"],
    #         self.scheduler["scheduler"].step_num,
    #         self.sed_net,
    #     )

    def validation_step(self, batch_data, batch_indx):
        """Apply validation to a batch (step). Used during trainer.fit

        Args:
            batch: torch.Tensor, input batch tensor
            batch_indx: torch.Tensor, 1D tensor of indexes to know which data are present in each batch.
        Returns:
        """

        # Print device of the input batch to check if it is on GPU
        # data_device = batch_data['spectrograms'].device
        # print(f"\n Validation on device: { data_device }")

        # You can also print the model's device by checking one of the parameters
        device = next(self.sed_net.parameters()).device
        print(f"\nValidation Model is on device: {device}")


        # Move data to device
        # spectrograms = batch_data['spectrograms'].to(device)
        # frame_labels = batch_data['frame_labels'].to(device)
        spectrograms = [torch.tensor(spectrogram).to(device) for spectrogram in batch_data['spectrograms']]
        frame_labels = [torch.tensor(frame_label).to(device) for frame_label in batch_data['frame_labels']]


        batch_audio_names = batch_data['c_ex_mixtures']  # List of audio names (metadata)


        # Prepare batch_data dictionary for the model
        batch_data_device = {
            'spectrograms': spectrograms,
            'frame_labels': frame_labels,
            'c_ex_mixtures': batch_audio_names,
        }

        # Forward pass through the model
        outputs = self.sed_net(batch_data_device)

        # Unpack model outputs
        node_predictions = outputs['node_predictions']  # Tensor of shape (total_nodes, num_classes)
        record_predictions = outputs['record_predictions']  # Tensor of shape (batch_size, num_classes)
        node_labels = outputs['node_labels']  # Tensor of shape (total_nodes, num_classes)
        batch_indices = outputs['batch_indices']  # Tensor of shape (total_nodes,)

        # Generate record-level labels based on node labels
        record_labels = self.sed_net.generate_record_label(node_labels, batch_indices)

        # Move labels to device if not already
        record_labels = record_labels.to(device)
        node_labels = node_labels.to(device)

        # Supervised loss
        node_loss = self.supervised_loss(node_predictions, node_labels)
        audio_bin_loss = self.supervised_loss(record_predictions, record_labels)

        # we apply consistency between the predictions, use the scheduler for learning rate (to be changed ?)
        # 保留该权重，用于后续调整两个损失的 系数；
        weight = (
                self.hparams["training"]["const_max"]
                * self.scheduler["scheduler"]._get_scaling_factor()
        )

        # audio_bin_loss = audio_bin_loss * weight
        tot_loss =  node_loss + audio_bin_loss

        self.log("valid/node_level_loss", node_loss)
        self.log("valid/audio_level_loss", audio_bin_loss)
        self.log("valid/total_loss", tot_loss)

        # Convert labels to one-hot encoding
        record_labels_one_hot = torch.nn.functional.one_hot(record_labels, num_classes=2).float()  # Shape: [12, 2]

        self.record_level_bin_f1_seg_macro(record_predictions, record_labels_one_hot)

        filenames_synth = [ x  for x in batch_audio_names
                        if Path(x).parent == Path(self.hparams["data"]["eval_folder_8k"])
                         ]

        #note, 获取当前batch 中每个音频的持续时间长度。
        batch_audio_duration = []
        valid_df = pd.read_csv(self.hparams["data"]["valid_dur"], sep='\t')
        # Iterate over your list of filenames
        for file in filenames_synth:
            file = os.path.basename(file)
            # Find the row in the DataFrame that matches the filename
            duration = valid_df.loc[valid_df['filename'] == file, 'duration'].values[0]
            # Append the filename and duration to the batch
            batch_audio_duration.append( duration)

        (  scores_raw_node_pred,
           scores_postprocessed_node_pred,
            decoded_node_pred,
        ) = batched_node_decode_preds(
            node_predictions,
            filenames_synth,
            self.encoder,
            batch_indices,
            batch_dur=batch_audio_duration,
            thresholds=list(self.val_buffer_node_level.keys()),
            median_filter=self.hparams["training"]["median_window"],
            frames_per_node= self.encoder.frames_per_node
         )

        self.val_scores_postprocessed_buffer_node_level.update(
            scores_postprocessed_node_pred
        )
        for th in self.val_buffer_node_level.keys():
            self.val_buffer_node_level[th] = pd.concat(
                [self.val_buffer_node_level[th], decoded_node_pred[th]],
                ignore_index=True,
            )

        # print(f"\n *******the validation  step check  out")
        return

    # note, 计算各种类型的 分数；
    def validation_epoch_end(self, outputs):
        """Fonction applied at the end of all the validation steps of the epoch.

        Args:
            outputs: torch.Tensor, the concatenation of everything returned by validation_step.

        Returns:
            torch.Tensor, the objective metric to be used to choose the best model from for example.
        """  # 该函数在每个验证周期结束后执行，接收所有validation_step返回值的拼接结果。主要功能如下
        # 计算弱监督学生模型和教师模型的宏平均F1分数
        record_binary_f1_macro =self.record_level_bin_f1_seg_macro.compute()


        # train  dataset  读取valid 数据集的真实标签和音频时长。
        ground_truth = sed_scores_eval.io.read_ground_truth_events(
            self.hparams["data"]["valid_tsv"]
        )
        audio_durations = sed_scores_eval.io.read_audio_durations(
            self.hparams["data"]["valid_dur"]
        )


        if self.fast_dev_run: # 如果处于快速开发模式 (fast_dev_run)，则只保留部分音频数据；否则，移除没有事件的音频
            ground_truth = {
                audio_id: ground_truth[audio_id]
                for audio_id in self.val_scores_postprocessed_buffer_node_level
            }
            audio_durations = {
                audio_id: audio_durations[audio_id]
                for audio_id in self.val_scores_postprocessed_buffer_node_level
            }
        else:
            # drop audios without events
            ground_truth = {
                audio_id: gt for audio_id, gt in ground_truth.items() if len(gt) > 0
            }
            audio_durations = {
                audio_id: audio_durations[audio_id] for audio_id in ground_truth.keys()
            }



        #  1.计算学生模型的PSD分数 (psds1_node_level_sed_scores_eval)
        # psds1_node_level_sed_scores_eval = None
        psds1_node_level_sed_scores_eval = compute_psds_from_scores(
            self.val_scores_postprocessed_buffer_node_level,
            ground_truth,
            audio_durations,
            dtc_threshold= 0.7,  #0.7,
            gtc_threshold= 0.7,  #0.7,
            cttc_threshold=None,
            alpha_ct=0,
            alpha_st=1,
            save_dir=os.path.join(self.hparams["log_dir"], "Node_level", "scenario1"),
            # save_dir=os.path.join(save_dir, "student", "scenario1"),
        )
        
        # 2.计算学生模型的交集级宏平均F1 (intersection_f1_macro_node_level)。
        # intersection_f1_macro_node_level = None
        intersection_f1_macro_node_level =  compute_per_intersection_macro_f1(
            self.val_buffer_node_level,
            self.hparams["data"]["valid_tsv"],
            self.hparams["data"]["valid_dur"],
        )

        #   note,  计算class wise 事件级别的错误率；
        # class wise, 类别上 3. 计算学生模型的事件级别宏平均F1,  (class_wise_event_macro)。
        class_wise_event_macro = log_sedeval_metrics(
            self.val_buffer_node_level[self.hparams["training"]["val_thresholds"][0]],
            self.hparams["data"]["valid_tsv"],
            save_dir=os.path.join(self.hparams["log_dir"], "event_level_metrics", ),
        )[0]



        obj_metric_synth_type = self.hparams["training"].get("obj_metric_synth_type")
        if obj_metric_synth_type is None:
            synth_metric = psds1_node_level_sed_scores_eval
        elif obj_metric_synth_type == "event":
            synth_metric = class_wise_event_macro
        elif obj_metric_synth_type == "intersection":
            synth_metric = intersection_f1_macro_node_level
        elif obj_metric_synth_type == "psds":
            synth_metric = psds1_node_level_sed_scores_eval
        else:
            raise NotImplementedError(
                f"obj_metric_synth_type: {obj_metric_synth_type} not implemented."
            )

        # 使用 torch.tensor 创建最终的目标度量 (obj_metric)，
        # 包含弱监督学生模型的F1宏平均分数和选定的合成数据集度量。
        obj_metric = torch.tensor(record_binary_f1_macro.item() + synth_metric)

        # 将各种验证指标记录到日志中，并打印出来。
        print(
            f"\tval/obj_metric: {obj_metric}\n"
            f"\tval/record_level/binary_macro_F1: {record_binary_f1_macro}\n"
            f"\tval/node_level/psds1_sed_scores_eval: {psds1_node_level_sed_scores_eval}\n"
            f"\tval/node_level/intersection_f1_macro: {intersection_f1_macro_node_level}\n"
            f"\t val/node_level/event_f1_macro: {class_wise_event_macro}"
        )

        self.log("val/obj_metric", obj_metric, prog_bar=True)
        self.log("val/record_level/binary_macro_F1", record_binary_f1_macro)
        self.log("val/node_level/psds1_sed_scores_eval", psds1_node_level_sed_scores_eval )
        self.log("val/node_level/intersection_f1_macro", intersection_f1_macro_node_level)
        self.log("val/node_level/event_f1_macro", class_wise_event_macro, prog_bar=True)

        # free the buffers 清空多个缓冲区，释放内存空间
        self.val_buffer_node_level = {
            k: pd.DataFrame() for k in self.hparams["training"]["val_thresholds"]
        }

        self.val_scores_postprocessed_buffer_node_level = {}
        self.record_level_bin_f1_seg_macro.reset()  # 重置计算F1宏平均分数的指标。

        print(f"\n *******the validation epoch end   check  out")

        # Calculate the validation metric, for example, accuracy or loss
        val_score =  class_wise_event_macro # Placeholder for your actual validation metric calculation

        # Update the best model if validation score improves
        if val_score > self.best_val_score:
            self.best_val_score = val_score
            self.best_model_weights = deepcopy(self.sed_net.state_dict())
            self.best_ema_weights = deepcopy(self.ema_model.state_dict())

        self.log('best_val_classWise_score', self.best_val_score, prog_bar=True)

        return obj_metric  # 返回最终的目标度量 (obj_metric)，用于选择最佳模型。


    def on_before_zero_grad(self, *args, **kwargs) :
        self.update_ema(
            self.hparams["training"]["ema_factor"],
            self.scheduler["scheduler"].step_num,
        )


    def on_save_checkpoint(self, checkpoint):
        checkpoint["sed_net"] = self.sed_net.state_dict()  # Regular model weights
        checkpoint["ema_model"] = self.ema_model.state_dict()  # EMA weights

        # Save the best model weights if available
        if self.best_model_weights is not None:
            checkpoint["best_model"] = self.best_model_weights
        if self.best_ema_weights is not None:
            checkpoint["best_ema_model"] = self.best_ema_weights

        return checkpoint

    def test_step(self, batch, batch_indx):
        """Apply Test to a batch (step), used only when (trainer.test is called)

        Args:
            batch: torch.Tensor, input batch tensor
            batch_indx: torch.Tensor, 1D tensor of indexes to know which data are present in each batch.
        Returns:
        """

        audio, labels, padded_indxs, filenames, _ = batch

        # prediction for student
        mels = self.mel_spec(audio)
        strong_preds_student, weak_preds_student = self.detect(mels, self.EzhouNet)
        # prediction for teacher
        strong_preds_teacher, weak_preds_teacher = self.detect(mels, self.sed_teacher)

        if not self.evaluation:
            loss_strong_student = self.supervised_loss(strong_preds_student, labels)
            loss_strong_teacher = self.supervised_loss(strong_preds_teacher, labels)

            self.log("test/student/loss_strong", loss_strong_student)
            self.log("test/teacher/loss_strong", loss_strong_teacher)

        # compute psds
        (
            scores_raw_student_strong,
            scores_postprocesEzhouNet_strong,
            decoded_student_strong,
        ) = batched_node_decode_preds(
            strong_preds_student,
            filenames,
            self.encoder,
            median_filter=self.hparams["training"]["median_window"],
            thresholds=list(self.test_psds_buffer_student.keys()) + [0.5],
            frames_per_node=self.encoder.frames_per_node
        )

        self.test_scores_raw_buffer_student.update(scores_raw_student_strong)
        self.test_scores_postprocessed_buffer_student.update(
            scores_postprocesEzhouNet_strong
        )
        for th in self.test_psds_buffer_student.keys():
            self.test_psds_buffer_student[th] = pd.concat(
                [self.test_psds_buffer_student[th], decoded_student_strong[th]],
                ignore_index=True,
            )

        (
            scores_raw_teacher_strong,
            scores_postprocessed_teacher_strong,
            decoded_teacher_strong,
        ) = batched_decode_preds(
            strong_preds_teacher,
            filenames,
            self.encoder,
            median_filter=self.hparams["training"]["median_window"],
            thresholds=list(self.test_psds_buffer_teacher.keys()) + [0.5],
        )

        self.test_scores_raw_buffer_teacher.update(scores_raw_teacher_strong)
        self.test_scores_postprocessed_buffer_teacher.update(
            scores_postprocessed_teacher_strong
        )
        for th in self.test_psds_buffer_teacher.keys():
            self.test_psds_buffer_teacher[th] = pd.concat(
                [self.test_psds_buffer_teacher[th], decoded_teacher_strong[th]],
                ignore_index=True,
            )

        # compute f1 score
        self.decoded_student_05_buffer = pd.concat(
            [self.decoded_student_05_buffer, decoded_student_strong[0.5]]
        )
        self.decoded_teacher_05_buffer = pd.concat(
            [self.decoded_teacher_05_buffer, decoded_teacher_strong[0.5]]
        )

    def on_test_epoch_end(self):
        # pub eval dataset
        save_dir = os.path.join(self.exp_dir, "metrics_test")

        if self.evaluation:
            # only save prediction scores
            save_dir_student_raw = os.path.join(save_dir, "student_scores", "raw")
            sed_scores_eval.io.write_sed_scores(
                self.test_scores_raw_buffer_student, save_dir_student_raw
            )
            print(f"\nRaw scores for student saved in: {save_dir_student_raw}")

            save_dir_student_postprocessed = os.path.join(
                save_dir, "student_scores", "postprocessed"
            )
            sed_scores_eval.io.write_sed_scores(
                self.test_scores_postprocessed_buffer_student,
                save_dir_student_postprocessed,
            )
            print(
                f"\nPostprocessed scores for student saved in: {save_dir_student_postprocessed}"
            )

            save_dir_teacher_raw = os.path.join(save_dir, "teacher_scores", "raw")
            sed_scores_eval.io.write_sed_scores(
                self.test_scores_raw_buffer_teacher, save_dir_teacher_raw
            )
            print(f"\nRaw scores for teacher saved in: {save_dir_teacher_raw}")

            save_dir_teacher_postprocessed = os.path.join(
                save_dir, "teacher_scores", "postprocessed"
            )
            sed_scores_eval.io.write_sed_scores(
                self.test_scores_postprocessed_buffer_teacher,
                save_dir_teacher_postprocessed,
            )
            print(
                f"\nPostprocessed scores for teacher saved in: {save_dir_teacher_postprocessed}"
            )

            self.tracker_eval.stop()
            eval_kwh = self.tracker_eval._total_energy.kWh
            results = {"/eval/tot_energy_kWh": torch.tensor(float(eval_kwh))}
            with open(
                    os.path.join(self.exp_dir, "evaluation_codecarbon", "eval_tot_kwh.txt"),
                    "w",
            ) as f:
                f.write(str(eval_kwh))
        else:
            # calculate the metrics
            ground_truth = sed_scores_eval.io.read_ground_truth_events(
                self.hparams["data"]["test_tsv"]
            )
            audio_durations = sed_scores_eval.io.read_audio_durations(
                self.hparams["data"]["test_dur"]
            )
            if self.fast_dev_run:
                ground_truth = {
                    audio_id: ground_truth[audio_id]
                    for audio_id in self.test_scores_postprocessed_buffer_student
                }
                audio_durations = {
                    audio_id: audio_durations[audio_id]
                    for audio_id in self.test_scores_postprocessed_buffer_student
                }
            else:
                # drop audios without events
                ground_truth = {
                    audio_id: gt for audio_id, gt in ground_truth.items() if len(gt) > 0
                }
                audio_durations = {
                    audio_id: audio_durations[audio_id]
                    for audio_id in ground_truth.keys()
                }
            psds1_student_psds_eval = compute_psds_from_operating_points(
                self.test_psds_buffer_student,
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
                dtc_threshold=0.7,
                gtc_threshold=0.7,
                alpha_ct=0,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "student", "scenario1"),
            )
            psds1_node_level_sed_scores_eval = compute_psds_from_scores(
                self.test_scores_postprocessed_buffer_student,
                ground_truth,
                audio_durations,
                dtc_threshold=0.7,
                gtc_threshold=0.7,
                cttc_threshold=None,
                alpha_ct=0,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "student", "scenario1"),
            )

            psds2_student_psds_eval = compute_psds_from_operating_points(
                self.test_psds_buffer_student,
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
                dtc_threshold=0.1,
                gtc_threshold=0.1,
                cttc_threshold=0.3,
                alpha_ct=0.5,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "student", "scenario2"),
            )
            psds2_student_sed_scores_eval = compute_psds_from_scores(
                self.test_scores_postprocessed_buffer_student,
                ground_truth,
                audio_durations,
                dtc_threshold=0.1,
                gtc_threshold=0.1,
                cttc_threshold=0.3,
                alpha_ct=0.5,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "student", "scenario2"),
            )

            psds1_teacher_psds_eval = compute_psds_from_operating_points(
                self.test_psds_buffer_teacher,
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
                dtc_threshold=0.7,
                gtc_threshold=0.7,
                alpha_ct=0,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "teacher", "scenario1"),
            )
            psds1_teacher_sed_scores_eval = compute_psds_from_scores(
                self.test_scores_postprocessed_buffer_teacher,
                ground_truth,
                audio_durations,
                dtc_threshold=0.7,
                gtc_threshold=0.7,
                cttc_threshold=None,
                alpha_ct=0,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "teacher", "scenario1"),
            )

            psds2_teacher_psds_eval = compute_psds_from_operating_points(
                self.test_psds_buffer_teacher,
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
                dtc_threshold=0.1,
                gtc_threshold=0.1,
                cttc_threshold=0.3,
                alpha_ct=0.5,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "teacher", "scenario2"),
            )
            psds2_teacher_sed_scores_eval = compute_psds_from_scores(
                self.test_scores_postprocessed_buffer_teacher,
                ground_truth,
                audio_durations,
                dtc_threshold=0.1,
                gtc_threshold=0.1,
                cttc_threshold=0.3,
                alpha_ct=0.5,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "teacher", "scenario2"),
            )

            event_macro_student = log_sedeval_metrics(
                self.decoded_student_05_buffer,
                self.hparams["data"]["test_tsv"],
                os.path.join(save_dir, "student"),
            )[0]

            event_macro_teacher = log_sedeval_metrics(
                self.decoded_teacher_05_buffer,
                self.hparams["data"]["test_tsv"],
                os.path.join(save_dir, "teacher"),
            )[0]

            # synth dataset
            intersection_f1_macro_node_level = compute_per_intersection_macro_f1(
                {"0.5": self.decoded_student_05_buffer},
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
            )

            # synth dataset
            intersection_f1_macro_teacher = compute_per_intersection_macro_f1(
                {"0.5": self.decoded_teacher_05_buffer},
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
            )

            best_test_result = torch.tensor(
                max(psds1_student_psds_eval, psds2_student_psds_eval)
            )

            results = {
                "hp_metric": best_test_result,
                "test/student/psds1_psds_eval": psds1_student_psds_eval,
                "test/student/psds1_sed_scores_eval": psds1_node_level_sed_scores_eval,
                "test/student/psds2_psds_eval": psds2_student_psds_eval,
                "test/student/psds2_sed_scores_eval": psds2_student_sed_scores_eval,
                "test/teacher/psds1_psds_eval": psds1_teacher_psds_eval,
                "test/teacher/psds1_sed_scores_eval": psds1_teacher_sed_scores_eval,
                "test/teacher/psds2_psds_eval": psds2_teacher_psds_eval,
                "test/teacher/psds2_sed_scores_eval": psds2_teacher_sed_scores_eval,
                "test/student/event_f1_macro": event_macro_student,
                "test/student/intersection_f1_macro": intersection_f1_macro_node_level,
                "test/teacher/event_f1_macro": event_macro_teacher,
                "test/teacher/intersection_f1_macro": intersection_f1_macro_teacher,
            }
            self.tracker_devtest.stop()
            eval_kwh = self.tracker_devtest._total_energy.kWh
            results.update({"/test/tot_energy_kWh": torch.tensor(float(eval_kwh))})
            with open(
                    os.path.join(self.exp_dir, "devtest_codecarbon", "devtest_tot_kwh.txt"),
                    "w",
            ) as f:
                f.write(str(eval_kwh))

        if self.logger is not None:
            self.logger.log_metrics(results)
            self.logger.log_hyperparams(self.hparams, results)

        for key in results.keys():
            self.log(key, results[key], prog_bar=True, logger=True)

    def configure_optimizers(self):
        return [self.opt], [self.scheduler]

    # Custom collate function to convert spectrograms to Data objects during batch preparation


    # def custom_collate(self, spect_data):
    #     #cnn_extractor = DCNN(input_channels=3, output_features=64)
    #
    #     processed_batch = []
    #     batch_indices = []
    #     filenames = []
    #
    #     # for spectrogram, label, filename in spect_data:
    #     for batch_idx, (spectrogram, label, filename) in enumerate(spect_data):
    #         n_channels, n_frames, n_mels = spectrogram.shape
    #         spectrogram = spectrogram.unsqueeze(0)
    #         node_features = []
    #
    #         for i in range(0, n_frames, 5):
    #             if i + 5 > n_frames:  # If last chunk has less than 5 frames, take the last 5 frames
    #                 chunk = spectrogram[:, :, -5:, :]
    #             else:
    #                 chunk = spectrogram[:, :, i:i + 5, :]
    #             node_feature = cnn_extractor(chunk)
    #             # 先创建一个用于调试
    #             #node_feature = torch.rand(256).cuda()
    #             node_features.append(node_feature)
    #
    #
    #         node_features = torch.stack(node_features)  #(num_node,  fea_dim=256)
    #         num_nodes = node_features.shape[0]
    #
    #         # Add batch index for each node, so the GNN knows which sample each node belongs to
    #         batch_indices.extend([batch_idx] * num_nodes)
    #
    #         # Create edge index for nodes in the current sample
    #         # Edge index connects nodes sequentially in the graph
    #         edge_index = torch.stack([torch.arange(num_nodes - 1), torch.arange(1, num_nodes)], dim=0)
    #
    #         node_labels = []
    #         for i in range(0, n_frames, 5):
    #             if i + 5 > n_frames:  # If last chunk has less than 5 frames, take the last 5 frames
    #                 label_chunk = label[:,-5:]
    #             else:
    #                 label_chunk = label[:, i:i + 5]
    #             #node_label = self.get_node_label(label_chunk.numpy())
    #             node_label = self.get_node_label(label_chunk)
    #             node_labels.append(node_label)
    #
    #         node_labels = torch.tensor(node_labels)
    #         # Note,  训练过程中， 生成图数据中 Node 节点特征， 以及每个节点对应的标签；
    #         graph_data = Data(x=node_features, edge_index=edge_index, y=node_labels)
    #         processed_batch.append(graph_data)
    #         filenames.append(filename)
    #
    #     # Combine all individual graph samples into a batch
    #     batch = Batch.from_data_list(processed_batch)
    #     batch.batch = torch.tensor(batch_indices, dtype=torch.long)
    #
    #     return  batch, filenames

    def custom_collate(self, batch):
        """
        Avoid Padding Entire Spectrograms:

        instead of padding spectrograms to the maximum length,
        process them into fixed-size chunks directly in the __getitem__ method or in the collate_fn.
        不要将频谱图填充到最大长度，而是直接在__getitem__方法或collate_fn中将它们处理成固定大小的块。

        This reduces the need for padding and allows for more efficient batching.
        Args:
            batch:

        Returns:
            This modification eliminates padding in the collate_fn
            and moves the responsibility of handling variable lengths to the forward method.

        """

        # Prepare batch data
        batch_data = {
            'spectrograms':  [item['spectrogram'] for item in batch],
            'frame_labels':  [item['frame_labels'] for item in batch],
            'c_ex_mixtures': [item['c_ex_mixture'] for item in batch],
            # Include any other necessary data
        }

        return batch_data


    def train_dataloader(self):
        #self.train_loader = torch_geometric.data.DataLoader(
        self.train_loader = DataLoader(
            self.train_data,
            batch_sampler=self.train_sampler,
            batch_size= self.hparams["training"]["batch_size"] ,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn = self.custom_collate,
            pin_memory = True,
            #worker_init_fn=worker_init_fn
        )

        return self.train_loader


    def val_dataloader(self):
        #self.val_loader =  torch_geometric.data.DataLoader(
        self.val_loader = DataLoader(
            self.valid_data,
            batch_size=self.hparams["training"]["batch_size_val"],
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            collate_fn=self.custom_collate
        )



        return self.val_loader

    def test_dataloader(self):
        self.test_loader = DataLoader(
        #self.test_loader = torch_geometric.data.DataLoader(
            self.test_data,
            batch_size=self.hparams["training"]["batch_size_val"],
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            collate_fn=self.custom_collate
        )
        return self.test_loader


    # 要在训练结束或评估期间加载最佳模型，可以使用：
    def on_train_end(self) -> None:
        # dump consumption
        self.tracker_train.stop()
        training_kwh = self.tracker_train._total_energy.kWh
        self.logger.log_metrics(
            {"/train/tot_energy_kWh": torch.tensor(float(training_kwh))}
        )
        with open(
                os.path.join(self.exp_dir, "training_codecarbon", "training_tot_kwh.txt"),
                "w",
        ) as f:
            f.write(str(training_kwh))

        # Load the best weights for evaluation
        if self.best_model_weights is not None:
            self.sed_net.load_state_dict(self.best_model_weights)
        if self.best_ema_weights is not None:
            self.ema_model.load_state_dict(self.best_ema_weights)

    def on_test_start(self) -> None:
        if self.evaluation:
            os.makedirs(
                os.path.join(self.exp_dir, "evaluation_codecarbon"), exist_ok=True
            )
            self.tracker_eval = OfflineEmissionsTracker(
                "Respiratory sound SED EVALUATION",
                output_dir=os.path.join(self.exp_dir, "evaluation_codecarbon"),
                log_level="warning",
                country_iso_code="FRA",
            )
            self.tracker_eval.start()
        else:
            os.makedirs(os.path.join(self.exp_dir, "devtest_codecarbon"), exist_ok=True)
            self.tracker_devtest = OfflineEmissionsTracker(
                "Respiratory sound SED DEVTEST",
                output_dir=os.path.join(self.exp_dir, "devtest_codecarbon"),
                log_level="warning",
                country_iso_code="FRA",
            )
            self.tracker_devtest.start()




def worker_init_fn(worker_id):
    print(f"Initializing worker {worker_id}")

