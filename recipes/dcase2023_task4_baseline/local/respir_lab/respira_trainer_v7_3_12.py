import os
import random
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.optim import Optimizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

import sed_scores_eval
import torch
import  torch.nn  as  nn
import torch_geometric.data
import torchmetrics
from codecarbon import OfflineEmissionsTracker
from desed_task.data_augm import mixup
from desed_task.evaluation.evaluation_measures import (
    compute_per_intersection_macro_f1, compute_psds_from_operating_points,
    compute_psds_from_scores)
from desed_task.utils.scaler import TorchScaler
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from .utils import  batched_node_edge_decode_preds,  compute_event_based_metrics
from torch.utils.data import  DataLoader

from torch.utils.data import Dataset, Subset, ConcatDataset

# v6-5-1:    直接在节点 级别上实现 7 分类，  以及帧级别实现 7 分类；


# v7-1-1:  引入边学习， 以及一致性损失， 但此时节点分类， 与边分类仍然使用未经过一致性的优化的logit进行loss计算；

# v7-2-1:   使用 节点分类的  6 分类， 以及边分类的 2 分类；


from collections import OrderedDict
# Define class labels
# classes_labels = OrderedDict({
#     "Normal": 0,
#     "Rhonchi": 1,
#     "Wheeze": 2,
#     "Stridor": 3,
#     "Coarse Crackle": 4,
#     "Fine Crackle": 5,
#     "Wheeze+Crackle": 6,
# })



# classes_6labels = OrderedDict({
#     "Normal": 0,
#     "Rhonchi": 1,
#     "Wheeze": 2,
#     "Stridor": 3,
#     "Coarse Crackle": 4,
#     "Fine Crackle": 5,
# })


classes_5labels = OrderedDict({
    "Normal": 0,
    "Rhonchi": 1,
    "Wheeze": 2,
    "Stridor": 3,
    "Crackle": 4,
})



bin_labels = OrderedDict({
    "Normal": 0,
    "Abnormal": 1,
})




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
            train_flag = False,
            train_data=None,
            valid_data=None,
            test_data=None,
            fast_dev_run= False,
            evaluation= False,

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

        self.classes_labels = classes_5labels
        self.bin_labels = bin_labels
        self.train_data = train_data
        self.valid_data = valid_data

        self.test_data = test_data



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


        # note,  对于验证集不要使用加权损失， 用来评估模型的返回性能。
        # self.bin_cls_loss_weight = bin_cls_weight
        # self.cls_node_loss_fn = torch.nn.CrossEntropyLoss()
        # self.validation_cls_loss_fn = torch.nn.CrossEntropyLoss()



        # # 用于节点任务的 分类损失；
        # alpha_values = [1.0, 10.1, 1.4, 27.3, 14.2, 1.0, 37.6]
        # self.supervised_loss = FocalLoss(alpha=alpha_values, gamma=2,device="cuda")  # Adjust alpha values as needed
        # self.validation_supervised_loss = torch.nn.CrossEntropyLoss()   # Validation Loss Function (Unweighted Cross-Entropy Loss)

        # Define alpha coefficients based on training set
        # this  for sample 10 normal at each epoch;
        # alpha = [0.0214, 0.0935, 0.0261, 0.2236, 0.1673, 0.0128, 0.4609] # for 7 classes;
        """
        alpha = [
            0.0400,  # Normal
            0.1716,  # Rhonchi
            0.0477,  # Wheeze
            0.4102,  # Stridor
            0.3070,  # Coarse Crackle
            0.0235   # Fine Crackle
        ]
        """
        # alpha = [0.0400, 0.1716, 0.0477, 0.4102, 0.3070, 0.0235] # for 6 classes
        alpha = [0.0579, 0.2483, 0.0691, 0.5932, 0.0315]  # for 5 classes

        self.alpha = torch.tensor(alpha, dtype=torch.float32)
        self.cls_node_loss_fn = FocalLoss(alpha=self.alpha, gamma=2.0, reduction='mean')

        edge_alpha = [0.30, 0.70]  # for 2 classes
        self.edge_alpha = torch.tensor(edge_alpha, dtype=torch.float32)
        self.cls_edge_loss_fn = FocalLoss(alpha=self.edge_alpha, gamma=2.0, reduction='mean')




        #self.vad_loss_fn = nn.BCELoss()  # For node VAD loss,    主导损失， 先完成正常异常二分类
        # Compute pos_weight
        num_neg = 88613
        num_pos = 29682
        alpha_negative = 1.0
        alpha_positive = num_neg / num_pos  # ≈2.988
        vad_focal_alpha = [alpha_negative, alpha_positive]
        # Initialize Binary Focal Loss for VAD
        self.vad_loss_fn = BinaryFocalLoss(alpha=vad_focal_alpha, gamma=2.0, reduction='mean')


        self.frames_per_node = 5

        # Loss weight parameters
        # self.node_loss_weight_initial = 1.0
        self.node_loss_weight_initial = 10.0
        self.edge_loss_weight_initial = 1.0

        self.classification_loss_weight_final = 10.0
        self.edge_loss_weight_final = 1.0


        # Current loss weights (start with initial values)
        self.node_loss_weight = self.node_loss_weight_initial     #  特征属性
        self.edge_loss_weight = self.edge_loss_weight_initial    # 特征属性，

        self.node_vad_loss_weight = 3.0   # 时间属性




        # self.scaler = self._init_scaler()  # 该方法根据指定参数初始化用于数据标准化的缩放器。
        # buffer for event based scores which we compute using sed-eval
        # 为不同阈值创建多个 DataFrame 缓冲区，用于存储合成数据和测试数据的验证结果。
        self.val_buffer_node_level = {
            k: pd.DataFrame() for k in self.hparams["training"]["val_thresholds"]
        }

        #  创建一个异常的缓冲， 用于专门存储异常的 预测节点；
        self.val_buffer_abnormal_node= {  # 为不同阈值创建多个 DataFrame 缓冲区，用于存储合成数据和测试数据的验证结果。
            k: pd.DataFrame() for k in self.hparams["training"]["val_thresholds"]
        }

        #   这是有边生成的时间戳信息与 异常预测的时间戳信息，两者结合生成的 预测的时间戳信息。
        self.val_buffer_refine_pred = pd.DataFrame()  # Initialize as a single DataFrame


        self.val_scores_postprocessed_buffer_node_level = {}
        self.val_buffer_node_level_test = {
            k: pd.DataFrame() for k in self.hparams["training"]["val_thresholds"]
        }

        self.val_scores_postprocessed_buffer_frame_level = {}
        self.val_buffer_frame_level = {th: pd.DataFrame() for th in self.hparams["training"]["val_thresholds"]}
        self.val_buffer_abnormal_frame = {th: pd.DataFrame() for th in self.hparams["training"]["val_thresholds"]}


        # 为测试阶段的不同阈值创建多个 DataFrame 缓冲区。
        # 使用字典形式为学生和教师分别创建了不同阈值下的psds缓冲区，
        test_n_thresholds = self.hparams["training"]["n_test_thresholds"]
        # 该段代码首先使用numpy库生成一个数组test_thresholds，
        # 其中包含从1/(test_n_thresholds * 2)到1之间的等差数列，
        # 步长为1/test_n_thresholds。该数组用于定义测试阶段的不同阈值。
        test_thresholds = np.arange(
            1 / (test_n_thresholds * 2), 1, 1 / test_n_thresholds
        )
        self.test_psds_buffer_node_level = {k: pd.DataFrame() for k in test_thresholds}
        self.decoded_node_level_05_buffer = pd.DataFrame()
        self.test_scores_raw_buffer_node_level = {}
        self.test_scores_postprocessed_buffer_node_level = {}


        # 创建了四个空字典，用于存储学生和教师的原始分数、以及经过后处理的分数。
        # 创建额外的缓冲区用于存储解码后的结果和原始分数。
        # 这些字典可能后续会被用来保存不同测试的数据。
        self.best_val_score = float('-inf')  # Track the best validation score
        self.best_model_weights = None       # Placeholder for the best model weights (if needed)
        self.best_ema_weights = None
        self.best_node_f1 = float('-inf')

        # 通过这些步骤，该初始化方法为后续的训练、验证和测试过程准备了必要的组件和数据结构。

        self.best_training_f1 = float('-inf')

        self.epochs_no_improve = 0
        self.adjustment_started = False
        self.adjustment_epoch = None  # Epoch at which adjustment starts
        self.patience =  60  # Number of epochs to wait for improvement before starting adjustment
        self.min_delta = 2e-2  # Minimum change in validation accuracy to qualify as an improvement
        self.adjustment_duration = 80  # Number of epochs over which to adjust the weights

        # # Variables to track validation performance
        # self.best_validation_accuracy = float('-inf')
        # self.best_validation_f1 = float('-inf')


        # Initialize learning rates and final learning rates
        self.node_lr_initial = 1e-3
        self.edge_lr_initial = 1e-4

        self.node_lr_final = 1e-4
        self.edge_lr_final = 1e-5

        # self.shared_lr_initial = 1e-3  # Adjust if needed



        self.training_node_preds = []
        self.training_node_targets = []
        self.training_edge_preds = []
        self.training_edge_targets = []

        # Initialize storage for node label counts per epoch
        self.epoch_node_label_counts = torch.zeros(5, dtype=torch.long)
        self.valid_epoch_node_label_counts = torch.zeros(5, dtype=torch.long)


        self.train_epoch_edge_label_counts = torch.zeros(2, dtype=torch.long)
        self.valid_epoch_edge_label_counts = torch.zeros(2, dtype=torch.long)


        self.train_epoch_node_vad_label_counts = torch.zeros(2, dtype=torch.long)
        self.valid_epoch_node_vad_label_counts = torch.zeros(2, dtype=torch.long)


        self.validation_node_preds = []
        self.validation_node_targets = []

        self.validation_edge_preds = []
        self.validation_edge_targets = []

        # Initialize lists to store metrics for each epoch
        self.training_step_outputs = {
            'node_level_loss': [],
            'edge_level_loss': [],
            'node_vad_loss': [],
            'total_loss': []
        }


        self.valid_step_outputs = {
            'node_level_loss': [],
            'edge_level_loss': [],
            'node_vad_loss': [],
            'total_loss': []
        }



    _exp_dir = None

    @property
    def exp_dir(self):
        if self._exp_dir is None:
            try:
                self._exp_dir = self.logger.log_dir
            except Exception as e:
                self._exp_dir = self.hparams["log_dir"]
        return self._exp_dir

    # 原始代码中使用的是step 的方式, 即基于每个batch 来更新学习率，
    # 修改后的代码使用 epoch 后来更新学习率，  py lightning 默认是基于epoch 来更新scheduler,
    # 因此这里需要删除
    # def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
    #     scheduler.step()

    def setup(self, stage=None):
        # Assuming labels are stored or accessible
        normal_indices = []
        abnormal_indices = []

        for idx in range(len(self.train_data)):
            # Access the label for each sample
            # Modify this line according to how you retrieve the label
            id_name = self.train_data.examples_list[idx]
            sample_info =  self.train_data.examples[id_name]
            bin_label = sample_info['record_bin_label']  # Implement get_label method if needed
            if bin_label == "Normal":
                normal_indices.append(idx)
            else:
                abnormal_indices.append(idx)

        # Create Subsets
        self.normal_data = Subset(self.train_data, normal_indices)
        self.abnormal_data = Subset(self.train_data, abnormal_indices)


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


    def training_step(self, batch_data, batch_indx):
        """Apply the training for one batch (a step). Used during trainer.fit

        Args:
            batch_data: dict containing batched data
            batch_idx: int, index of the batch

        Returns:
            torch.Tensor, the loss to take into account.
        """

        # You can also print the model's device by checking one of the parameters
        device = next(self.sed_net.parameters()).device
        # print(f"Model is on device: {device}")

        # Move data to device
        spectrograms = [spectrogram.clone().detach().to(device) for spectrogram in batch_data['spectrograms']]
        frame_labels = [frame_label.clone().detach().to(device) for frame_label in batch_data['frame_labels']]
        record_binary_labels = torch.tensor(batch_data['record_binary_label'], dtype=torch.long).to(device)  # Ensure tensor type
        c_ex_mixtures = batch_data['c_ex_mixtures']  # List of audio names (metadata)

        chest_pos = [chest_pos.clone().detach().to(device) for chest_pos in batch_data['chest_info']]
        gender_info = [gender.clone().detach().to(device) for gender in batch_data['gender_info']]
        # vad_timestamps = [vad_time.clone().detach().to(device) for vad_time in batch_data['vad_timestamps']]
        vad_timestamps =  batch_data['vad_timestamps']

        # Prepare batch_data dictionary for the model
        batch_data_device = {
            'spectrograms': spectrograms,
            'frame_labels': frame_labels,
            'c_ex_mixtures': c_ex_mixtures,
            'vad_timestamps': vad_timestamps,
            'chest_loc': chest_pos,
            'genders': gender_info
        }


        ''' ====================== node   level  stage==================='''
        outputs_detection = self.sed_net(batch_data_device, level="node")

        node_predictions = outputs_detection['node_predictions']  # Shape: (total_nodes, num_classes)
        node_labels = outputs_detection['node_labels']  # Shape: (total_nodes,)
        node_pred_refine =  outputs_detection['node_pred_refine']

        # You need to have edge_labels similarly prepared in your forward pass, e.g., outputs_detection['edge_labels']
        # For demonstration, assume they exist:
        edge_predictions = outputs_detection['edge_predictions']  # (batch_edge_nums, edge_num_classes)
        edge_labels = outputs_detection['edge_labels']  # (batch_edge_nums)
        edge_pred_refine = outputs_detection['edge_pred_refine']



        node_vad_pred = outputs_detection['node_vad_logit']  # (E, edge_num_classes)
        node_vad_labels = outputs_detection['node_vad_falg']  # (batch_num_nodes,) # only 0,1 two value;


        batch_indices = outputs_detection['batch_indices']  # Shape: (total_nodes,)
        batch_audio_names = outputs_detection['batch_audio_names']  # List of audio names

        batch_edge_indices = outputs_detection['batch_edge_index']
        batch_graph_data =  outputs_detection['batch_graph']


        # Count occurrences of each label type in the batch
        batch_label_counts = torch.bincount(node_labels, minlength=5)
        self.epoch_node_label_counts += batch_label_counts.to(self.epoch_node_label_counts.device)


        edge_label_counts = torch.bincount(edge_labels, minlength=2)
        self.train_epoch_edge_label_counts += edge_label_counts.to(self.train_epoch_edge_label_counts.device)

        node_vad_labels = torch.tensor(node_vad_labels,)
        node_vad_label_counts = torch.bincount(node_vad_labels, minlength=2)
        self.train_epoch_node_vad_label_counts += node_vad_label_counts.to(self.train_epoch_node_vad_label_counts.device)





        # Identify abnormal and normal nodes
        # note, 训练阶段是提取真实值是异常类型的节点，
        abnormal_node_indices = (node_labels != 0).nonzero(as_tuple=True)[0]
        normal_node_indices = (node_labels == 0).nonzero(as_tuple=True)[0]

        # Determine number of normal nodes to sample (e.g., equal to the number of abnormal nodes)
        num_abnormal_nodes = len(abnormal_node_indices)
        num_normal_nodes_to_sample = max(64, num_abnormal_nodes // 3)  # Adjust denominator based on desired ratio

        # Sample normal nodes
        if len(normal_node_indices) > num_normal_nodes_to_sample:
            sampled_normal_indices = normal_node_indices[
                torch.randperm(len(normal_node_indices))[:num_normal_nodes_to_sample]]
        else:
            sampled_normal_indices = torch.tensor([], dtype=torch.long, device=device)

        # Combine indices
        node_indices_to_use = torch.cat((abnormal_node_indices, sampled_normal_indices))

        # Ensure indices are unique
        node_indices_to_use = node_indices_to_use.unique()

        # Extract predictions and labels for selected nodes
        node_predictions_selected = node_predictions[node_indices_to_use]
        node_labels_selected = node_labels[node_indices_to_use]

        # Compute node-level loss
        node_loss = self.cls_node_loss_fn(node_predictions_selected, node_labels_selected)
        node_level_loss = ( self.node_loss_weight *  node_loss  )

        # Compute edge  loss
        edge_loss = self.cls_edge_loss_fn(edge_predictions, edge_labels)
        edge_level_loss = ( self.edge_loss_weight * edge_loss  )

        # Convert to tensor
        node_vad_labels = torch.tensor(node_vad_labels, dtype=torch.float32)
        # Ensure predictions have the same shape as labels
        node_vad_pred = node_vad_pred.squeeze(-1)  # Shape: [1611]
        device = node_vad_pred.device  # Get the device of predictions
        node_vad_labels = node_vad_labels.to(device)  # Move labels to the same device
        node_vad_loss = self.vad_loss_fn(node_vad_pred, node_vad_labels)  * self.node_vad_loss_weight


        train_consistency_loss = self.compute_consistency_loss(node_pred_refine, edge_pred_refine, batch_graph_data,
                                                    consistency_loss_weight=1.0)

        # Accumulate node-level predictions and labels for confusion matrix
        self.training_node_preds.append(torch.argmax(node_predictions_selected, dim=1).cpu())
        self.training_node_targets.append(node_labels_selected.cpu())
        self.training_edge_preds.append(torch.argmax(edge_predictions, dim=1).cpu())
        self.training_edge_targets.append(edge_labels.cpu())

        # Total loss
        total_loss = (node_level_loss + edge_level_loss  +  train_consistency_loss +
                      node_vad_loss )

        # Log individual losses and total loss
        # Log losses
        self.log("train/node_level_loss", node_level_loss, prog_bar=True)
        self.log("train/edge_level_loss", edge_level_loss, prog_bar=True)
        self.log("train/node_vad_loss", node_vad_loss, prog_bar=True)
        self.log("train/consis_loss", train_consistency_loss, prog_bar=True)
        self.log("train/total_loss", total_loss, prog_bar=True)

        # Store values for epoch-end calculation
        self.training_step_outputs['node_level_loss'].append(node_level_loss)
        self.training_step_outputs['edge_level_loss'].append(edge_level_loss)
        self.training_step_outputs['node_vad_loss'].append(node_vad_loss)
        self.training_step_outputs['total_loss'].append(total_loss)

        # 记录日志信息：记录多个训练过程中重要的损失值，便于监控训练进度。
        # print(f"\n *******the train step check  out")
        return total_loss

    def on_before_zero_grad(self, *args, **kwargs) :
        # "Called per batch:
        # This hook is executed before optimizer.zero_grad() during training,
        # which happens after every batch."
        self.update_ema(
            self.hparams["training"]["ema_factor"],
            # self.scheduler["scheduler"].step_num, # 由于 scheduler 中设置的更新是 step , 代表每个batch 更新；
            self.global_step
        )

    def print_matrix_with_labels(self, matrix, class_labels, title):
        label_names = [name for name, _ in sorted(class_labels.items(), key=lambda item: item[1])]
        print(f"\n{title}:")
        header = " " * 15 + " ".join(f"{name:>15}" for name in label_names)
        print(header)
        for i, row in enumerate(matrix):
            row_string = f"{label_names[i]:15}" + " ".join(
                f"{val:15.4f}" if isinstance(val, float) else f"{val:15d}" for val in row)
            print(row_string)
    def training_epoch_end(self, outputs):
        '''Process and compute metrics at the end of each training epoch'''

        # Node-level metrics
        if len(self.training_node_preds) > 0:
            all_node_preds = torch.cat(self.training_node_preds).cpu().numpy()
            all_node_targets = torch.cat(self.training_node_targets).cpu().numpy()

            # Define labels based on your classes_labels dictionary
            labels = list(self.classes_labels.values())  # Ensure self.classes_labels is defined
            print(f"the labels {labels}")

            # Compute node-level confusion matrix
            node_cm = confusion_matrix(all_node_targets, all_node_preds, labels=labels)
            self.print_matrix_with_labels(node_cm, self.classes_labels, "Node-level Training Confusion Matrix")

            # Compute ratio format of the confusion matrix
            with np.errstate(divide='ignore', invalid='ignore'):
                node_cm_ratio = node_cm.astype('float') / node_cm.sum(axis=1)[:, np.newaxis]
                node_cm_ratio = np.nan_to_num(node_cm_ratio)  # Replace NaN with zero

            self.print_matrix_with_labels(node_cm_ratio, self.classes_labels,
                                          "Node-level Training Confusion Matrix (Ratio Format)")

            # Compute macro metrics
            node_precision = precision_score(all_node_targets, all_node_preds, average='macro', labels=labels, zero_division=0)
            node_recall = recall_score(all_node_targets, all_node_preds, average='macro', labels=labels, zero_division=0)
            node_f1_score = f1_score(all_node_targets, all_node_preds, average='macro', labels=labels, zero_division=0)
            print(f"\nNode-level Training "
                  f"\nMacro Metrics:"
                  f" F1 score: {node_f1_score:.4f}, Precision: {node_precision:.4f}, Recall: {node_recall:.4f}\n")

            # Compute weighted metrics
            weighted_precision = precision_score(all_node_targets, all_node_preds, average='weighted',  labels=labels, zero_division=0)
            weighted_recall = recall_score(all_node_targets, all_node_preds, average='weighted', labels=labels, zero_division=0)
            weighted_f1 = f1_score(all_node_targets, all_node_preds, average='weighted', labels=labels, zero_division=0)
            print(
                f"Weighted Metrics: Precision: {weighted_precision:.4f}, Recall: {weighted_recall:.4f}, F1 Score: {weighted_f1:.4f}\n")

            # Compute precision, recall, and F1 for each category
            category_precision = precision_score(all_node_targets, all_node_preds, labels=labels, average=None, zero_division=0)
            category_recall = recall_score(all_node_targets, all_node_preds, labels=labels, average=None, zero_division=0)
            category_f1 = f1_score(all_node_targets, all_node_preds, labels=labels, average=None, zero_division=0)

            # Print results for each category
            for i, label in enumerate(labels):
                print(
                    f"Category: {label}, Precision: {category_precision[i]:.4f}, Recall: {category_recall[i]:.4f}, F1: {category_f1[i]:.4f}")

            # Log macro and weighted metrics
            # self.log('train/node_precision_macro', node_precision)
            # self.log('train/node_recall_macro', node_recall)
            # self.log('train/node_f1_score_macro', node_f1_score)
            #
            # self.log('train/node_precision_weighted', weighted_precision)
            # self.log('train/node_recall_weighted', weighted_recall)
            # self.log('train/node_f1_score_weighted', weighted_f1)
            #
            # # Log per-category metrics
            # for i, label in enumerate(labels):
            #     self.log(f'train/precision_{label}', category_precision[i])
            #     self.log(f'train/recall_{label}', category_recall[i])
            #     self.log(f'train/f1_{label}', category_f1[i])

            # --- New Adjustment Logic ---
            # Check if node-level F1-score improved
            if node_f1_score > self.best_node_f1:
                self.best_node_f1 = node_f1_score

            # Start adjusting if node-level F1-score reaches 0.80
            if node_f1_score >= 0.88 and not self.adjustment_started:
                self.adjustment_started = True
                self.adjustment_epoch = self.current_epoch + 1  # Adjustments start next epoch
                print(f"\nStarting loss weight and learning rate adjustment at epoch {self.adjustment_epoch}")

        else:
            print("\n No node-level predictions to compute confusion matrix.")

        # Log the epoch-wide node label counts
        print("\n Training  epoch-wide node label counts")
        for i, count in enumerate(self.epoch_node_label_counts.tolist()):
            print(f"Node_label_{i} number: {count}")
            self.log(f'node_label_{i}_count', count)

        # Reset counts for the next epoch
        self.epoch_node_label_counts.zero_()


        print("\n Training  epoch-wide edge label counts")
        for i, count in enumerate(self.train_epoch_edge_label_counts.tolist()):
            print(f"Edge_label_{i} number: {count}")
            self.log(f'edge_label_{i}_count', count)
        # Reset counts for the next epoch
        self.train_epoch_edge_label_counts.zero_()


        print("\n Training  epoch-wide  Node vad label counts")
        for i, count in enumerate(self.train_epoch_node_vad_label_counts.tolist()):
            print(f"Node_vad_{i} number: {count}")
            self.log(f'node_vad_{i}_count', count)
        # Reset counts for the next epoch
        self.train_epoch_node_vad_label_counts.zero_()


        # Node-level metrics
        if len(self.training_edge_preds) > 0:
            all_edge_preds = torch.cat(self.training_edge_preds).cpu().numpy()
            all_edge_targets = torch.cat(self.training_edge_targets).cpu().numpy()

            # Define labels based on your classes_labels dictionary
            labels = list(self.bin_labels.values())  # Ensure self.classes_labels is defined

            # Compute edge-level confusion matrix
            edge_cm = confusion_matrix(all_edge_targets, all_edge_preds, labels=labels)
            self.print_matrix_with_labels(edge_cm, self.bin_labels, "edge-level Training Confusion Matrix")

            # Compute ratio format of the confusion matrix
            with np.errstate(divide='ignore', invalid='ignore'):
                edge_cm_ratio = edge_cm.astype('float') / edge_cm.sum(axis=1)[:, np.newaxis]
                edge_cm_ratio = np.nan_to_num(edge_cm_ratio)  # Replace NaN with zero

            self.print_matrix_with_labels(edge_cm_ratio, self.bin_labels,
                                          "edge-level Training Confusion Matrix (Ratio Format)")


            # Compute weighted metrics
            weighted_precision = precision_score(all_edge_targets, all_edge_preds, average='weighted',  labels=labels, zero_division=0)
            weighted_recall = recall_score(all_edge_targets, all_edge_preds, average='weighted', labels=labels, zero_division=0)
            weighted_f1 = f1_score(all_edge_targets, all_edge_preds, average='weighted', labels=labels, zero_division=0)
            print(
                f"Weighted Metrics: Precision: {weighted_precision:.4f}, Recall: {weighted_recall:.4f}, F1 Score: {weighted_f1:.4f}\n")



        # Calculate epoch statistics
        epoch_metrics = {
            key: torch.stack(values).mean().item()
            for key, values in self.training_step_outputs.items()
        }
        # self.training_step_outputs = {
        #     'node_level_loss': [],
        #     'edge_level_loss': [],
        #     'total_loss': []
        # }
        # Log epoch-level metrics
        self.log("train/node_loss/epoch", epoch_metrics['node_level_loss'], on_epoch=True)
        self.log("train/edge_loss/epoch", epoch_metrics['edge_level_loss'], on_epoch=True)
        self.log("train/node_vad_loss/epoch", epoch_metrics['node_vad_loss'], on_epoch=True)
        self.log("train/total_loss/epoch", epoch_metrics['total_loss'], on_epoch=True)


        # Clear the lists for next epoch
        self.training_node_preds = []
        self.training_node_targets = []
        self.training_edge_preds = []
        self.training_edge_targets = []


        # Clear lists for next epoch
        for key in self.training_step_outputs:
            self.training_step_outputs[key] = []



    def validation_step(self, batch_data, batch_indx):
        """Apply validation to a batch (step). Used during trainer.fit
        Args:
            batch: torch.Tensor, input batch tensor
            batch_indx: torch.Tensor, 1D tensor of indexes to know which data are present in each batch.
        Returns:
        """

        # You can also print the model's device by checking one of the parameters
        device = next(self.sed_net.parameters()).device
        # print(f"\nValidation Model is on device: {device}")

        # Move data to device
        spectrograms = [spectrogram.clone().detach().to(device) for spectrogram in batch_data['spectrograms']]
        frame_labels = [frame_label.clone().detach().to(device) for frame_label in batch_data['frame_labels']]
        record_binary_labels = torch.tensor(batch_data['record_binary_label'], dtype=torch.long).to(device)  # Ensure tensor type
        batch_audio_names = batch_data['c_ex_mixtures']  # List of audio names (metadata)


        chest_pos = [chest_pos.clone().detach().to(device) for chest_pos in batch_data['chest_info']]
        gender_info = [gender.clone().detach().to(device) for gender in batch_data['gender_info']]
        # vad_timestamps = [vad_time.clone().detach().to(device) for vad_time in batch_data['vad_timestamps']]
        vad_timestamps =  batch_data['vad_timestamps']

        # Prepare batch_data dictionary for the model
        batch_data_device = {
            'spectrograms': spectrograms,
            'frame_labels': frame_labels,

            'c_ex_mixtures':batch_audio_names,
            'vad_timestamps': vad_timestamps,

            'chest_loc': chest_pos,
            'genders': gender_info
        }


        ''' ====================== Node Level Stage ===================== '''
        # Perform detection at node level
        outputs_detection = self.sed_net(batch_data_device, level="node")

        node_predictions = outputs_detection['node_predictions']  # Shape: (total_nodes, num_classes)
        node_labels = outputs_detection['node_labels']  # Shape: (total_nodes,)
        node_pred_refine =  outputs_detection['node_pred_refine']

        # You need to have edge_labels similarly prepared in your forward pass, e.g., outputs_detection['edge_labels']
        # For demonstration, assume they exist:
        edge_predictions = outputs_detection['edge_predictions']  # (E, edge_num_classes)
        edge_labels = outputs_detection['edge_labels']  # (E,)
        edge_pred_refine = outputs_detection['edge_pred_refine']


        node_vad_pred = outputs_detection['node_vad_logit']  # (E, edge_num_classes)
        node_vad_labels = outputs_detection['node_vad_falg']  # (E,)


        batch_edge_indices = outputs_detection['batch_edge_index']
        batch_graph_data =  outputs_detection['batch_graph']

        batch_indices = outputs_detection['batch_indices']  # Shape: (total_nodes,)
        batch_detection_names = outputs_detection['batch_audio_names']  # List of audio names

        # Count occurrences of each label type in the batch
        batch_label_counts = torch.bincount(node_labels, minlength=5)
        # Update epoch-wide counts
        self.valid_epoch_node_label_counts += batch_label_counts.to(self.valid_epoch_node_label_counts.device)



        edge_label_counts = torch.bincount(edge_labels, minlength=2)
        self.valid_epoch_edge_label_counts += edge_label_counts.to(self.valid_epoch_edge_label_counts.device)

        node_vad_labels = torch.tensor(node_vad_labels,)
        node_vad_label_counts = torch.bincount(node_vad_labels, minlength=2)
        self.valid_epoch_node_vad_label_counts += node_vad_label_counts.to(self.valid_epoch_node_vad_label_counts.device)



        # print(f"node_predictions shape: {node_predictions.shape}")
        # Convert logits to predicted labels (0 or 1)
        edge_pred_labels =  torch.argmax(edge_predictions, dim=1)  # Shape: (E,)
        node_pred_labels = torch.argmax(node_predictions, dim=1)  # Get predicted node labels


        # Identify nodes predicted as abnormal
        predicted_abnormal_node_indices = (node_pred_labels != 0).nonzero(as_tuple=True)[0]
        normal_node_indices = (node_pred_labels == 0).nonzero(as_tuple=True)[0]

        # Sample normal nodes (always sample a fixed number if available)
        num_normal_nodes_to_sample = min(64, len(normal_node_indices))

        if len(normal_node_indices) > num_normal_nodes_to_sample:
            sampled_normal_indices = normal_node_indices[
                torch.randperm(len(normal_node_indices))[:num_normal_nodes_to_sample]
            ]
        else:
            sampled_normal_indices = normal_node_indices

        # Combine indices
        node_indices_to_use = torch.cat((predicted_abnormal_node_indices, sampled_normal_indices))
        node_indices_to_use = node_indices_to_use.unique()

        if len(node_indices_to_use) > 0:
            # Extract predictions and labels for selected nodes
            node_predictions_selected = node_predictions[node_indices_to_use]
            node_labels_selected = node_labels[node_indices_to_use]

            # Compute node-level loss
            self.valid_node_loss = self.cls_node_loss_fn(node_predictions_selected, node_labels_selected)

            # Accumulate node-level predictions and labels for confusion matrix
            self.validation_node_preds.append(torch.argmax(node_predictions_selected, dim=1).cpu())
            self.validation_node_targets.append(node_labels_selected.cpu())
        else:
            # No nodes to compute loss
            self.valid_node_loss = torch.tensor(0.0, device=device)


       # self.valid_frame_loss =  self.frame_loss_weight * self.valid_frame_loss
        self.valid_node_loss  =   self.node_loss_weight  *  self.valid_node_loss

        # Compute edge  loss
        edge_loss = self.cls_edge_loss_fn(edge_predictions, edge_labels)
        self.valid_edge_loss = ( self.edge_loss_weight *  edge_loss  )

        self.consistency_loss = self.compute_consistency_loss(node_pred_refine, edge_pred_refine, batch_graph_data,
                                                    consistency_loss_weight=1.0)

        node_vad_labels = torch.tensor(node_vad_labels, dtype=torch.float32)
        node_vad_pred = node_vad_pred.squeeze(-1)  # Shape: [1611]
        device = node_vad_pred.device  # Get the device of predictions
        node_vad_labels = node_vad_labels.to(device)  # Move labels to the same device
        self.valid_node_vad_loss = self.vad_loss_fn(node_vad_pred, node_vad_labels)  * self.node_vad_loss_weight
        # edge_vad_attr_pred = torch.tensor( edge_vad_attr_pred, dtype=torch.float32)


        self.valid_total_loss =   (self.valid_node_loss   + self.valid_edge_loss
                                   + self.consistency_loss
                                   + self.valid_node_vad_loss)


        # Log individual losses and total loss
        self.log("valid/node_loss", self.valid_node_loss, prog_bar=True)
        self.log("valid/edge_loss", self.valid_edge_loss, prog_bar=True)
        self.log("valid/node_vad_loss", self.valid_node_vad_loss, prog_bar=True)
        self.log("valid/consis_loss", self.consistency_loss, prog_bar=True)
        self.log("valid/total_loss", self.valid_total_loss, prog_bar=True)


        self.valid_step_outputs['node_level_loss'].append(self.valid_node_loss)
        self.valid_step_outputs['edge_level_loss'].append(self.valid_edge_loss)
        self.valid_step_outputs['node_vad_loss'].append(self.valid_node_vad_loss)
        self.valid_step_outputs['total_loss'].append( self.valid_total_loss)


        self.validation_edge_preds.append(torch.argmax(edge_predictions, dim=1).cpu())
        self.validation_edge_targets.append(edge_labels.cpu())

        # 初始化以下指标， 开始为计算 psd 分数， 交集 f1 score, 基于阈值的f score，
        # 这三种声音事件检测指标而做准备。
        # note, batch_audio_names  需要更新成 batch detection names， 表示只获取预测为异常样本的音频文件；
        filenames_synth = [ x  for x in  batch_detection_names
                        if Path(x).parent == Path(self.hparams["data"]["eval_folder_8k"])
                         ]

        # 获取当前batch 中每个音频的持续时间长度。
        batch_audio_duration = []
        batch_audio_timestamp = []

        valid_df = pd.read_csv(self.hparams["data"]["valid_dur"], sep='\t') # 验证集上所有的音频的持续时间；
        #valid_vad = pd.read_csv(self.hparams["data"]["valid_vad"], sep='\t')  # 验证集上所有的音频的vad 包含的时间戳信息；

        # Iterate over your list of filenames
        for file in filenames_synth:
            file = os.path.basename(file)
            # Find the row in the DataFrame that matches the filename
            duration = valid_df.loc[valid_df['filename'] == file, 'duration'].values[0]
            # Append the filename and duration to the batch
            batch_audio_duration.append( duration)
            sample_info = self.valid_data.examples[file]
            batch_audio_timestamp.append(sample_info['vad_timestamps'])

        # todo,  检查这里的node predictions, batch indices 是否对应上需要的数据；
        # 保留原始数据：scores_raw 保留了原始预测分数，方便后续调试和分析。
        # 提高准确性：scores_postprocessed 通过中值滤波提高了预测分数的稳定性，减少了噪声的影响。
        # 多阈值评估：prediction_dfs 提供了不同阈值下的预测结果，便于选择最佳阈值，提高预测性能。
        # 综上所述，这三种输出各有侧重，共同构成了一个完整的预测流程，确保了从原始数据到最终预测结果的全面覆盖。


        # Decoding outputs at frame level
        scores_postprocessed_node_pred, decoded_node_pred,decoded_abnormal_pred, refine_pred \
            = batched_node_edge_decode_preds(
            node_predictions,
            edge_pred_labels,
            filenames_synth,
            self.encoder,  #   note,  这里传入 异常类别的标签；
            batch_indices, #  此时的索引， 代表的是这些节点 位于当前预测为异常样本构成的 batch 中的索引；
            bt_edge_index= batch_edge_indices,
            batch_dur=batch_audio_duration,
            thresholds=list(self.val_buffer_node_level.keys()),
            median_filter=self.hparams["training"]["node_median_window"],
            frames_per_node= self.encoder.frames_per_node
         )


        self.val_scores_postprocessed_buffer_node_level.update(
            scores_postprocessed_node_pred)


        for th in self.val_buffer_node_level.keys():
            self.val_buffer_node_level[th] = pd.concat(
                [self.val_buffer_node_level[th], decoded_node_pred[th]],
                ignore_index=True,
            )


        for th in self.val_buffer_abnormal_node.keys():
            self.val_buffer_abnormal_node[th] = pd.concat(
                [self.val_buffer_abnormal_node[th], decoded_abnormal_pred[th]],
                ignore_index=True,
            )

        # Handling refined_pred as a single DataFrame
        self.val_buffer_refine_pred = pd.concat(
            [self.val_buffer_refine_pred, refine_pred], ignore_index=True
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
        """
        # print(f"\n Validation stage, current epoch: {self.current_epoch} ")

        # Node-level metrics
        if len(self.validation_node_preds) > 0:
            all_node_preds = torch.cat(self.validation_node_preds)
            all_node_targets = torch.cat(self.validation_node_targets)

            # Compute node-level confusion matrix
            labels = list(self.classes_labels.values())
            node_cm = confusion_matrix(all_node_targets.numpy(), all_node_preds.numpy(), labels=labels)
            self.print_matrix_with_labels(node_cm, self.classes_labels, "Node-level Validation Confusion Matrix")

            # Compute ratio format of the confusion matrix
            with np.errstate(divide='ignore', invalid='ignore'):
                node_cm_ratio = node_cm.astype('float') / node_cm.sum(axis=1)[:, np.newaxis]
                node_cm_ratio = np.nan_to_num(node_cm_ratio)  # Replace NaN with zero

            self.print_matrix_with_labels(node_cm_ratio, self.classes_labels,
                                     "Node-level Validation Confusion Matrix (Ratio Format)")

            # Compute precision, recall, F1 score
            node_precision = precision_score(all_node_targets.numpy(), all_node_preds.numpy(), average='macro',zero_division=0)
            node_recall = recall_score(all_node_targets.numpy(), all_node_preds.numpy(), average='macro',zero_division=0)
            node_f1_score = f1_score(all_node_targets.numpy(), all_node_preds.numpy(), average='macro',zero_division=0)
            print(f"\n Node-level Validation"
                  f" \nMacro Metrics:"
                  f" F1 score: {node_f1_score}, Precision: {node_precision}, Recall: {node_recall}\n")

            # Compute weighted metrics
            weighted_precision = precision_score(all_node_targets, all_node_preds, average='weighted', labels=labels, zero_division=0)
            weighted_recall = recall_score(all_node_targets, all_node_preds, average='weighted', labels=labels,  zero_division=0)
            weighted_f1 = f1_score(all_node_targets, all_node_preds, average='weighted', labels=labels, zero_division=0)
            print(
                f"Weighted Metrics: Precision: {weighted_precision:.4f}, Recall: {weighted_recall:.4f}, F1 Score: {weighted_f1:.4f}\n")

            # Compute precision, recall, and F1 for each category
            category_precision = precision_score(all_node_targets, all_node_preds, labels=labels, average=None, zero_division=0)
            category_recall = recall_score(all_node_targets, all_node_preds, labels=labels, average=None,zero_division=0)
            category_f1 = f1_score(all_node_targets, all_node_preds, labels=labels, average=None, zero_division=0)

            # Print results for each category
            for i, label in enumerate(labels):
                print(
                    f"Category: {label}, Precision: {category_precision[i]:.4f}, Recall: {category_recall[i]:.4f}, F1: {category_f1[i]:.4f}")

            # Log node-level metrics
            self.log('val/node_precision', node_precision)
            self.log('val/node_recall', node_recall)
            self.log('val/node_f1_score', node_f1_score)
        else:
            print("\n No node-level predictions to compute confusion matrix.")


        if len(self.validation_edge_preds) > 0:
            all_edge_preds = torch.cat(self.validation_edge_preds)
            all_edge_targets = torch.cat(self.validation_edge_targets)

            # Compute edge-level confusion matrix
            labels = list(bin_labels.values())
            edge_cm = confusion_matrix(all_edge_targets.numpy(), all_edge_preds.numpy(), labels=labels)
            self.print_matrix_with_labels(edge_cm, bin_labels, "edge-level Validation Confusion Matrix")

            # Compute ratio format of the confusion matrix
            with np.errstate(divide='ignore', invalid='ignore'):
                edge_cm_ratio = edge_cm.astype('float') / edge_cm.sum(axis=1)[:, np.newaxis]
                edge_cm_ratio = np.nan_to_num(edge_cm_ratio)  # Replace NaN with zero

            self.print_matrix_with_labels(edge_cm_ratio, bin_labels,
                                          "edge-level Validation Confusion Matrix (Ratio Format)")

            # Compute weighted metrics
            weighted_precision = precision_score(all_edge_targets, all_edge_preds, average='weighted', labels=labels, zero_division=0)
            weighted_recall = recall_score(all_edge_targets, all_edge_preds, average='weighted', labels=labels, zero_division=0)
            weighted_f1 = f1_score(all_edge_targets, all_edge_preds, average='weighted', labels=labels, zero_division=0)
            print(
                f"Weighted Metrics: Precision: {weighted_precision:.4f}, Recall: {weighted_recall:.4f}, F1 Score: {weighted_f1:.4f}\n")



        #  读取valid 数据集的真实标签和音频时长。
        # 更新后的函数可以用来建立空列表的事件， 由于空列表代表的是正常类型的record ,后续会将其过滤掉；

        DEFAULT_EVENT_SEGMENT_SCORED = (0.0,)  # Adjust based on expected structure
        # Compute node-level event-based metrics
        try:
            cur_thread = self.hparams["training"]["val_thresholds"][0]
            print(f"\n Using thread-based to compute,cur_thread:{cur_thread}")
            node_event_segment_scored = compute_event_based_metrics(
                self.val_buffer_abnormal_node[cur_thread],
                self.hparams["data"]["valid_tsv"],
                save_dir=os.path.join(self.hparams["log_dir"], "node_event_level_metrics"),
            )

            print("\n Using Edge_pred & combine thread-based to compute")
            node_combine_edge_scored = compute_event_based_metrics(
                self.val_buffer_refine_pred,
                self.hparams["data"]["valid_tsv"],
                save_dir=os.path.join(self.hparams["log_dir"], "node_event_level_metrics"),
            )


        except Exception as e:
            print(f"Error in compute_event_based_metrics for node level: {e}")
            node_event_segment_scored = DEFAULT_EVENT_SEGMENT_SCORED

        node_event_thread_f_score = node_event_segment_scored[0]

        # Update the objective metric
        obj_metric = torch.tensor( node_event_thread_f_score)

        # Update the best model if validation score improves
        val_score =  node_event_thread_f_score

        if val_score > self.best_val_score:
            self.best_val_score = val_score
            self.best_model_weights = deepcopy(self.sed_net.state_dict())
            self.best_ema_weights = deepcopy(self.ema_model.state_dict())

        self.log('best_val_class_wise_score', self.best_val_score, prog_bar=True)

        # Log the epoch-wide node label counts
        print("\n Validation  epoch-wide node label counts")
        for i, count in enumerate(self.valid_epoch_node_label_counts.tolist()):
            print(f"Node_label_{i} number: {count}")
            self.log(f'node_label_{i}_count', count)
        self.valid_epoch_node_label_counts.zero_()



        print("\n Validation  epoch-wide edge label counts")
        for i, count in enumerate(self.valid_epoch_edge_label_counts.tolist()):
            print(f"Edge_label_{i} number: {count}")
            self.log(f'edge_label_{i}_count', count)
        # Reset counts for the next epoch
        self.valid_epoch_edge_label_counts.zero_()


        print("\n Validation epoch-wide  Node vad label counts")
        for i, count in enumerate(self.valid_epoch_node_vad_label_counts.tolist()):
            print(f"Node_vad_{i} number: {count}")
            self.log(f'node_vad_{i}_count', count)
        # Reset counts for the next epoch
        self.valid_epoch_node_vad_label_counts.zero_()




        # Log validation metrics
        print(
            f"\tval/obj_metric: {obj_metric}"
            f"\tval/node_level/event_based_class_wise_average_f1_score: {node_event_thread_f_score}\n"
        )

        self.log("val/obj_metric", obj_metric, prog_bar=True)
        self.log("val/node_level/event_based_class_wise_average_f1_score", node_event_thread_f_score, prog_bar=True)

        
        
        
        # Calculate epoch statistics
        epoch_metrics = {
            key: torch.stack(values).mean().item()
            for key, values in self.valid_step_outputs.items()
        }

        # Log epoch-level metrics
        self.log("valid/node_loss/epoch", epoch_metrics['node_level_loss'], on_epoch=True)
        self.log("valid/edge_loss/epoch", epoch_metrics['edge_level_loss'], on_epoch=True)
        self.log("valid/node_vad_loss/epoch", epoch_metrics['node_vad_loss'], on_epoch=True)
        self.log("valid/total_loss/epoch", epoch_metrics['total_loss'], on_epoch=True)

        
        # Clear validation buffers for the next epoch
        self.validation_node_preds = []
        self.validation_node_targets = []
        self.validation_edge_preds = []
        self.validation_edge_targets = []


        # Clear buffers for node level
        self.val_scores_postprocessed_buffer_node_level = {}
        self.val_buffer_node_level = {th: pd.DataFrame() for th in self.val_buffer_node_level.keys()}
        self.val_buffer_abnormal_node = {th: pd.DataFrame() for th in self.val_buffer_abnormal_node.keys()}
        self.val_buffer_refine_pred = pd.DataFrame()

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
        ) = batched_node_edge_decode_preds(
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

            # event_macro_student = log_sedeval_metrics(
            #     self.decoded_student_05_buffer,
            #     self.hparams["data"]["test_tsv"],
            #     os.path.join(save_dir, "student"),
            # )[0]
            #
            # event_macro_teacher = log_sedeval_metrics(
            #     self.decoded_teacher_05_buffer,
            #     self.hparams["data"]["test_tsv"],
            #     os.path.join(save_dir, "teacher"),
            # )[0]

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

        node_learing_params = (
            list(self.sed_net.node_fea_generator.parameters()) +
            list(self.sed_net.node_fea_proj2_vad.parameters()) +
            list(self.sed_net.edge_encoder.parameters()) +
            list(self.sed_net.node_edge_cls.parameters()) +

            list(self.sed_net.factor_graph_layer.parameters())

        )

        ##### training params and optimizers ############
        # Create parameter groups with names
        optimizer = torch.optim.Adam([
            {'params': node_learing_params, 'lr': self.node_lr_initial, 'name': 'node_edge_cls'}, ])


        # Scheduler using LambdaLR
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=[
                lambda epoch: self.compute_lr_factor(epoch, 'node_edge_cls'), ]
        )

        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'epoch',  # Step the scheduler every epoch note,  此处表明学习率更新是按照epoch 执行sheduler， 如果是 step 则是按照batch 执行的；
            'frequency': 1,
            'name': 'custom_lr_scheduler',
        }


        self.optimizer = optimizer
        self.scheduler = scheduler_config

        for param_group in self.optimizer.param_groups:
            print(f"Parameter group: {param_group['name']}")
            for param in param_group['params']:
                if param.requires_grad:
                    print(f"  - {param.shape}")


        return [self.optimizer], [self.scheduler]

    def compute_lr_factor(self, epoch, group_name):
        # note, 对学习率进行调整；
        # Log the epoch and group name
        # print(f"\nComputing LR factor for group '{group_name}' at epoch {epoch}")
        epoch = self.current_epoch
        print(f"\n[Debug] Computing LR factor for group '{group_name}' at epoch {epoch}")
        # ... rest of your code ...

        if not self.adjustment_started or epoch < self.adjustment_epoch:
            # Before adjustment starts, keep initial learning rate
            return 1.0
        else:
            epochs_since_adjustment = epoch - self.adjustment_epoch + 1
            total_adjustment_epochs = self.adjustment_duration

            if epochs_since_adjustment >= total_adjustment_epochs:
                # After adjustment period, keep final learning rate
                if group_name == 'node_edge_cls':
                    lr_factor = self.node_lr_final / self.node_lr_initial
                elif group_name == 'edge_para':
                    lr_factor = self.edge_lr_final / self.edge_lr_initial
                else:
                    lr_factor = 1.0  # No adjustment
            else:
                # During adjustment, linearly interpolate learning rate
                adjustment_factor = epochs_since_adjustment / total_adjustment_epochs

                if group_name == 'node_edge_cls':
                    lr_initial = self.node_lr_initial
                    lr_final = self.node_lr_final
                elif group_name == 'edge_para':
                    lr_initial = self.edge_lr_initial
                    lr_final = self.edge_lr_final
                else:
                    return 1.0  # No adjustment

                # Linearly interpolate current_lr between initial and final
                current_lr = lr_initial + adjustment_factor * (lr_final - lr_initial)

                # Compute lr_factor as current_lr / lr_initial
                lr_factor = current_lr / lr_initial

            return lr_factor

    # Custom collate function to convert spectrograms to Data objects during batch preparation

    def compute_consistency_loss(self, node_logits_refined, edge_logits_refined, batch_graph, consistency_loss_weight=1.0):
        # Convert refined logits to predicted classes
        node_pred_classes_refined = torch.argmax(node_logits_refined, dim=-1)  # (N,)
        edge_pred_classes_refined = torch.argmax(edge_logits_refined, dim=-1)  # (E,)

        src, dst = batch_graph.edge_index

        # Identify abnormal edges
        abnormal_edges = (edge_pred_classes_refined != 0)
        # Identify normal nodes
        both_nodes_normal = (node_pred_classes_refined[src] == 0) & (node_pred_classes_refined[dst] == 0)

        # Inconsistency: edge abnormal but both connected nodes normal
        inconsistent_edges = abnormal_edges & both_nodes_normal
        consistency_loss = inconsistent_edges.float().mean() * consistency_loss_weight

        return consistency_loss

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

        print(f"\n Batch size: {len(batch)}")
        # Prepare batch data
        batch_data = {
            'spectrograms':  [item['spectrogram'] for item in batch],
            'frame_labels':  [item['frame_labels'] for item in batch],
            'c_ex_mixtures': [item['c_ex_mixture'] for item in batch],
            'record_binary_label':[item['record_binary_label'] for item in batch],
            "chest_info" :  [item['chest_pos'] for item in batch],
            "gender_info": [item['gender_info'] for item in batch],
            'vad_timestamps': [item['vad_timestamps'] for item in batch]
            # Include any other necessary data
        }

        return batch_data

    def train_dataloader(self):
        # Randomly select indices relative to the normal subset
        sampled_indices_in_subset = np.random.choice(len(self.normal_data), size=10, replace=False)
        # Map back to original indices in self.train_data
        sampled_normal_indices = [self.normal_data.indices[i] for i in sampled_indices_in_subset]
        # Create a new Subset with these original indices
        normal_subset = Subset(self.train_data, sampled_normal_indices)

        # Abnormal data remains the same
        abnormal_subset = self.abnormal_data

        # Combine the normal and abnormal subsets
        combined_dataset = ConcatDataset([normal_subset, abnormal_subset])

        # Create DataLoader
        train_loader = DataLoader(
            combined_dataset,
            batch_size=self.hparams["training"]["batch_size"],
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.custom_collate,
            pin_memory=True,
        )

        return train_loader

        return self.train_dataloader


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

    def on_train_epoch_start(self):

        # note,  对损失权重进行调整，
        current_epoch = self.current_epoch
        print(f"\n[Debug] Current Epoch in training loop: {current_epoch}")
        # ... rest of your code ...

        # Update loss weights if adjustment has started
        if self.adjustment_started:
            epochs_since_adjustment = self.current_epoch - self.adjustment_epoch + 1
            total_adjustment_epochs = self.adjustment_duration

            if epochs_since_adjustment >= total_adjustment_epochs:
                # Use final weights after adjustment period
                self.node_loss_weight = self.node_loss_weight_final
                self.detection_loss_weight = self.detection_loss_weight_final
            else:
                # Calculate adjustment factor (between 0 and 1)
                factor = epochs_since_adjustment / total_adjustment_epochs

                # Linearly interpolate between initial and final weights
                self.node_loss_weight = self.node_loss_weight_initial + factor * (
                        self.node_loss_weight_final - self.node_loss_weight_initial
                )
                self.detection_loss_weight = self.edge_loss_weight_initial + factor * (
                        self.detection_loss_weight_final - self.edge_loss_weight_initial
                )

            # Log the updated loss weights
            self.log('loss_weights/node', self.node_loss_weight_initial)
            self.log('loss_weights/detection', self.detection_loss_weight)

        # Adjust learning rates when adjustment starts, in the  compute_lr_function

        # Log learning rates
        optimizer = self.optimizers()
        if not isinstance(optimizer, list):
            optimizer = [optimizer]
        for opt in optimizer:
            for i, param_group in enumerate(opt.param_groups):
                lr = param_group['lr']
                group_name = param_group.get('name', f'group_{i}')
                self.log(f'lr/{group_name}', lr, prog_bar=True)


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

import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        if self.alpha is not None:
            if isinstance(self.alpha, (list, torch.Tensor)):
                self.alpha = torch.tensor(self.alpha, dtype=torch.float32)
            else:
                self.alpha = torch.tensor([alpha], dtype=torch.float32)

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)  # Shape: (batch_size, num_classes)
        probs = torch.exp(log_probs)  # Shape: (batch_size, num_classes)

        targets = targets.view(-1, 1)  # Shape: (batch_size, 1)
        log_probs = log_probs.gather(1, targets)  # Shape: (batch_size, 1)
        probs = probs.gather(1, targets)  # Shape: (batch_size, 1)

        if self.alpha is not None:
            self.alpha = self.alpha.to(inputs.device)
            alpha = self.alpha.gather(0, targets.squeeze())  # Shape: (batch_size,)
        else:
            alpha = 1.0

        loss = -alpha * (1 - probs) ** self.gamma * log_probs.squeeze()

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # 'none'


#  多分类（包括二分类） 与 二进制分类的区别是
# 每个样本至少会有2个以及以上的 logit 数值；
# 而 二进制分类，  每个样本只存在一个 logit 数值；
#   可以理解为binary cls 本质上就是单类别，
#   一个类别的分类即判断正负
# Define Binary Focal Loss
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

        if isinstance(alpha, (list, torch.Tensor)):
            if len(alpha) != 2:
                raise ValueError("Alpha list must have exactly two elements for binary classification.")
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = torch.tensor([1.0, alpha], dtype=torch.float32)  # [negative, positive]

    def forward(self, inputs, targets):
        if inputs.dim() > 1:
            inputs = inputs.view(-1)
        if targets.dim() > 1:
            targets = targets.view(-1)

        targets = targets.float()
        probs = torch.sigmoid(inputs)

        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_factor = (1 - pt) ** self.gamma

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha = torch.where(targets == 1, self.alpha[1], self.alpha[0])
            BCE_loss = alpha * BCE_loss

        focal_loss = focal_factor * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss  # 'none'


def worker_init_fn(worker_id):
    print(f"Initializing worker {worker_id}")

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
import numpy as np

# def calculate_class_weights(dataset):
#     all_labels = []
#     for sample in dataset.examples.values():
#         labels_df = sample['label_df']
#         if not labels_df.empty:
#             labels = labels_df['event_label'].tolist()
#             all_labels.extend(labels)
#     class_counts = Counter(all_labels)
#     total_counts = sum(class_counts.values())
#     class_weights = {cls: total_counts / count for cls, count in class_counts.items()}
#     # Optionally normalize the class weights
#     sum_weights = sum(class_weights.values())
#     class_weights = {cls: weight / sum_weights for cls, weight in class_weights.items()}
#     return class_weights



# def print_matrix_with_alignment(matrix, title):
#     print(f"\n{title}:")
#     for row in matrix:
#         print(" ".join(f"{val:10.4f}" if isinstance(val, float) else f"{val:10d}" for val in row))

# def calculate_binary_class_weights(dataset):
#     all_labels = []
#     for sample in dataset.examples.values():
#         #print(sample)
#         cur_binary_label = sample['record_bin_label'] #.tolist()
#         #all_labels.extend(cur_binary_label)
#         all_labels.append(cur_binary_label)
#     class_counts = Counter(all_labels)
#     total_counts = sum(class_counts.values())
#     class_weights = {cls: total_counts / count for cls, count in class_counts.items()}
#     # Optionally normalize the class weights
#     sum_weights = sum(class_weights.values())
#     class_weights = {cls: weight / sum_weights for cls, weight in class_weights.items()}
#     print(f"The binary class  sample weights and for the loss weight \n {class_weights}")
#     return class_weights
