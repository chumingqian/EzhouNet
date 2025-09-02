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
from .utils import  batched_node_decode_preds,  compute_event_based_metrics, compute_event_based_metrics_ori
from torch.utils.data import  DataLoader


# v6-5-1:    直接在节点 级别上实现 7 分类，  以及帧级别实现 7 分类；





from collections import OrderedDict
# Define class labels
classes_labels = OrderedDict({
    "Normal": 0,
    "Rhonchi": 1,
    "Wheeze": 2,
    "Stridor": 3,
    "Coarse Crackle": 4,
    "Fine Crackle": 5,
    "Wheeze+Crackle": 6,
})

def print_matrix_with_labels(matrix, class_labels, title):
    label_names = [name for name, _ in sorted(class_labels.items(), key=lambda item: item[1])]
    print(f"\n{title}:")
    header = " " * 15 + " ".join(f"{name:>15}" for name in label_names)
    print(header)
    for i, row in enumerate(matrix):
        row_string = f"{label_names[i]:15}" + " ".join(f"{val:15.4f}" if isinstance(val, float) else f"{val:15d}" for val in row)
        print(row_string)





class FBetaLoss(nn.Module):  # 用于 record 级别的分类损失
    def __init__(self, beta=1.0, eps=1e-7):
        super(FBetaLoss, self).__init__()
        self.beta = beta
        self.eps = eps

    def forward(self, logits, targets):
        """
        logits: Tensor of shape (batch_size, num_classes)
        targets: Tensor of shape (batch_size,), with class indices
        - **FBetaLoss**: The `FBetaLoss` class computes a differentiable approximation of the F-beta score,
        which combines precision and recall.**FBetaLoss** ：
         `FBetaLoss`类计算 F-beta 分数的可微近似值，它结合了精度和召回率。

        - **Loss Computation**: In the `forward` method of `FBetaLoss`,
         we compute precision and recall using soft predictions (`probs`)
         and one-hot encoded targets.**损失计算**：
         在`FBetaLoss`的`forward`方法中，我们使用软预测（ `probs` ）和 one-hot 编码目标来计算精度和召回率。

        - **Loss Value**: Since we want to maximize the F-beta score,
        we minimize `1 - F-beta` as the loss.

        """
        num_classes = logits.size(1)
        targets_one_hot = nn.functional.one_hot(targets, num_classes=num_classes).float()

        probs = torch.softmax(logits, dim=1)
        true_positive = (probs * targets_one_hot).sum(dim=0)
        predicted_positive = probs.sum(dim=0)
        actual_positive = targets_one_hot.sum(dim=0)

        precision = true_positive / (predicted_positive + self.eps)
        recall = true_positive / (actual_positive + self.eps)

        beta_sq = self.beta ** 2
        fbeta = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall + self.eps)
        fbeta = fbeta.clamp(min=self.eps, max=1 - self.eps)  # Avoid numerical issues

        # We want to maximize F-beta, so we minimize (1 - F-beta)
        loss = 1 - fbeta.mean()
        return loss

#
# import torch
# import torch.nn as nn


# v6-1, 更新损失权重为2分类的 损失权重；
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
            bin_cls_weight = None,
            scheduler=None,
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


        self.train_data = train_data
        self.valid_data = valid_data

        self.test_data = test_data
        self.train_sampler = train_sampler

        # self.optimizer = opt
        # self.scheduler = scheduler

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
        self.bin_cls_loss_weight = bin_cls_weight
        self.cls_node_loss_fn = torch.nn.CrossEntropyLoss(weight=self.bin_cls_loss_weight)
        self.validation_cls_loss_fn = torch.nn.CrossEntropyLoss()

        self.fbeta_loss_fn = FBetaLoss(beta=1.0)


        # 用于节点任务的 分类损失；
        alpha_values = [1.0, 10.1, 1.4, 27.3, 14.2, 1.0, 37.6]
        self.supervised_loss = FocalLoss(alpha=alpha_values, gamma=2,device="cuda")  # Adjust alpha values as needed
        self.validation_supervised_loss = torch.nn.CrossEntropyLoss()   # Validation Loss Function (Unweighted Cross-Entropy Loss)


        self.frames_per_node = 5



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
        self.best_ema_weights = None
        self.best_node_f1 = float('-inf')

        # 通过这些步骤，该初始化方法为后续的训练、验证和测试过程准备了必要的组件和数据结构。

        self.best_record_f1 = float('-inf')

        self.epochs_no_improve = 0
        self.adjustment_started = False
        self.adjustment_epoch = None  # Epoch at which adjustment starts
        # self.patience =  60  # Number of epochs to wait for improvement before starting adjustment
        self.min_delta = 2e-2  # Minimum change in validation accuracy to qualify as an improvement
        self.adjustment_duration = 80  # Number of epochs over which to adjust the weights

        # # Variables to track validation performance
        # self.best_validation_accuracy = float('-inf')
        # self.best_validation_f1 = float('-inf')


        # Initialize learning rates and final learning rates
        self.record_lr_initial = 1e-3
        self.node_lr_initial = 1e-3

        self.record_lr_final = 1e-5
        self.node_lr_final = 1e-4


        # Loss weight parameters
        self.record_loss_weight_initial = 1.2
        self.node_loss_weight_initial = 1.5

        self.record_loss_weight_final = 1.0
        self.node_loss_weight_final = 1.0

        # Current loss weights (start with initial values)
        self.record_loss_weight = self.record_loss_weight_initial
        self.node_loss_weight = self.node_loss_weight_initial


        self.training_bin_preds = []
        self.training_bin_targets = []   
        self.training_node_preds = []
        self.training_node_targets = []


        self.validation_bin_preds = []
        self.validation_bin_targets = []
        self.validation_node_preds = []
        self.validation_node_targets = []

        # Initialize lists to store metrics for each epoch
        self.training_step_outputs = {
            'bin_cls_loss': [],
            'node_loss': [],
            'node_level_loss': [],
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

        # record_binary_labels = batch_data['record_binary_label']
        record_binary_labels = torch.tensor(batch_data['record_binary_label'], dtype=torch.long).to(device)  # Ensure tensor type
        c_ex_mixtures = batch_data['c_ex_mixtures']  # List of audio names (metadata)


        # Prepare batch_data dictionary for the model
        batch_data_device = {
            'spectrograms': spectrograms,
            'frame_labels': frame_labels,
            'c_ex_mixtures': c_ex_mixtures,
        }
        

        ''' ======================  record   level  binary cls stage===================''' 
        #  完成样本级别的 二分类任务；
        # 1. Obtain actual abnormal samples in the current batch
        actual_abnormal_indices = [i for i, label in enumerate(record_binary_labels) if label == 1]
        actual_abnormal_names = [c_ex_mixtures[i] for i in actual_abnormal_indices]

        # 2. Forward pass for classification
        outputs_classification = self.sed_net(batch_data_device, level="record")
        record_binary_predictions = outputs_classification[
            'record_binary_predictions']  # Shape: (batch_size, num_classes)

        # Compute classification losses , 二分类损失
        record_binary_loss_ce = self.cls_node_loss_fn(record_binary_predictions, record_binary_labels)

        #  FP, Np, 误报和漏报构成的损失；
        record_binary_loss_fbeta = self.fbeta_loss_fn(record_binary_predictions, record_binary_labels)

        binary_cls_loss = (
                self.record_loss_weight * record_binary_loss_ce +
                self.record_loss_weight * record_binary_loss_fbeta
             
        )

        # Get predicted labels
        record_binary_pred_labels = torch.argmax(record_binary_predictions, dim=1)  # Shape: (batch_size,)

        # Accumulate predictions and labels for confusion matrix
        self.training_bin_preds.append(record_binary_pred_labels.cpu())
        self.training_bin_targets.append(record_binary_labels.cpu())




        ''' ====================== node   level  stage==================='''
        if len(actual_abnormal_indices) > 0: #  训练阶段，提取当前batch 中实际值 是异常类型的样本。
                # Case 1: There are samples with real abnormal labels
                # Form a new batch with these samples
                selected_indices = actual_abnormal_indices
    
                # Extract spectrograms and frame_labels for selected samples
                spectrograms_selected = [spectrograms[i] for i in selected_indices]
                frame_labels_selected = [frame_labels[i] for i in selected_indices]
                c_ex_mixtures_selected = [c_ex_mixtures[i] for i in selected_indices]
    
                # Prepare batch_data for detection stage
                batch_abnormal_data = {
                    'spectrograms': spectrograms_selected,
                    'frame_labels': frame_labels_selected,
                    'c_ex_mixtures': c_ex_mixtures_selected,
                }
            
                outputs_detection = self.sed_net(batch_abnormal_data, level="node")
                node_predictions = outputs_detection['node_predictions']  # Shape: (total_nodes, num_classes)
                node_labels = outputs_detection['node_labels']  # Shape: (total_nodes,)
                batch_indices = outputs_detection['batch_indices']  # Shape: (total_nodes,)
                batch_audio_names = outputs_detection['batch_audio_names']  # List of audio names
        
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
                node_loss = self.supervised_loss(node_predictions_selected, node_labels_selected)
                # node_level_loss = ( self.node_loss_weight *  node_loss  )
        
                # Accumulate node-level predictions and labels for confusion matrix
                self.training_node_preds.append(torch.argmax(node_predictions_selected, dim=1).cpu())
                self.training_node_targets.append(node_labels_selected.cpu())
                # total_loss = node_level_loss + binary_cls_loss
                #
                # self.log("train/bin_cls_loss", binary_cls_loss)
                # self.log("train/node_loss", node_loss)
                # self.log("train/node_level_loss", node_level_loss)
                # self.log("train/total_loss", total_loss)
                #
                # # Store values for epoch-end calculation
                # self.training_step_outputs['node_loss'].append(node_loss)
                # self.training_step_outputs['bin_cls_loss'].append(binary_cls_loss)
                # self.training_step_outputs['node_level_loss'].append(node_level_loss)
                # # self.training_step_outputs['frame_level_loss'].append(frame_level_loss)
                # self.training_step_outputs['total_loss'].append(total_loss)
        else:
                print('\n current batch No abnormal samples')
                # Case 2: No samples with real abnormal labels
                # Use samples predicted as abnormal by the model
                predicted_abnormal_indices = (record_binary_pred_labels == 1).nonzero(as_tuple=True)[0].tolist()
        
                # Extract spectrograms and frame_labels for selected samples
                spectrograms_selected = [spectrograms[i] for i in predicted_abnormal_indices]
                frame_labels_selected = [frame_labels[i] for i in predicted_abnormal_indices]
                c_ex_mixtures_selected = [c_ex_mixtures[i] for i in predicted_abnormal_indices]
        
                if len(predicted_abnormal_indices) == 0:
                    # No samples predicted as abnormal
                    node_loss = torch.tensor(0.0, device=device)
                else:
                    # Prepare batch_data for detection stage
                    batch_predicated_abnormal = {
                        'spectrograms': spectrograms_selected,
                        'frame_labels': frame_labels_selected,
                        'c_ex_mixtures': c_ex_mixtures_selected,
                    }

                    outputs_detection = self.sed_net(batch_predicated_abnormal, level="node")
                    node_predictions = outputs_detection['node_predictions']  # Shape: (total_nodes, num_classes)
                    node_labels = outputs_detection['node_labels']  # Shape: (total_nodes,)
                    batch_indices = outputs_detection['batch_indices']  # Shape: (total_nodes,)
                    batch_audio_names = outputs_detection['batch_audio_names']  # List of audio names
    
                    #由于此时当前batch 中， 实际上不存在异常样本，因此也应存在异常节点， 故此时随机提取固定数目的节点，进行损失计算；
                    # Randomly select a batch of nodes
                    total_nodes = node_predictions.shape[0]
                    num_nodes_to_sample = min(64, total_nodes)  # Adjust the number as needed
    
                    # Randomly sample nodes
                    node_indices_to_use = torch.randperm(total_nodes)[:num_nodes_to_sample]
    
                    # Extract predictions and labels for selected nodes
                    node_predictions_selected = node_predictions[node_indices_to_use]
                    node_labels_selected = node_labels[node_indices_to_use]
    
                    # Compute node-level loss
                    node_loss = self.supervised_loss(node_predictions_selected, node_labels_selected)
                    #node_level_loss = (self.node_loss_weight * node_loss)
    
                    # Accumulate node-level predictions and labels for confusion matrix
                    self.training_node_preds.append(torch.argmax(node_predictions_selected, dim=1).cpu())
                    self.training_node_targets.append(node_labels_selected.cpu())

        node_level_loss = (self.node_loss_weight * node_loss)
        # Total loss
        total_loss = node_level_loss + binary_cls_loss


        # Log individual losses and total loss
        # Log losses
        self.log("train/bin_cls_loss",  binary_cls_loss)
        self.log("train/node_loss", node_loss)
        self.log("train/node_level_loss", node_level_loss)
        self.log("train/total_loss", total_loss)


        # Store values for epoch-end calculation
        self.training_step_outputs['node_loss'].append(node_loss)
        self.training_step_outputs['bin_cls_loss'].append(binary_cls_loss)
        self.training_step_outputs['node_level_loss'].append(node_level_loss)
        # self.training_step_outputs['frame_level_loss'].append(frame_level_loss)
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


    def training_epoch_end(self, outputs):
        '''Process and compute metrics at the end of each training epoch'''

        # Frame-level metrics
        if len(self.training_bin_preds) > 0:
            all_preds = torch.cat(self.training_bin_preds)
            all_targets = torch.cat(self.training_bin_targets)

            # Compute confusion matrix
            cm = confusion_matrix(all_targets.numpy(), all_preds.numpy())
            print(f"\n Training Confusion Matrix:\n{cm}")

            # Compute f score  metrics
            precision = precision_score(all_targets.numpy(), all_preds.numpy(), average='macro')
            recall = recall_score(all_targets.numpy(), all_preds.numpy(), average='macro')
            record_f1_score = f1_score(all_targets.numpy(), all_preds.numpy(), average='macro')
            print(f"Training, for binary classification, using skearn method to compute:\n"
                  f" F1 score: {record_f1_score}, Precision: {precision}, Recall: {recall}\n")


        else:
            print("\n No Record-level binary predictions to compute confusion matrix.")

        # Node-level metrics
        if len(self.training_node_preds) > 0:
            all_node_preds = torch.cat(self.training_node_preds)
            all_node_targets = torch.cat(self.training_node_targets)

            # Compute node-level confusion matrix
            labels = list(classes_labels.values())
            node_cm = confusion_matrix(all_node_targets.numpy(), all_node_preds.numpy(), labels=labels)
            print_matrix_with_labels(node_cm, classes_labels, "Node-level Training Confusion Matrix")

            # Compute ratio format of the confusion matrix
            with np.errstate(divide='ignore', invalid='ignore'):
                node_cm_ratio = node_cm.astype('float') / node_cm.sum(axis=1)[:, np.newaxis]
                node_cm_ratio = np.nan_to_num(node_cm_ratio)  # Replace NaN with zero

            print_matrix_with_labels(node_cm_ratio, classes_labels,
                                     "Node-level Training Confusion Matrix (Ratio Format)")

            # Compute precision, recall, F1 score
            node_precision = precision_score(all_node_targets.numpy(), all_node_preds.numpy(), average='macro')
            node_recall = recall_score(all_node_targets.numpy(), all_node_preds.numpy(), average='macro')
            node_f1_score = f1_score(all_node_targets.numpy(), all_node_preds.numpy(), average='macro')
            print(f"Node-level Training Metrics:\n"
                  f" F1 score: {node_f1_score}, Precision: {node_precision}, Recall: {node_recall}\n")

            # Log node-level metrics
            self.log('train/node_precision', node_precision)
            self.log('train/node_recall', node_recall)
            self.log('train/node_f1_score', node_f1_score)

        else:
            print("\n No node-level predictions to compute confusion matrix.")
            
            
        # --- New Adjustment Logic ---
        # Check if node-level F1-score improved
        if record_f1_score > self.best_record_f1:
            self.best_node_f1 = record_f1_score

        # Start adjusting if node-level F1-score reaches 0.80
        if record_f1_score >= 0.96 and not self.adjustment_started:
            self.adjustment_started = True
            self.adjustment_epoch = self.current_epoch + 1  # Adjustments start next epoch
            print(f"\nStarting loss weight and learning rate adjustment at epoch {self.adjustment_epoch}")


        # Calculate epoch statistics
        epoch_metrics = {
            key: torch.stack(values).mean().item()
            for key, values in self.training_step_outputs.items()
        }

        # Log epoch-level metrics
        self.log("train/frame_loss/epoch", epoch_metrics['bin_cls_loss'], on_epoch=True)
        self.log("train/node_loss/epoch",  epoch_metrics['node_loss'], on_epoch=True)
        self.log("train/total_loss/epoch", epoch_metrics['total_loss'], on_epoch=True)


        # Clear the lists for next epoch
        self.training_bin_preds = []
        self.training_bin_targets = []
        self.training_node_preds = []
        self.training_node_targets = []


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

        # Print device of the input batch to check if it is on GPU
        # data_device = batch_data['spectrograms'].device
        # print(f"\n Validation on device: { data_device }")

        # You can also print the model's device by checking one of the parameters
        device = next(self.sed_net.parameters()).device
        # print(f"\nValidation Model is on device: {device}")

        # Move data to device
        spectrograms = [spectrogram.clone().detach().to(device) for spectrogram in batch_data['spectrograms']]
        frame_labels = [frame_label.clone().detach().to(device) for frame_label in batch_data['frame_labels']]
        record_binary_labels = torch.tensor(batch_data['record_binary_label'], dtype=torch.long).to(device)  # Ensure tensor type
        batch_audio_names = batch_data['c_ex_mixtures']  # List of audio names (metadata)

        # Prepare batch_data dictionary for the model
        batch_data_device = {
            'spectrograms': spectrograms,
            'frame_labels': frame_labels,
            'c_ex_mixtures': batch_audio_names,
        }


        ''' ======================  record  Level 2 cls ===================== '''
        # 1. Obtain actual abnormal samples in the current batch
        actual_abnormal_indices = [i for i, label in enumerate(record_binary_labels) if label == 1]
        actual_abnormal_names = [batch_audio_names[i] for i in actual_abnormal_indices]

        # Forward pass for classification
        outputs_classification = self.sed_net(batch_data_device, level='record')
        record_binary_predictions = outputs_classification['record_binary_predictions']  # Shape: (batch_size, num_classes)

        # Compute classification losses
        self.valid_record_binary_loss_ce = self.validation_cls_loss_fn(record_binary_predictions, record_binary_labels)
        self.valid_record_binary_loss_fbeta = self.fbeta_loss_fn(record_binary_predictions, record_binary_labels)

        self.valid_record_loss = (
                self.record_loss_weight * self.valid_record_binary_loss_ce +
                self.record_loss_weight * self.valid_record_binary_loss_fbeta
        )

        # Get predicted labels
        record_binary_pred_labels = torch.argmax(record_binary_predictions, dim=1)  # Shape: (batch_size,)

        # Store predictions and labels for confusion matrix computation
        self.validation_bin_preds.append(record_binary_pred_labels.cpu())
        self.validation_bin_targets.append(record_binary_labels.cpu())

        ''' ====================== Node Level Stage ===================== '''

        # Choose the method for validation
        validation_method = self.hparams.get("validation_method", "method2")  # Can be 'method1' or 'method2'

        if validation_method == "method1":  # note, 输入到检测网络中的样本是原始批次中所有的样本，
            print("\n call the  valiation method1 ")
            # **Method 1**: Retain all samples in the batch and input them into the detection network
            batch_data_detection = {
                'spectrograms': spectrograms,
                'frame_labels': frame_labels,
                'c_ex_mixtures': batch_audio_names,
            }

            # Perform detection
            outputs_detection = self.sed_net(batch_data_detection,  level="node")

            node_predictions = outputs_detection['node_predictions']  # Shape: (total_nodes, num_classes)
            node_labels = outputs_detection['node_labels']  # Shape: (total_nodes,)
            batch_indices = outputs_detection['batch_indices']  # Shape: (total_nodes,)
            batch_detection_names = outputs_detection['batch_audio_names']  # List of audio names

            # Even if there are no predicted abnormal samples, proceed with node extraction
            print(f"node_predictions shape: {node_predictions.shape}")
            # Get predicted node labels
            node_pred_labels = torch.argmax(node_predictions, dim=1)

            # Identify nodes predicted as abnormal，
            # note, 验证阶段则是提取 被模型预测为异常类型的节点。
            predicted_abnormal_node_indices = (node_pred_labels != 0).nonzero(as_tuple=True)[0]
            normal_node_indices = (node_pred_labels == 0).nonzero(as_tuple=True)[0]

            # Sample normal nodes (always sample 32 normal nodes if available)
            num_normal_nodes_to_sample = min(64, len(normal_node_indices))  # Adjust as needed

            if len(normal_node_indices) > num_normal_nodes_to_sample:
                sampled_normal_indices = normal_node_indices[
                    torch.randperm(len(normal_node_indices))[:num_normal_nodes_to_sample]
                ]
            else:
                sampled_normal_indices = torch.tensor([], dtype=torch.long, device=device)

            # Combine indices
            node_indices_to_use = torch.cat((predicted_abnormal_node_indices, sampled_normal_indices))
            node_indices_to_use = node_indices_to_use.unique()

            if len(node_indices_to_use) > 0:
                # Extract predictions and labels for selected nodes
                node_predictions_selected = node_predictions[node_indices_to_use]
                node_labels_selected = node_labels[node_indices_to_use]

                # Compute node-level loss
                self.valid_node_loss = self.validation_supervised_loss(node_predictions_selected, node_labels_selected)
                # Accumulate node-level predictions and labels for confusion matrix
                self.validation_node_preds.append(torch.argmax(node_predictions_selected, dim=1).cpu())
                self.validation_node_targets.append(node_labels_selected.cpu())

            else:
                # No nodes to compute loss
                self.valid_node_loss = torch.tensor(0.0, device=device)

                # Perform detection at node level
                outputs_detection = self.sed_net(batch_data_device, level="node")

                node_predictions = outputs_detection['node_predictions']  # Shape: (total_nodes, num_classes)
                node_labels = outputs_detection['node_labels']  # Shape: (total_nodes,)
                batch_indices = outputs_detection['batch_indices']  # Shape: (total_nodes,)
                batch_detection_names = outputs_detection['batch_audio_names']  # List of audio names


                print(f"node_predictions shape: {node_predictions.shape}")
                # Get predicted node labels
                node_pred_labels = torch.argmax(node_predictions, dim=1)

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
                    self.valid_node_loss = self.validation_supervised_loss(node_predictions_selected, node_labels_selected)

                    # Accumulate node-level predictions and labels for confusion matrix
                    self.validation_node_preds.append(torch.argmax(node_predictions_selected, dim=1).cpu())
                    self.validation_node_targets.append(node_labels_selected.cpu())
                else:
                    # No nodes to compute loss
                    self.valid_node_loss = torch.tensor(0.0, device=device)

        elif validation_method == "method2":  # note, 输入到检测网络中的样本是只有被分类为异常类型的样本。
            print("\n call the  valiation method2 ")
            # **Method 2**: Use samples predicted as abnormal by the model
            predicted_abnormal_indices = (record_binary_pred_labels == 1).nonzero(as_tuple=True)[0].tolist()
            predicted_abnormal_names = [batch_audio_names[i] for i in predicted_abnormal_indices]

            # Extract spectrograms and frame_labels for predicted abnormal samples
            spectrograms_predicted_abnormal = [spectrograms[i] for i in predicted_abnormal_indices]
            frame_labels_predicted_abnormal = [frame_labels[i] for i in predicted_abnormal_indices]
            c_ex_mixtures_predicted_abnormal = [batch_audio_names[i] for i in predicted_abnormal_indices]

            if len(predicted_abnormal_indices) == 0:
                #note,  如果当前batch 中， 没有样本被预测为异常类型， 则直接将当前batch 所有样本输入到检测网络中。
                print(f" this batch in Method2,  there are no samples predicate as abnormal : {len(predicted_abnormal_indices)}")
                # No samples predicted as abnormal
                #
                batch_data_detection = {
                    'spectrograms': spectrograms,
                    'frame_labels': frame_labels,
                    'c_ex_mixtures': batch_audio_names,
                }

                # Perform detection
                outputs_detection = self.sed_net(batch_data_detection, level="node")

                node_predictions = outputs_detection['node_predictions']  # Shape: (total_nodes, num_classes)
                node_labels = outputs_detection['node_labels']  # Shape: (total_nodes,)
                batch_indices = outputs_detection['batch_indices']  # Shape: (total_nodes,)
                batch_detection_names = outputs_detection['batch_audio_names']  # List of audio names


            else:
                # Prepare batch_data for detection stage
                batch_data_detection = {
                    'spectrograms': spectrograms_predicted_abnormal,
                    'frame_labels': frame_labels_predicted_abnormal,
                    'c_ex_mixtures': c_ex_mixtures_predicted_abnormal,
                }

                # Perform detection
                outputs_detection = self.sed_net(batch_data_detection,  level="node")

                node_predictions = outputs_detection['node_predictions']  # Shape: (total_nodes, num_classes)
                node_labels = outputs_detection['node_labels']  # Shape: (total_nodes,)
                batch_indices = outputs_detection['batch_indices']  # Shape: (total_nodes,)
                batch_detection_names = outputs_detection['batch_audio_names']  # List of audio names

            # Even if there are no predicted abnormal samples, proceed with node extraction
            print(f"node_predictions shape: {node_predictions.shape}")

            if node_predictions.numel() == 0:
                # No node predictions to process
                self.valid_node_loss = torch.tensor(0.0, device=node_predictions.device)
            else:
                # Get predicted node labels
                node_pred_labels = torch.argmax(node_predictions, dim=1)

                # Identify nodes predicted as abnormal
                predicted_abnormal_node_indices = (node_pred_labels != 0).nonzero(as_tuple=True)[0]
                normal_node_indices = (node_pred_labels == 0).nonzero(as_tuple=True)[0]

                # Sample normal nodes (always sample 32 normal nodes if available)
                num_normal_nodes_to_sample = min(64, len(normal_node_indices))  # Adjust as needed

                if num_normal_nodes_to_sample > 0:
                    sampled_normal_indices = normal_node_indices[
                        torch.randperm(len(normal_node_indices))[:num_normal_nodes_to_sample]
                    ]
                else:
                    sampled_normal_indices = torch.tensor([], dtype=torch.long, device=device)

                # Combine indices
                node_indices_to_use = torch.cat((predicted_abnormal_node_indices, sampled_normal_indices))
                node_indices_to_use = node_indices_to_use.unique()

                if len(node_indices_to_use) > 0:
                    # Extract predictions and labels for selected nodes
                    node_predictions_selected = node_predictions[node_indices_to_use]
                    node_labels_selected = node_labels[node_indices_to_use]

                    # Compute node-level loss
                    self.valid_node_loss = self.validation_supervised_loss(node_predictions_selected, node_labels_selected)
                    # Accumulate node-level predictions and labels for confusion matrix
                    self.validation_node_preds.append(torch.argmax(node_predictions_selected, dim=1).cpu())
                    self.validation_node_targets.append(node_labels_selected.cpu())

                else:
                    # No nodes to compute loss
                    self.valid_node_loss = torch.tensor(0.0, device=device)

        else:
            raise ValueError(f"Unknown validation method: {validation_method}")


       # self.valid_frame_loss =  self.frame_loss_weight * self.valid_frame_loss
        self.valid_node_loss  =  self.node_loss_weight  *  self.valid_node_loss

        # Compute total loss
        self.valid_total_loss =  self.valid_record_loss +  self.valid_node_loss

        # Log individual losses and total loss
        self.log("valid/record_loss", self.valid_record_loss, prog_bar=True)
        self.log("valid/node_loss", self.valid_node_loss, prog_bar=True)
        self.log("valid/total_loss", self.valid_total_loss, prog_bar=True)

        # 初始化以下指标， 开始为计算 psd 分数， 交集 f1 score, 基于阈值的f score，
        # 这三种声音事件检测指标而做准备。
        # note, batch_audio_names  需要更新成 batch detection names， 表示只获取预测为异常样本的音频文件；
        filenames_synth = [ x  for x in  batch_detection_names
                        if Path(x).parent == Path(self.hparams["data"]["eval_folder_8k"])
                         ]

        # 获取当前batch 中每个音频的持续时间长度。
        batch_audio_duration = []
        valid_df = pd.read_csv(self.hparams["data"]["valid_dur"], sep='\t') # 验证集上所有的音频的持续时间；
        # Iterate over your list of filenames
        for file in filenames_synth:
            file = os.path.basename(file)
            # Find the row in the DataFrame that matches the filename
            duration = valid_df.loc[valid_df['filename'] == file, 'duration'].values[0]
            # Append the filename and duration to the batch
            batch_audio_duration.append( duration)

        # todo,  检查这里的node predictions, batch indices 是否对应上需要的数据；
        # 保留原始数据：scores_raw 保留了原始预测分数，方便后续调试和分析。
        # 提高准确性：scores_postprocessed 通过中值滤波提高了预测分数的稳定性，减少了噪声的影响。
        # 多阈值评估：prediction_dfs 提供了不同阈值下的预测结果，便于选择最佳阈值，提高预测性能。
        # 综上所述，这三种输出各有侧重，共同构成了一个完整的预测流程，确保了从原始数据到最终预测结果的全面覆盖。


        # Decoding outputs at frame level
        ( scores_raw_node_pred,
          scores_postprocessed_node_pred,
          decoded_node_pred,
          decoded_abnormal_pred,
        ) = batched_node_decode_preds(
            node_predictions,
            filenames_synth,
            self.encoder,  #   note,  这里传入 异常类别的标签；
            batch_indices, #  此时的索引， 代表的是这些节点 位于当前预测为异常样本构成的 batch 中的索引；
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
        # Frame-level metrics
        if len(self.validation_bin_preds) > 0:
            # Concatenate all predictions and targets
            all_preds = torch.cat(self.validation_bin_preds)
            all_targets = torch.cat(self.validation_bin_targets)
            # Compute confusion matrix
            confu_mat = confusion_matrix(all_targets.numpy(), all_preds.numpy())
            print(f"\n Confusion Matrix:\n{confu_mat}")  # note, 输入到检测网络中的样本是只有被分类为异常类型的样本。

            # Compute additional metrics
            precision = precision_score(all_targets.numpy(), all_preds.numpy(), average='macro')
            recall = recall_score(all_targets.numpy(), all_preds.numpy(), average='macro')
            binary_f1_score = f1_score(all_targets.numpy(), all_preds.numpy(), average='macro')
            print(
                f" Calling the sklearn method : \t Record level binary F1 score: {binary_f1_score}, precision：{precision}, recall:{recall} \n")

            # Log frame-level metrics
            self.log('val/record_precision', precision)
            self.log('val/record_recall', recall)
            self.log('val/record_f1_score', binary_f1_score)
        else:
            print("\n No record-level predictions to compute confusion matrix.")

        # Node-level metrics
        if len(self.validation_node_preds) > 0:
            all_node_preds = torch.cat(self.validation_node_preds)
            all_node_targets = torch.cat(self.validation_node_targets)

            # Compute node-level confusion matrix
            labels = list(classes_labels.values())
            node_cm = confusion_matrix(all_node_targets.numpy(), all_node_preds.numpy(), labels=labels)
            print_matrix_with_labels(node_cm, classes_labels, "Node-level Validation Confusion Matrix")

            # Compute ratio format of the confusion matrix
            with np.errstate(divide='ignore', invalid='ignore'):
                node_cm_ratio = node_cm.astype('float') / node_cm.sum(axis=1)[:, np.newaxis]
                node_cm_ratio = np.nan_to_num(node_cm_ratio)  # Replace NaN with zero

            print_matrix_with_labels(node_cm_ratio, classes_labels,
                                     "Node-level Validation Confusion Matrix (Ratio Format)")

            # Compute precision, recall, F1 score
            node_precision = precision_score(all_node_targets.numpy(), all_node_preds.numpy(), average='macro')
            node_recall = recall_score(all_node_targets.numpy(), all_node_preds.numpy(), average='macro')
            node_f1_score = f1_score(all_node_targets.numpy(), all_node_preds.numpy(), average='macro')
            print(f"Node-level Validation Metrics:\n"
                  f" F1 score: {node_f1_score}, Precision: {node_precision}, Recall: {node_recall}\n")

            # Log node-level metrics
            self.log('val/node_precision', node_precision)
            self.log('val/node_recall', node_recall)
            self.log('val/node_f1_score', node_f1_score)
        else:
            print("\n No node-level predictions to compute confusion matrix.")

        #  读取valid 数据集的真实标签和音频时长。
        # ground_truth = sed_scores_eval.io.read_ground_truth_events(
        #     self.hparams["data"]["valid_tsv"]
        # )
        #
        # 更新后的函数可以用来建立空列表的事件， 由于空列表代表的是正常类型的record ,后续会将其过滤掉；
        ground_truth = sed_scores_eval.io.read_ground_truth_events_w_NA(
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


        DEFAULT_EVENT_SEGMENT_SCORED = (0.0,)  # Adjust based on expected structure



        # Compute node-level event-based metrics
        try:
            node_event_segment_scored = compute_event_based_metrics(
                self.val_buffer_abnormal_node[self.hparams["training"]["val_thresholds"][0]],
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

        self.log('best_val_classWise_score', self.best_val_score, prog_bar=True)

        # Log validation metrics
        print(
            f"\tval/record_level/binary f1 score: {binary_f1_score}"
            f"\tval/obj_metric: {obj_metric}\n"
            f"\tval/node_level/event_based_class_wise_average_f1_score: {node_event_thread_f_score}"
        )

        self.log("val/obj_metric", obj_metric, prog_bar=True)
        self.log("val/record_level/binary f1 score", binary_f1_score, prog_bar=True)
        self.log("val/node_level/event_based_class_wise_average_f1_score", node_event_thread_f_score, prog_bar=True)

        # Clear validation buffers for the next epoch
        self.validation_bin_preds = []
        self.validation_bin_targets = []
        self.validation_node_preds = []
        self.validation_node_targets = []

        # Clear buffers for frame level
        # self.val_scores_postprocessed_buffer_frame_level = {}
        # self.val_buffer_frame_level = {th: pd.DataFrame() for th in self.val_buffer_frame_level.keys()}
        # self.val_buffer_abnormal_frame = {th: pd.DataFrame() for th in self.val_buffer_abnormal_frame.keys()}

        # Clear buffers for node level
        self.val_scores_postprocessed_buffer_node_level = {}
        self.val_buffer_node_level = {th: pd.DataFrame() for th in self.val_buffer_node_level.keys()}
        self.val_buffer_abnormal_node = {th: pd.DataFrame() for th in self.val_buffer_abnormal_node.keys()}

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

        # Separate parameters
        # Separate parameters into classification and detection
        record_bin_params = (
            list(self.sed_net.node_fea_generator.parameters()) +
            list(self.sed_net.binary_cls_module.parameters())
        )
        node_learing_params = (
            #list(self.sed_net.node_fea_generator.parameters()) +
            list(self.sed_net.node_cls_module.parameters())
        )
        ##### training params and optimizers ############

        # Create parameter groups with names
        optimizer = torch.optim.Adam([
            {'params': record_bin_params, 'lr': self.record_lr_initial, 'name': 'record_cls'},
            {'params': node_learing_params, 'lr': self.node_lr_initial, 'name': 'detection'},
            # {'params': shared_params, 'lr': self.classification_lr_initial, 'name': 'shared'},
        ])


        # Scheduler using LambdaLR
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=[
                lambda epoch: self.compute_lr_factor(epoch, 'record_cls'),
                lambda epoch: self.compute_lr_factor(epoch, 'detection'),
                # lambda epoch: self.compute_lr_factor(epoch, 'shared'),
            ]
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
                    print(f" - {param.shape}")



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
                if group_name == 'classification':
                    lr_factor = self.classification_lr_final / self.classification_lr_initial
                elif group_name == 'detection':
                    lr_factor = self.detection_lr_final / self.node_lr_initial
                else:
                    lr_factor = 1.0  # No adjustment
            else:
                # During adjustment, linearly interpolate learning rate
                adjustment_factor = epochs_since_adjustment / total_adjustment_epochs

                if group_name == 'classification':
                    lr_initial = self.classification_lr_initial
                    lr_final = self.classification_lr_final
                elif group_name == 'detection':
                    lr_initial = self.node_lr_initial
                    lr_final = self.detection_lr_final
                else:
                    return 1.0  # No adjustment

                # Linearly interpolate current_lr between initial and final
                current_lr = lr_initial + adjustment_factor * (lr_final - lr_initial)

                # Compute lr_factor as current_lr / lr_initial
                lr_factor = current_lr / lr_initial

            return lr_factor

    # Custom collate function to convert spectrograms to Data objects during batch preparation

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
            'record_binary_label':[item['record_binary_label'] for item in batch]
            # Include any other necessary data
        }

        return batch_data


    def train_dataloader(self): #   训练 DataLoader 使用自定义的 batch sampler;
        #self.train_loader = torch_geometric.data.DataLoader(
        self.train_loader = DataLoader(
            self.train_data,
            # batch_sampler=self.train_sampler,

            # batch_sampler expects an iterable over batches of indices,
            #  whereas WeightedRandomSampler yields individual indices.
            #  batch_sampler期望对批量的索引进行迭代，
            #  而WeightedRandomSampler生成单独的索引。

            sampler= self.train_sampler,
            # 使用sampler参数而不是batch_sampler,
            # 此参数接受生成单独索引的采样器，这就是WeightedRandomSampler提供的
            batch_size= self.hparams["training"]["batch_size"] , # None,   #
            num_workers=self.num_workers,
            # shuffle=True,
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
                self.node_loss_weight_initial = self.classification_loss_weight_final
                self.detection_loss_weight = self.detection_loss_weight_final
            else:
                # Calculate adjustment factor (between 0 and 1)
                factor = epochs_since_adjustment / total_adjustment_epochs

                # # Linearly interpolate between initial and final weights
                # self.node_loss_weight_initial = self.node_loss_weight_initial - factor * (
                #         self.node_loss_weight_initial - self.classification_loss_weight_final
                # )
                # self.detection_loss_weight = self.frame_loss_weight_initial + factor * (
                #         self.detection_loss_weight_final - self.frame_loss_weight_initial
                # )

                # Linearly interpolate between initial and final weights
                self.node_loss_weight_initial = self.node_loss_weight_initial + factor * (
                        self.classification_loss_weight_final - self.node_loss_weight_initial
                )
                self.detection_loss_weight = self.frame_loss_weight_initial + factor * (
                        self.detection_loss_weight_final - self.frame_loss_weight_initial
                )

            # Log the updated loss weights
            self.log('loss_weights/classification', self.node_loss_weight_initial)
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
class FocalLoss(nn.Module): # 用于节点的分类损失计算；
    def __init__(self, alpha=None, gamma=2, reduction='mean', device='cpu'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.device = device

        if self.alpha is not None:
            if isinstance(self.alpha, (list, torch.Tensor)):
                self.alpha = torch.tensor(self.alpha, dtype=torch.float32, device=device)
            else:
                self.alpha = torch.tensor([alpha], dtype=torch.float32, device=device)

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-CE_loss)  # Probabilities of the correct class
        F_loss = (1 - pt) ** self.gamma * CE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss



def worker_init_fn(worker_id):
    print(f"Initializing worker {worker_id}")

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
import numpy as np

def calculate_class_weights(dataset):
    all_labels = []
    for sample in dataset.examples.values():
        labels_df = sample['label_df']
        if not labels_df.empty:
            labels = labels_df['event_label'].tolist()
            all_labels.extend(labels)
    class_counts = Counter(all_labels)
    total_counts = sum(class_counts.values())
    class_weights = {cls: total_counts / count for cls, count in class_counts.items()}
    # Optionally normalize the class weights
    sum_weights = sum(class_weights.values())
    class_weights = {cls: weight / sum_weights for cls, weight in class_weights.items()}
    return class_weights



# def print_matrix_with_alignment(matrix, title):
#     print(f"\n{title}:")
#     for row in matrix:
#         print(" ".join(f"{val:10.4f}" if isinstance(val, float) else f"{val:10d}" for val in row))

def calculate_binary_class_weights(dataset):
    all_labels = []
    for sample in dataset.examples.values():
        #print(sample)
        cur_binary_label = sample['record_bin_label'] #.tolist()
        #all_labels.extend(cur_binary_label)
        all_labels.append(cur_binary_label)
    class_counts = Counter(all_labels)
    total_counts = sum(class_counts.values())
    class_weights = {cls: total_counts / count for cls, count in class_counts.items()}
    # Optionally normalize the class weights
    sum_weights = sum(class_weights.values())
    class_weights = {cls: weight / sum_weights for cls, weight in class_weights.items()}
    print(f"The binary class  sample weights and for the loss weight \n {class_weights}")
    return class_weights
