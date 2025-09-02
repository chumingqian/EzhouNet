import os
import random
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim import Optimizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix, precision_score, recall_score
import sed_scores_eval

import sed_scores_eval
import torch
import  torch.nn  as  nn

from codecarbon import OfflineEmissionsTracker
from desed_task.data_augm import mixup
from desed_task.evaluation.evaluation_measures import (
    compute_per_intersection_macro_f1, compute_psds_from_operating_points,
    compute_psds_from_scores)
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from .utils import  batched_node_edge_decode_preds,  compute_event_based_metrics
from torch.utils.data import Dataset, Subset, ConcatDataset


# hf_lab2_1: 直接使用1层refine layer 直接学习 offset;
#  节点损失参与计算    +  time shift 数据增强；
#  且此时 self.gat_model,  interval_refine 全部实现并行化处理，



##  v3-1,  数据集：修改先验区间的持续时间为 ， 并且保证尺度2的个数最多，
#  Scale 1: 0.5s, Scale 2: 0.8s, Scale 3: 1.5s
# 2. 网络， 对于粗粒度的 区间，同样分配更多的bins，  从而可以进行更大范围的偏移学习；
# 即较粗的间隔需要更多的偏移容量才能有效地细化。
# 3. 训练， 将节点分类与区间置信度同步优化，
#  即第一阶段先优化 节点分类与区间置信度，  第二阶段优化区间分类，以及定位；



#  v3-2,  调整基于 step 的频率来更新学习率，  由于新的的数据集每个epoch 上 batch 个数的增加，
#  因此， 需要调整为基于 step 的方式来更新学习率；



from functools import partial

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

#
# node_cls_labels = OrderedDict({
#     "Normal": 0,
#     "Rhonchi": 1,
#     "Wheeze": 2,
#     "Stridor": 3,
#     "Crackle": 4,
# })

node_cls_labels  = OrderedDict({
    "Rhonchi": 0,
    "Wheeze": 1,
    "Stridor": 2,
    "Crackle": 3,
})



interval_4labels = {
    "Crackle": 0,
    "Wheeze": 1,
    "Stridor": 2,
    "Rhonchi": 3
}



bin_labels = OrderedDict({
    "Normal": 0,
    "Abnormal": 1,
})

bin_node_vad_labels = OrderedDict({
    "Non_vad": 0,
    "In_vad": 1,
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




        self.shared_params_all = list(self.sed_net.parameters())  # No filtering
        print("Assigned all params to shared_params")

        # Parameter separation
        self.shared_params =  (  list(self.sed_net.node_fea_generator.parameters()) +
                                 list(self.sed_net.node_edge_proj.parameters() ) +
                                 list(self.sed_net.edge_encoder.parameters())
                                                     )

        self.node_update_conf_cls_params = (  list(self.sed_net.node_interval_pred.gat_model.parameters()) +
                                              list(self.sed_net.node_interval_pred.node_class_heads.parameters() )
                                                     )

        self.interval_conf_cls_refine_params = (list(self.sed_net.node_interval_pred.interval_refine.parameters())
                                           )


        print("Parameter grouping completed successfully")
        print(f"Shared params: {len(self.shared_params)}")
        print(f"Node interval conf cls params: {len(self.node_update_conf_cls_params )}")
        print(f"Interval offset params: {len(self.interval_conf_cls_refine_params)}")
        print(f"Total model params: {len(list(self.sed_net.parameters()))}")


        self.train_flag = train_flag
        # if sed_teacher is None:
        #     self.sed_teacher = deepcopy(EzhouNet)
        # else:
        #     self.sed_teacher = sed_teacher
        # this  threshold use to judge ,when the iou  above this,
        # the  ground truth of the pred  interval will be  assign the  abnormal type;


        self.ema_model = deepcopy(self.sed_net)
        for para in self.ema_model.parameters():
            para.requires_grad = False

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


        # # 用于节点任务的 分类损失；

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

        """"
        node_cls_labels  = OrderedDict({
                "Rhonchi": 0,
                "Wheeze": 1,
                "Stridor": 2,
                "Crackle": 3,
            })   
        """

        # Loss functions for node tasks
        # self.node_conf_loss_fn = FocalLossBinary(alpha=0.70, gamma=2.0, reduction='mean')
        abnormal_ratio = 0.30  # , this will  be change  according to the  frams stride
        pos_weight_node = (1 - abnormal_ratio) / abnormal_ratio
        pos_weight_node_tensor = torch.tensor(pos_weight_node, dtype=torch.float32)
        self.node_conf_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_node_tensor)

        #     note, the number of nodes  will change
        # according to different dataset  and sample rate with, hop_len;
        # node_counts = torch.tensor([43064, 85324, 7730, 169670], dtype=torch.float32)
        node_counts = torch.tensor([56676, 93746, 2598, 135418], dtype=torch.float32) # sr =4k, hop=64;
        node_cls_alpha = 1.0 / node_counts
        self.node_cls_alpha = node_cls_alpha / node_cls_alpha.sum()
        self.node_cls_loss_fn = FocalLoss(alpha=self.node_cls_alpha, gamma=2.0, reduction='mean')



        # Initialize the loss function (you can experiment with alpha and gamma)
        #self.interval_conf_loss_fn = FocalLossBinary(alpha=0.85, gamma=2.0, reduction='mean')
        interval_abnormal_ratio = 0.2  # , this will  be change  according to the  frames stride
        pos_weight_it = (1 - interval_abnormal_ratio) / interval_abnormal_ratio
        pos_weight_it_tensor = torch.tensor(pos_weight_it, dtype=torch.float32)
        self.interval_conf_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_it_tensor)


        # Compute interval class alpha (class weights)
        #  [Crackle, Wheeze,Stridor, Rhonchi, ]
        # alpha = [0.023, 0.035, 0.256, 0.686] for [Crackle, Wheeze,, Stridor, Rhonchi]
        # counts = torch.tensor([1289, 871, 44, 119,], dtype=torch.float32)
        # counts = torch.tensor([13794, 7027, 657,3780, ], dtype=torch.float32)
        counts = torch.tensor([5536, 3627, 128, 2471, ], dtype=torch.float32)
        interval_alpha = 1.0 / counts  # Inverse of class counts,
        self.interval_cls_alpha =  interval_alpha / interval_alpha.sum()  # Normalize to sum to 1
        self.interval_cls_loss_fn = FocalLoss(alpha=self.interval_cls_alpha, gamma=2.0, reduction='mean')


        # Loss weight parameters

        # Loss weights (adjust as needed, aligned with v9_2_2)
        self.node_conf_loss_weight = 2.0
        self.node_loss_weight =  20.0 #2.0


        self.interval_conf_loss_weight =   5.0  # 40.0
        self.interval_cls_loss_weight =    40.0 # 15.0    #50.0
        self.interval_location_loss_weight = 1.05


        #  创建一个异常的缓冲， 用于专门存储异常的 预测节点；
        self.val_buffer_intervals= {  # 为不同阈值创建多个 DataFrame 缓冲区，用于存储合成数据和测试数据的验证结果。
            conf_th: pd.DataFrame() for conf_th in self.hparams["training"]["val_conf_thresholds"]

        }


        self.best_val_score = float('-inf')  # Track the best validation score
        self.best_model_weights = None       # Placeholder for the best model weights (if needed)
        self.best_ema_weights = None
        self.best_node_f1 = float('-inf')

        # 通过这些步骤，该初始化方法为后续的训练、验证和测试过程准备了必要的组件和数据结构。



        # Initialize storage for node label counts per epoch
        self.epoch_node_label_counts = torch.zeros(4, dtype=torch.long)
        self.valid_epoch_node_label_counts = torch.zeros(4, dtype=torch.long)



        # Storage for epoch metrics
        self.training_node_preds = []  # Classification predictions
        self.training_node_targets = []  # Classification targets

        self.training_node_conf_preds = []  # Confidence logits
        self.training_node_conf_targets = []  # Confidence targets
        self.training_node_conf_targets_binary = []
        self.training_step_outputs = {
            'node_level_loss': [],
            'node_conf_loss': [],

            'interval_conf_loss': [],
            'interval_cls_loss': [],
            'interval_loc_loss': [],

            'total_loss': []
        }

        self.val_buffer_intervals= {  # 为不同阈值创建多个 DataFrame 缓冲区，用于存储合成数据和测试数据的验证结果。
            conf_th: pd.DataFrame() for conf_th in self.hparams["training"]["val_conf_thresholds"]

        }

        # Validation storage
        self.validation_node_preds = []
        self.validation_node_targets = []
        self.validation_node_conf_preds = []
        self.validation_node_conf_targets = []
        self.validation_node_conf_targets_binary = []
        self.valid_step_outputs = {
            'node_level_loss': [],
            'node_conf_loss': [],

            'interval_conf_loss': [],
            'interval_cls_loss': [],
            'interval_loc_loss': [],

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

    def _get_true_abnormal_counts(self, batch):
        """Extract true abnormal interval counts from batch"""
        # Implementation depends on your data structure
        # Example: batch["gt_intervals"] could be a list of interval lists
        return sum(len(intervals) for intervals in batch["vad_timestamps"])
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


    def interval_iou(self,pred_interval, gt_interval):
        """
        note,    # compute  one sample iou value
        pred_interval: [p_start, p_end]
        gt_interval:   [g_start, g_end]
        returns: IoU (float)
        """
        p_start, p_end = pred_interval
        g_start, g_end = gt_interval

        # Compute the intersection
        inter_start = max(p_start, g_start)
        inter_end = min(p_end, g_end)
        intersection = max(0.0, inter_end - inter_start)

        # Compute each interval length
        p_length = p_end - p_start
        g_length = g_end - g_start

        # Compute union
        union = p_length + g_length - intersection

        # Avoid division by zero
        if union <= 0.0:
            return 0.0

        iou = intersection / union
        return iou

    import torch
    import torch.nn.functional as F

    def inference_one_sample( self,
            bounds: torch.Tensor,  # (M,2)
            conf_logits: torch.Tensor,  # (M,1)
            class_logits: torch.Tensor,  # (M,num_interval_classes)
            conf_threshold: float = 0.5,
            nms_iou_threshold: float = 0.8
    ):
        """
        Applies confidence threshold + NMS + picks the best abnormal sub-type.
        Returns final intervals, confidences, and predicted sub-type IDs.
        """
        # 1) Convert confidence logits to probabilities,
        # note, here need to be noticed that does it use sigmoid  for the conf logit during training;
        conf_scores = torch.sigmoid(conf_logits.view(-1))  # shape (M,)

        # 2) Keep intervals above threshold
        keep_mask = (conf_scores > conf_threshold)
        bounds = bounds[keep_mask]
        conf_scores = conf_scores[keep_mask]
        class_logits = class_logits[keep_mask]

        if bounds.numel() == 0:
            # No intervals survive => return empty
            return torch.empty((0, 2)), torch.empty((0,)), torch.empty((0,), dtype=torch.long)

        # 3) Sort by descending confidence
        sort_inds = torch.argsort(conf_scores, descending=True)
        bounds = bounds[sort_inds]
        conf_scores = conf_scores[sort_inds]
        class_logits = class_logits[sort_inds]

        # 4) NMS
        final_indices = []
        for i in range(len(bounds)):
            keep = True
            for j in final_indices:
                if self.interval_iou(bounds[i], bounds[j]) > nms_iou_threshold:
                    keep = False
                    break
            if keep:
                final_indices.append(i)

        final_bounds = bounds[final_indices]
        final_conf = conf_scores[final_indices]
        final_class_logits = class_logits[final_indices]

        # 5) Among abnormal classes, pick the best sub-type
        final_class_probs = F.softmax(final_class_logits, dim=-1)  # shape (K, num_interval_classes)
        pred_subtypes = torch.argmax(final_class_probs, dim=-1)  # shape (K,)

        return final_bounds, final_conf, pred_subtypes

    # def batch_inference_interval(self,
    #         model_outputs: dict,
    #         conf_threshold: float = 0.5,
    #         nms_iou_threshold: float = 0.8
    # ):
    #     """
    #     model_outputs: {
    #         "pred_intervals": Tensor,           # [batch_size, total_intervals, 2]
    #         "pred_intervals_conf_logits": Tensor,   # [batch_size, total_intervals, 1]
    #         "pred_intervals_cls_logits": Tensor,    # [batch_size, total_intervals, num_classes - 1]
    #         ...
    #     }
    #     Returns: A list of length batch_size, each element is a dict with final intervals, confidences, sub-type IDs.
    #     """
    #     final_bounds = model_outputs["pred_intervals"]  # [batch_size, total_intervals, 2]
    #     interval_conf_logits = model_outputs["pred_intervals_conf_logits"]  # [batch_size, total_intervals, 1]
    #     interval_cls_logits = model_outputs["pred_intervals_cls_logits"]  # [batch_size, total_intervals, num_classes - 1]
    #
    #     batch_size = final_bounds.size(0)
    #     batch_results = []
    #
    #     for b_idx in range(batch_size):
    #         # Per-sample predictions (index into batch dimension)
    #         bounds_b = final_bounds[b_idx]          # [total_intervals, 2]
    #         conf_b = interval_conf_logits[b_idx]    # [total_intervals, 1]
    #         cls_b = interval_cls_logits[b_idx]      # [total_intervals, num_classes - 1]
    #
    #         # Run single-sample inference
    #         final_b, final_conf_b, final_subtypes_b = self.inference_one_sample(
    #             bounds_b, conf_b, cls_b,
    #             conf_threshold=conf_threshold,
    #             nms_iou_threshold=nms_iou_threshold
    #         )
    #
    #         # Store results in a dict
    #         sample_result = {
    #             "pred_intervals": final_b,          # [K, 2]
    #             "pred_interval_confident": final_conf_b,  # [K]
    #             "pred_interval_cls": final_subtypes_b     # [K]
    #         }
    #         batch_results.append(sample_result)
    #
    #     return batch_results
    #




    def inference_batch_interval(self,
            model_outputs: dict,
            conf_threshold: float = 0.5,
            nms_iou_threshold: float = 0.8
    ):
        """
        model_outputs: {
          "final_bounds": List[Tensor],           # len B
          "interval_conf_logits": List[Tensor],   # len B
          "interval_cls_logits": List[Tensor],    # len B
          ...
        }
        Returns: A list of length B, each element is a dict with final intervals, confidences, sub-type IDs.
        """
        final_bounds_list = model_outputs["pred_intervals"]  # List[Tensor], each (M_b, 2)
        interval_conf_list = model_outputs["pred_intervals_conf_logits"]  # List[Tensor], each (M_b, 1)
        interval_cls_list = model_outputs["pred_intervals_cls_logits"]  # List[Tensor], each (M_b, num_interval_classes)

        batch_size = len(final_bounds_list)
        batch_results = []

        for b_idx in range(batch_size):
            # Per-sample predictions
            bounds_b = final_bounds_list[b_idx]
            conf_b = interval_conf_list[b_idx]
            cls_b = interval_cls_list[b_idx]

            # Run single-sample inference
            final_b, final_conf_b, final_subtypes_b = self.inference_one_sample(
                bounds_b, conf_b, cls_b,
                conf_threshold=conf_threshold,
                nms_iou_threshold=nms_iou_threshold
            )

            # Store results in a dict
            sample_result = {
                "pred_intervals": final_b,  # (K,2)
                "pred_interval_confident": final_conf_b,  # (K,)
                "pred_interval_cls": final_subtypes_b  # (K,)
            }
            batch_results.append(sample_result)

        return batch_results



    def decode_results(self,batch_results, batch_audio_names, interval_4labels):
        """
        Decodes a batch of prediction results into a single DataFrame.

        Parameters:
            batch_results (list[dict]): List where each element is a dictionary
                containing keys 'pred_intervals', 'pred_interval_confident',
                and 'pred_interval_cls'. Each 'pred_intervals' is expected to be
                a tensor of shape (K, 2) for K predicted intervals.
            batch_audio_names (list[str]): List of file names corresponding to each
                audio in the batch.
            interval_4labels (dict): Mapping from string label to predicted id,
                e.g., {"Crackle": 0, "Wheeze": 1, "Stridor": 2, "Rhonchi": 3}.

        Returns:
            pd.DataFrame: A DataFrame with columns ['event_label', 'onset', 'offset', 'file_name'].
        """
        # Create a reverse mapping: predicted id -> event label string.
        id_to_label = {v: k for k, v in interval_4labels.items()}

        records = []

        # Iterate over each audio result and its corresponding file name.
        for audio_name, result in zip(batch_audio_names, batch_results):
            # Each result should have keys: 'pred_intervals', 'pred_interval_confident', 'pred_interval_cls'
            pred_intervals = result.get("pred_intervals")
            pred_cls = result.get("pred_interval_cls")

            # Skip if there are no predictions
            if pred_intervals is None or pred_intervals.numel() == 0:
                continue

            # Move tensors to CPU and convert to numpy arrays.
            pred_intervals_np = pred_intervals.detach().cpu().numpy()
            pred_cls_np = pred_cls.detach().cpu().numpy()

            # Iterate over each prediction for the current audio.
            for i in range(pred_intervals_np.shape[0]):
                onset = float(pred_intervals_np[i, 0])
                offset = float(pred_intervals_np[i, 1])
                cls_id = int(pred_cls_np[i])
                event_label = id_to_label.get(cls_id, "Unknown")
                file_name = os.path.basename(audio_name)

                records.append({
                    "event_label": event_label,
                    "onset": onset,
                    "offset": offset,
                    "filename": file_name
                })

        # Build the DataFrame with the required columns.
        df = pd.DataFrame(records, columns=["event_label", "onset", "offset", "filename"])
        return df

    # def decode_results(self, batch_results, batch_audio_names, interval_4labels):
    #     # Create a reverse mapping: predicted id -> event label string.
    #     id_to_label = {v: k for k, v in interval_4labels.items()}
    #
    #     all_records = []
    #     for audio_name, result in zip(batch_audio_names, batch_results):
    #         pred_intervals = result.get("pred_intervals")
    #         pred_cls = result.get("pred_interval_cls")
    #
    #         # Skip if there are no predictions
    #         if pred_intervals is None or pred_intervals.numel() == 0:
    #             continue
    #
    #         # Move tensors to CPU and convert to numpy arrays.
    #         pred_intervals_np = pred_intervals.detach().cpu().numpy()
    #         pred_cls_np = pred_cls.detach().cpu().numpy()
    #
    #         # Create arrays for onset, offset, and class labels
    #         onsets = pred_intervals_np[:, 0]
    #         offsets = pred_intervals_np[:, 1]
    #         cls_ids = pred_cls_np
    #
    #         # Map class ids to labels
    #         event_labels = [id_to_label.get(cls_id, "Unknown") for cls_id in cls_ids]
    #         file_names = [os.path.basename(audio_name)] * len(onsets)
    #
    #         # Create a list of dictionaries for the current audio
    #         records = [
    #             {
    #                 "event_label": event_label,
    #                 "onset": onset,
    #                 "offset": offset,
    #                 "filename": file_name
    #             }
    #             for event_label, onset, offset, file_name in zip(
    #                 event_labels, onsets, offsets, file_names
    #             )
    #         ]
    #
    #         all_records.extend(records)
    #
    #     # Build the DataFrame with the required columns.
    #     df = pd.DataFrame(all_records, columns=["event_label", "onset", "offset", "filename"])
    #     return df

    def compute_iou_matrix(self,pred_boxes, gt_boxes):
        """
        Compute IoU between each pair of anchor interval  and GT intervals.

        Args:
            pred_boxes (torch.Tensor): Shape (N, 2), [[start1, end1], ...]
            gt_boxes (torch.Tensor): Shape (M, 2), [[start1, end1], ...]

        Returns:
            torch.Tensor: IoU matrix of shape (N, M)
        """
        N = pred_boxes.size(0)
        M = gt_boxes.size(0)
        pred_starts = pred_boxes[:, 0].unsqueeze(1)  # (N, 1)
        pred_ends = pred_boxes[:, 1].unsqueeze(1)  # (N, 1)
        gt_starts = gt_boxes[:, 0].unsqueeze(0)  # (1, M)
        gt_ends = gt_boxes[:, 1].unsqueeze(0)  # (1, M)

        inter_start = torch.max(pred_starts, gt_starts)  # (N, M)
        inter_end = torch.min(pred_ends, gt_ends)  # (N, M)
        inter_len = (inter_end - inter_start).clamp(min=0)  # Intersection length

        pred_len = pred_ends - pred_starts
        gt_len = gt_ends - gt_starts
        union_len = pred_len + gt_len - inter_len  # Union length

        iou = inter_len / union_len.clamp(min=1e-6)  # Avoid division by zero
        return iou  # Shape (N, M)

    def bt_interval_iou(self,pred_bounds: torch.Tensor, gt_bounds: torch.Tensor) -> torch.Tensor:
        """
        note  # compute batch sample iou value
        pred_bounds: (N, 2) => [start, end]
        gt_bounds:   (N, 2) => [start, end]
        returns: IoU for each of the N pairs in a 1D vector (N,)
        """
        # pred_bounds[:, 0] => predicted start, pred_bounds[:, 1] => predicted end
        pred_starts = pred_bounds[:, 0]
        pred_ends = pred_bounds[:, 1]
        gt_starts = gt_bounds[:, 0]
        gt_ends = gt_bounds[:, 1]

        # Intersection
        inter_starts = torch.max(pred_starts, gt_starts)
        inter_ends = torch.min(pred_ends, gt_ends)
        intersection = torch.clamp(inter_ends - inter_starts, min=0.0)

        # Union
        pred_lengths = pred_ends - pred_starts
        gt_lengths = gt_ends - gt_starts
        union = pred_lengths + gt_lengths - intersection

        # IoU
        iou = intersection / union.clamp(min=1e-8)  # avoid /0
        return iou

    def iou_loss_1d(self,pred_bounds: torch.Tensor, gt_bounds: torch.Tensor, loss_type="neg_log_iou") -> torch.Tensor:
        """
        pred_bounds: (N,2)
        gt_bounds:   (N,2)
        returns: scalar IoU-based loss
        loss_type: can be "1_minus_iou" or "neg_log_iou"
        """
        iou_vals = self.bt_interval_iou(pred_bounds, gt_bounds)  # (N,)
        if loss_type == "1_minus_iou":
            loss = 1.0 - iou_vals
        elif loss_type == "neg_log_iou":
            # -log(iou), watch out for iou=0 => clamp
            loss = -torch.log(iou_vals.clamp(min=1e-8))
        else:
            raise ValueError("Unknown loss_type")

        return loss.mean()  # average across all intervals

    #def training_step(self, batch_data, batch_indx, optimizer_idx):
    def training_step(self, batch_data, batch_indx):
        """Apply the training for one batch (a step). Used during trainer.fit

        Args:
            batch_data: dict containing batched data
            batch_idx: int, index of the batch

        Returns:
            torch.Tensor, the loss to take into account.
        """

        # You can also print the model's device by checking one of the parameters
        # device = next(self.sed_net.parameters()).device
        # print(f"Model is on device: {device}")

        device = self.device

        frozen_params = [(name, p) for name, p in self.sed_net.named_parameters() if not p.requires_grad]
        if frozen_params:
            print(f"Warning: Found {len(frozen_params)} frozen parameters at training_step start")
            for name, _ in frozen_params:
                print(f"  - {name}")
            for _, p in frozen_params:
                p.requires_grad = True
        print("training_step: After frozen check, sed_net requires_grad:",
              all(p.requires_grad for p in self.sed_net.parameters()))

        # Move data to device
        # Move data to device
        spectrograms = [s.clone().detach().to(device) for s in batch_data['spectrograms']]
        frame_labels = [f.clone().detach().to(device) for f in batch_data['frame_labels']]

        record_binary_labels = torch.tensor(batch_data['record_binary_label'], dtype=torch.long).to(device)
        audios_dur = [d.clone().detach().to(device) for d in batch_data['audio_dur']]
        c_ex_mixtures = batch_data['c_ex_mixtures']
        chest_pos = [c.clone().detach().to(device) for c in batch_data['chest_info']]
        # gender_info = [g.clone().detach().to(device) for g in batch_data['gender_info']]
        vad_timestamps = batch_data['vad_timestamps']
        anchor_intervals = batch_data["anchor_intervals"]
        assignments = batch_data["assignments"]

        # Prepare batch_data dictionary for the model
        batch_data_device = {
            'spectrograms': spectrograms,
            'frame_labels': frame_labels,
            'c_ex_mixtures': c_ex_mixtures,
            'vad_timestamps': vad_timestamps,
            'chest_loc': chest_pos,
            # 'genders': gender_info,
            'audio_dur': audios_dur,

            "anchor_intervals": anchor_intervals,
            "assignments": assignments

        }



        ''' ====================== node   level  stage==================='''
        outputs_detection = self.sed_net(batch_data_device, )

        node_predictions = outputs_detection['node_predictions']  # Shape: (total_nodes, 1 + num_abnormal_classes )
        node_type_labels = outputs_detection['node_type_labels']  # (total_nodes,)
        node_confidences = outputs_detection['node_confidences']  # (total_nodes,)

        # Node confidence loss
        node_conf_logits = node_predictions[:, 0]
        node_conf_targets = node_confidences
        node_conf_loss = self.node_conf_loss_fn(node_conf_logits, node_conf_targets)


        # Node classification loss (abnormal nodes only)
        node_fg_mask = (node_type_labels != -1)
        if node_fg_mask.sum() > 0:
            fg_node_preds = node_predictions[node_fg_mask, 1:]  # (num_fg_nodes, 4)
            fg_node_labels = node_type_labels[node_fg_mask]  # Values in {0, 1, 2, 3}
            node_cls_loss = self.node_cls_loss_fn(fg_node_preds, fg_node_labels)
        else:
            node_cls_loss = torch.tensor(0.0, device=device)


        # Update label counts
        valid_mask = node_type_labels >= 0
        valid_labels = node_type_labels[valid_mask]
        batch_abnormal_node_counts = torch.bincount(valid_labels, minlength=4)
        self.epoch_node_label_counts += batch_abnormal_node_counts.to(self.epoch_node_label_counts.device)


        # Store predictions and targets
        self.training_node_conf_preds.append(node_conf_logits.cpu())
        self.training_node_conf_targets.append(node_conf_targets.cpu())

        # Binarize targets for classification
        node_conf_targets_binary = (node_conf_targets > 0).float()
        # node_conf_targets_binary = (node_type_labels != -1).float()  # 1 if abnormal, 0 if normal
        self.training_node_conf_targets_binary.append(node_conf_targets_binary.cpu())  # Binary targets
        if node_fg_mask.sum() > 0:
            self.training_node_preds.append(torch.argmax(fg_node_preds, dim=1).cpu())
            self.training_node_targets.append(fg_node_labels.cpu())



        # --------------inteval part ==========================
        # e.g., 39 + 19 + 9 = 67 if num_intervals_per_scale=[39, 19, 9]
        # We'll collect labels for the entire batch to compute the final loss

        # Extract target lists directly from batch_data
        conf_targets_list = [t.to(device) for t in batch_data['conf_targets']]
        cls_targets_list = [t.to(device) for t in batch_data['cls_targets']]
        box_targets_list = [t.to(device) for t in batch_data['box_targets']]


        # Concatenate targets across the batch
        conf_targets_cat = torch.cat(conf_targets_list, dim=0)  # Shape: [total_targets, 1] or [total_targets]
        cls_targets_cat = torch.cat(cls_targets_list, dim=0)  # Shape: [total_targets]
        box_targets_cat = torch.cat(box_targets_list, dim=0)  # Shape: [total_targets, 2]


        # pred_intervals = outputs_detection["pred_intervals"]  # Shape: [batch_size, total_intervals, 2]
        # interval_conf_logits = outputs_detection[ "pred_intervals_conf_logits"]  # Shape: [batch_size, total_intervals, 1]
        # interval_cls_logits = outputs_detection[ "pred_intervals_cls_logits"]  # Shape: [batch_size, total_intervals, num_classes - 1]

        # Reshape predictions to match concatenated targets
        # # Flatten batch and interval dimensions
        # conf_preds_cat = interval_conf_logits.view(-1, 1)  # Shape: [batch_size * total_intervals, 1]
        # cls_preds_cat = interval_cls_logits.view(-1, interval_cls_logits.size(
        #     -1))  # Shape: [batch_size * total_intervals, num_classes - 1]
        # all_pred_bounds = pred_intervals.view(-1, 2)  # Shape: [batch_size * total_intervals, 2]

        # Interval predictions
        conf_preds_cat = torch.cat(outputs_detection["pred_intervals_conf_logits"], dim=0)
        cls_preds_cat = torch.cat(outputs_detection["pred_intervals_cls_logits"], dim=0)
        all_pred_bounds = torch.cat(outputs_detection["pred_intervals"], dim=0)

         # Interval confidence loss
        conf_loss = self.interval_conf_loss_fn(conf_preds_cat, conf_targets_cat)
        # 6. Sub-type classification => only for intervals where conf=1
        # Foreground mask
        # fg_mask = (conf_targets_cat == 1)
        fg_mask = (conf_targets_cat > 0)

        # Classification Loss
        if fg_mask.sum() > 0:
            fg_cls_preds = cls_preds_cat[fg_mask]
            fg_cls_targets = cls_targets_cat[fg_mask]
            interval_cls_loss = self.interval_cls_loss_fn(fg_cls_preds, fg_cls_targets)
        else:
            interval_cls_loss = torch.tensor(0.0, device=self.device)

        # Localization Loss
        if fg_mask.sum() > 0:
            fg_pred_bounds = all_pred_bounds[fg_mask]
            fg_box_targets = box_targets_cat[fg_mask]
            loc_loss = self.iou_loss_1d(fg_pred_bounds, fg_box_targets, loss_type="neg_log_iou")
        else:
            loc_loss = torch.tensor(0.0, device=self.device)



        # Weighted losses
        node_conf_loss = self.node_conf_loss_weight * node_conf_loss
        node_level_loss = self.node_loss_weight * node_cls_loss

        conf_loss = self.interval_conf_loss_weight * conf_loss
        interval_cls_loss = self.interval_cls_loss_weight * interval_cls_loss
        loc_loss = self.interval_location_loss_weight * loc_loss

        # Total loss
        # total_loss = node_level_loss + node_conf_loss + conf_loss + interval_cls_loss + loc_loss


        node_conf_op_epochs = 160
        s1_transition_epochs = 40
        node_stage_epoch = node_conf_op_epochs + s1_transition_epochs

        s2_transition_epochs = 100
        interval_conf_op_epochs = 300
        s3_transition_epochs = 100
        interval_cls_loc_start_epoch = interval_conf_op_epochs + s3_transition_epochs


        # optimize the node  confidence : 0-160 epoch
        if self.current_epoch < node_conf_op_epochs:
            total_loss = node_conf_loss

        # optimize the  node classification  : 160-200 epoch
        elif self.current_epoch < node_stage_epoch:
            alpha1 = (self.current_epoch - node_conf_op_epochs) / s1_transition_epochs
            total_loss =  alpha1 * (node_conf_loss )  +  node_level_loss


        # optimize the interval confidence : 200-300 epoch
        elif node_stage_epoch <= self.current_epoch <  interval_conf_op_epochs :
            alpha2 = (self.current_epoch - node_stage_epoch) / s2_transition_epochs
            total_loss = alpha2 * (node_conf_loss + node_level_loss ) + conf_loss


        # optimize the interval class & localization: 300-400 epoch
        elif interval_conf_op_epochs <= self.current_epoch < interval_cls_loc_start_epoch:
            alpha3 = (self.current_epoch - interval_conf_op_epochs) / s3_transition_epochs
            total_loss = alpha3 * (node_conf_loss + node_level_loss + conf_loss ) +  interval_cls_loss + loc_loss

        # joint  optimize the total loss  : 400 epoch   ~
        else:
            total_loss = node_level_loss + node_conf_loss + conf_loss + interval_cls_loss + loc_loss



        # Logging
        self.log("train/node_conf_loss", node_conf_loss, prog_bar=True, on_epoch=True)
        self.log("train/node_level_loss", node_level_loss, prog_bar=True, on_epoch=True)
        self.log("train/interval_conf_loss", conf_loss, prog_bar=True, on_epoch=True)
        self.log("train/interval_cls_loss", interval_cls_loss, prog_bar=True, on_epoch=True)
        self.log("train/interval_loc_loss", loc_loss, prog_bar=True, on_epoch=True)
        self.log("train/total_loss", total_loss, prog_bar=True, on_epoch=True)

        # Store outputs
        self.training_step_outputs['node_level_loss'].append(node_level_loss)
        self.training_step_outputs['node_conf_loss'].append(node_conf_loss)
        self.training_step_outputs['interval_conf_loss'].append(conf_loss)
        self.training_step_outputs['interval_cls_loss'].append(interval_cls_loss)
        self.training_step_outputs['interval_loc_loss'].append(loc_loss)
        self.training_step_outputs['total_loss'].append(total_loss)


        return total_loss


    def print_matrix_with_labels(self, matrix, class_labels, title):
        # Sort by keys to maintain the order specified in conf_labels or cls_labels
        sorted_keys = sorted(class_labels.keys())
        # For node classification, use numerical labels as strings; for node confidence, use string labels
        label_names = [str(class_labels[key]) if isinstance(class_labels[key], int) else class_labels[key] for key in
                       sorted_keys]
        print(f"\n{title}:")
        header = " " * 15 + " ".join(f"{name:>15}" for name in label_names)
        print(header)
        for i, row in enumerate(matrix):
            row_string = f"{label_names[i]:15}" + " ".join(
                f"{val:15.4f}" if isinstance(val, float) else f"{val:15d}" for val in row)
            print(row_string)

    def on_train_epoch_end(self, outputs=None):
        """Process and compute metrics at the end of each training epoch"""
        # Node confidence metrics
        if len(self.training_node_conf_preds) > 0:
            all_conf_preds = torch.cat(self.training_node_conf_preds).detach().numpy()  # Predicted probabilities
            all_conf_targets = torch.cat(self.training_node_conf_targets).detach().numpy()  # Soft targets
            all_conf_targets_binary = torch.cat(
                self.training_node_conf_targets_binary).detach().numpy()  # Binary targets

            # Regression metrics
            mse = mean_squared_error(all_conf_targets, all_conf_preds)
            mae = mean_absolute_error(all_conf_targets, all_conf_preds)
            self.log("train_node_conf_mse", mse, prog_bar=True)
            self.log("train_node_conf_mae", mae, prog_bar=True)

            # Classification metrics
            conf_preds_binary = (all_conf_preds > 0.5).astype(int)
            conf_labels = [0, 1]
            conf_cm = confusion_matrix(all_conf_targets_binary, conf_preds_binary, labels=conf_labels)
            accuracy = (conf_cm.diagonal().sum()) / conf_cm.sum()
            precision = precision_score(all_conf_targets_binary, conf_preds_binary, zero_division=0)
            recall = recall_score(all_conf_targets_binary, conf_preds_binary, zero_division=0)

            conf_class_labels = {0: "Normal", 1: "Abnormal"}
            self.print_matrix_with_labels(conf_cm, conf_class_labels, "Node Confidence Training Confusion Matrix")

            with np.errstate(divide='ignore', invalid='ignore'):
                conf_cm_ratio = conf_cm.astype('float') / conf_cm.sum(axis=1)[:, np.newaxis]
                conf_cm_ratio = np.nan_to_num(conf_cm_ratio)
            self.print_matrix_with_labels(conf_cm_ratio, conf_class_labels,
                                          "Node Confidence Training Confusion Matrix (Ratio Format)")

            print(f"Node Confidence MSE: {mse:.4f}, MAE: {mae:.4f}")
            print(f"Node Confidence Confusion Matrix:\n{conf_cm}")
            print(f"Node Confidence Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

            self.log("train_node_conf_accuracy", accuracy, prog_bar=True, on_epoch=True)
            self.log("train_node_conf_precision", precision, prog_bar=True, on_epoch=True)
            self.log("train_node_conf_recall", recall, prog_bar=True, on_epoch=True)

        # Node classification metrics
        if len(self.training_node_preds) > 0:
            all_node_preds = torch.cat(self.training_node_preds).detach().numpy()
            all_node_targets = torch.cat(self.training_node_targets).detach().numpy()

            cls_labels = [0, 1, 2, 3]
            cls_class_labels = node_cls_labels
            node_cm = confusion_matrix(all_node_targets, all_node_preds, labels=cls_labels)

            self.print_matrix_with_labels(node_cm, cls_class_labels, "Node Classification Training Confusion Matrix")

            with np.errstate(divide='ignore', invalid='ignore'):
                node_cm_ratio = node_cm.astype('float') / node_cm.sum(axis=1)[:, np.newaxis]
                node_cm_ratio = np.nan_to_num(node_cm_ratio)
            self.print_matrix_with_labels(node_cm_ratio, cls_class_labels,
                                          "Node Classification Training Confusion Matrix (Ratio Format)")

            # Save heatmap (automatically overwrites previous)
            current_epoch = self.current_epoch
            cls_label_names = list(node_cls_labels.keys())
            save_confusion_matrix_heatmap(node_cm_ratio, cls_label_names, current_epoch)

            # Macro-averaged metrics
            node_precision = precision_score(all_node_targets, all_node_preds, average='macro', labels=cls_labels,
                                             zero_division=0)
            node_recall = recall_score(all_node_targets, all_node_preds, average='macro', labels=cls_labels,
                                       zero_division=0)
            node_f1 = f1_score(all_node_targets, all_node_preds, average='macro', labels=cls_labels,
                               zero_division=0)
            print(
                f"\nNode Classification Training Macro Metrics: F1 score: {node_f1:.4f}, Precision: {node_precision:.4f}, Recall: {node_recall:.4f}\n")

            # Weighted metrics
            weighted_precision = precision_score(all_node_targets, all_node_preds, average='weighted',
                                                 labels=cls_labels, zero_division=0)
            weighted_recall = recall_score(all_node_targets, all_node_preds, average='weighted', labels=cls_labels,
                                           zero_division=0)
            weighted_f1 = f1_score(all_node_targets, all_node_preds, average='weighted', labels=cls_labels,
                                   zero_division=0)
            print(
                f"Node Classification Training Weighted Metrics: Precision: {weighted_precision:.4f}, Recall: {weighted_recall:.4f}, F1 Score: {weighted_f1:.4f}\n")

            # Per-category metrics
            category_precision = precision_score(all_node_targets, all_node_preds, labels=cls_labels, average=None,
                                                 zero_division=0)
            category_recall = recall_score(all_node_targets, all_node_preds, labels=cls_labels, average=None,
                                           zero_division=0)
            category_f1 = f1_score(all_node_targets, all_node_preds, labels=cls_labels, average=None,
                                   zero_division=0)

            for i, label in enumerate(cls_labels):
                print(
                    f"Category: {cls_label_names[label]}, Precision: {category_precision[i]:.4f}, Recall: {category_recall[i]:.4f}, F1: {category_f1[i]:.4f}")

            self.log("train/node_precision", node_precision, on_epoch=True)
            self.log("train/node_recall", node_recall, on_epoch=True)
            self.log("train/node_f1_score", node_f1, on_epoch=True)

            if node_f1 > self.best_node_f1:
                self.best_node_f1 = node_f1

        # Log node label counts
        print("\nTraining epoch-wide node label counts")
        for i, count in enumerate(self.epoch_node_label_counts.tolist()):
            print(f"Node_label_{i - 4 if i == 4 else i}: {count}")
            self.log(f'node_label_{i - 4 if i == 4 else i}_count', count)
        self.epoch_node_label_counts.zero_()

        # Calculate and log epoch metrics
        epoch_metrics = {
            key: torch.stack(values).mean().item()
            for key, values in self.training_step_outputs.items()
        }
        self.log("train/node_loss/epoch", epoch_metrics['node_level_loss'], on_epoch=True)
        self.log("train/node_conf_loss/epoch", epoch_metrics['node_conf_loss'], on_epoch=True)
        self.log("train/interval_conf_loss/epoch", epoch_metrics['interval_conf_loss'], on_epoch=True)
        self.log("train/interval_cls_loss/epoch", epoch_metrics['interval_cls_loss'], on_epoch=True)
        self.log("train/interval_loc_loss/epoch", epoch_metrics['interval_loc_loss'], on_epoch=True)
        self.log("train/total_loss/epoch", epoch_metrics['total_loss'], on_epoch=True)

        # Clear lists
        self.training_node_preds = []
        self.training_node_targets = []
        self.training_node_conf_preds = []
        self.training_node_conf_targets = []
        self.training_node_conf_targets_binary = []
        for key in self.training_step_outputs:
            self.training_step_outputs[key] = []


    def on_before_zero_grad(self, *args, **kwargs) :
        # "Called per batch:
        # This hook is executed before optimizer.zero_grad() during training,
        # which happens after every batch."
        self.update_ema(
            self.hparams["training"]["ema_factor"],
            # self.scheduler["scheduler"].step_num, # 由于 scheduler 中设置的更新是 step , 代表每个batch 更新；
            self.global_step
        )


    def validation_step(self, batch_data, batch_idx):
        device = self.device

        # Move data to device
        spectrograms = [s.clone().detach().to(device) for s in batch_data['spectrograms']]
        frame_labels = [f.clone().detach().to(device) for f in batch_data['frame_labels']]
        record_binary_labels = torch.tensor(batch_data['record_binary_label'], dtype=torch.long).to(device)
        batch_audio_names = batch_data['c_ex_mixtures']
        chest_pos = [c.clone().detach().to(device) for c in batch_data['chest_info']]
        vad_timestamps = batch_data['vad_timestamps']
        audios_dur = [d.clone().detach().to(device) for d in batch_data['audio_dur']]
        anchor_intervals = batch_data["anchor_intervals"]
        assignments = batch_data["assignments"]

        batch_data_device = {
            'spectrograms': spectrograms,
            'frame_labels': frame_labels,
            'c_ex_mixtures': batch_audio_names,
            'vad_timestamps': vad_timestamps,
            'chest_loc': chest_pos,
            'audio_dur': audios_dur,
            "anchor_intervals": anchor_intervals,
            "assignments": assignments
        }

        # Forward pass
        outputs_detection = self.sed_net(batch_data_device)

        # Interval inference (unchanged)
        batch_detection_names = outputs_detection['batch_audio_names']

        batch_audio_duration = []
        batch_audio_timestamp = []

        filenames_synth = [x for x in batch_detection_names
                           if Path(x).parent == Path(self.hparams["data"]["eval_folder_4k"])]
        valid_df = pd.read_csv(self.hparams["data"]["valid_dur"], sep='\t')
        for file in filenames_synth:
            file = os.path.basename(file)
            duration = valid_df.loc[valid_df['filename'] == file, 'duration'].values[0]
            batch_audio_duration.append(duration)
            sample_info = self.valid_data.examples[file]
            batch_audio_timestamp.append(sample_info['vad_timestamps'])

        decoded_abnormal_pred = {}
        for idx, conf_th in enumerate(self.hparams["training"]["val_conf_thresholds"]):
            iou_th = self.hparams["training"]["val_iou_NMS"][idx]
            # batch_results = self.batch_inference_interval(outputs_detection, conf_threshold=conf_th,
            #                                               nms_iou_threshold=iou_th)
            batch_results = self.inference_batch_interval(outputs_detection, conf_threshold=conf_th,
                                                          nms_iou_threshold=iou_th)
            decoded_df = self.decode_results(batch_results, batch_audio_names, interval_4labels)
            decoded_abnormal_pred[conf_th] = decoded_df
        for conf_th in self.val_buffer_intervals.keys():
            self.val_buffer_intervals[conf_th] = pd.concat(
                [self.val_buffer_intervals[conf_th], decoded_abnormal_pred[conf_th]], ignore_index=True)

        # Node predictions and labels
        node_predictions = outputs_detection['node_predictions']
        node_type_labels = outputs_detection['node_type_labels']
        node_confidences = outputs_detection['node_confidences']

        # Node confidence loss
        node_conf_logits = node_predictions[:, 0]
        node_conf_targets = node_confidences
        node_conf_loss = self.node_conf_loss_fn(node_conf_logits, node_conf_targets)

        # Node classification loss
        node_fg_mask = (node_type_labels != -1)
        if node_fg_mask.sum() > 0:
            fg_node_preds = node_predictions[node_fg_mask, 1:]
            fg_node_labels = node_type_labels[node_fg_mask]
            node_cls_loss = self.node_cls_loss_fn(fg_node_preds, fg_node_labels)
        else:
            node_cls_loss = torch.tensor(0.0, device=device)


        # Update label counts
        valid_mask = node_type_labels >= 0
        valid_labels = node_type_labels[valid_mask]
        batch_abnormal_node_counts = torch.bincount(valid_labels, minlength=4)
        self.valid_epoch_node_label_counts += batch_abnormal_node_counts.to(self.valid_epoch_node_label_counts.device)

        # Binarize targets for classification
        node_conf_targets_binary = (node_conf_targets > 0).float()

        # Store predictions
        # Store predictions and targets
        self.validation_node_conf_preds.append(torch.sigmoid(node_conf_logits).cpu())  # For regression
        self.validation_node_conf_targets.append(node_conf_targets.cpu())  # Soft targets
        self.validation_node_conf_targets_binary.append(node_conf_targets_binary.cpu())  # Binary targets
        if node_fg_mask.sum() > 0:
            self.validation_node_preds.append(torch.argmax(fg_node_preds, dim=1).cpu())
            self.validation_node_targets.append(fg_node_labels.cpu())




        # Interval losses
        conf_targets_list = [t.to(device) for t in batch_data['conf_targets']]
        cls_targets_list = [t.to(device) for t in batch_data['cls_targets']]
        box_targets_list = [t.to(device) for t in batch_data['box_targets']]

        # Concatenate across the batch
        conf_targets_cat = torch.cat(conf_targets_list, dim=0)
        cls_targets_cat = torch.cat(cls_targets_list, dim=0)
        box_targets_cat = torch.cat(box_targets_list, dim=0)
        #


        # Interval predictions
        conf_preds_cat = torch.cat(outputs_detection["pred_intervals_conf_logits"], dim=0)
        cls_preds_cat = torch.cat(outputs_detection["pred_intervals_cls_logits"], dim=0)
        all_pred_bounds = torch.cat(outputs_detection["pred_intervals"], dim=0)


        conf_loss = self.interval_conf_loss_fn(conf_preds_cat, conf_targets_cat)
        fg_mask = (conf_targets_cat > 0)
        if fg_mask.sum() > 0:
            fg_cls_preds = cls_preds_cat[fg_mask]
            fg_cls_targets = cls_targets_cat[fg_mask]
            interval_cls_loss = self.interval_cls_loss_fn(fg_cls_preds, fg_cls_targets)
            fg_pred_bounds = all_pred_bounds[fg_mask]
            fg_box_targets = box_targets_cat[fg_mask]
            loc_loss = self.iou_loss_1d(fg_pred_bounds, fg_box_targets, loss_type="neg_log_iou")
        else:
            interval_cls_loss = torch.tensor(0.0, device=device)
            loc_loss = torch.tensor(0.0, device=device)

        # Weighted losses
        node_conf_loss = self.node_conf_loss_weight * node_conf_loss
        node_level_loss = self.node_loss_weight * node_cls_loss

        conf_loss = self.interval_conf_loss_weight * conf_loss
        interval_cls_loss = self.interval_cls_loss_weight * interval_cls_loss
        loc_loss = self.interval_location_loss_weight * loc_loss

        # total_loss = node_level_loss + node_conf_loss + conf_loss + interval_cls_loss + loc_loss

        node_conf_op_epochs = 160
        s1_transition_epochs = 40
        node_stage_epoch = node_conf_op_epochs + s1_transition_epochs

        s2_transition_epochs = 100
        interval_conf_op_epochs = 300
        s3_transition_epochs = 100
        interval_cls_loc_start_epoch = interval_conf_op_epochs + s3_transition_epochs

        # optimize the node  confidence : 0-60 epoch
        if self.current_epoch < node_conf_op_epochs:
            total_loss = node_conf_loss

        # optimize the  node classification  : 60-100 epoch
        elif self.current_epoch < node_stage_epoch:
            alpha1 = (self.current_epoch - node_conf_op_epochs) / s1_transition_epochs
            total_loss = alpha1 * (node_conf_loss) + node_level_loss


        # optimize the interval confidence : 100-160 epoch
        elif node_stage_epoch <= self.current_epoch < interval_conf_op_epochs:
            alpha2 = (self.current_epoch - node_stage_epoch) / s2_transition_epochs
            total_loss = alpha2 * (node_conf_loss + node_level_loss) + conf_loss


        # optimize the interval class & localization: 160-240 epoch
        elif interval_conf_op_epochs <= self.current_epoch < interval_cls_loc_start_epoch:
            alpha3 = (self.current_epoch - interval_conf_op_epochs) / s3_transition_epochs
            total_loss = alpha3 * (node_conf_loss + node_level_loss + conf_loss) + interval_cls_loss + loc_loss

        # joint  optimize the total loss  : 400 epoch   ~
        else:
            total_loss = node_level_loss + node_conf_loss + conf_loss + interval_cls_loss + loc_loss

        # Logging
        self.log("valid/node_conf_loss", node_conf_loss, prog_bar=True, on_epoch=True)
        self.log("valid/node_level_loss", node_level_loss, prog_bar=True, on_epoch=True)
        self.log("valid/interval_conf_loss", conf_loss, prog_bar=True, on_epoch=True)
        self.log("valid/interval_cls_loss", interval_cls_loss, prog_bar=True, on_epoch=True)
        self.log("valid/interval_loc_loss", loc_loss, prog_bar=True, on_epoch=True)
        self.log("valid/total_loss", total_loss, prog_bar=True, on_epoch=True)

        # Store outputs
        self.valid_step_outputs['node_level_loss'].append(node_level_loss)
        self.valid_step_outputs['node_conf_loss'].append(node_conf_loss)
        self.valid_step_outputs['interval_conf_loss'].append(conf_loss)
        self.valid_step_outputs['interval_cls_loss'].append(interval_cls_loss)
        self.valid_step_outputs['interval_loc_loss'].append(loc_loss)
        self.valid_step_outputs['total_loss'].append(total_loss)

        # # Store node predictions and targets
        # self.validation_node_conf_preds.append(node_conf_logits.cpu())
        # self.validation_node_conf_targets.append(node_conf_targets.cpu())
        # if node_fg_mask.sum() > 0:
        #     self.validation_node_preds.append(torch.argmax(fg_node_preds, dim=1).cpu())
        return total_loss


    # note, 计算各种类型的 分数；
    def on_validation_epoch_end(self, outputs=None):

        # Interval event-based metrics
        DEFAULT_EVENT_SEGMENT_SCORED = (0.0,)
        try:
            scores = []
            for idx, cur_thread in enumerate(self.hparams["training"]["val_conf_thresholds"]):
                cur_iou_NMS = self.hparams["training"]["val_iou_NMS"][idx]
                print(f"\nUsing confidence & NMS threshold: conf={cur_thread}, iou_NMS={cur_iou_NMS}")
                score = compute_event_based_metrics(
                    self.val_buffer_intervals[cur_thread],
                    self.hparams["data"]["valid_tsv"],
                    save_dir=os.path.join(self.hparams["log_dir"], f"conf{idx}_pred_interval_event_metrics")
                )
                scores.append(score[0])
        except Exception as e:
            print(f"Error in compute_event_based_metrics: {e}")
            scores = [DEFAULT_EVENT_SEGMENT_SCORED[0]] * len(self.hparams["training"]["val_conf_thresholds"])

        node_event_thread_f_score = max(scores)
        obj_metric = torch.tensor(node_event_thread_f_score)

        if node_event_thread_f_score > self.best_val_score:
            self.best_val_score = node_event_thread_f_score
            self.best_model_weights = deepcopy(self.sed_net.state_dict())
            self.best_ema_weights = deepcopy(self.ema_model.state_dict())

        self.log('best_val_class_wise_score', self.best_val_score, prog_bar=True)
        self.log_counts(self.valid_epoch_node_label_counts, "node_label", "node label")

        print(f"\tval/obj_metric: {obj_metric}"
              f"\tval/node_level/event_based_class_wise_average_f1_score: {node_event_thread_f_score}\n")

        self.log("val/obj_metric", obj_metric, prog_bar=True)
        self.log("val/node_level/event_based_class_wise_average_f1_score", node_event_thread_f_score, prog_bar=True)

        """Process metrics at the end of validation epoch"""
        # Node confidence metrics
        # Node confidence metrics
        if len(self.validation_node_conf_preds) > 0:
            all_conf_preds = torch.cat(self.validation_node_conf_preds).numpy()  # Predicted probabilities
            all_conf_targets = torch.cat(self.validation_node_conf_targets).numpy()  # Soft targets
            all_conf_targets_binary = torch.cat(self.validation_node_conf_targets_binary).numpy()  # Binary targets

            # Regression metrics
            mse = mean_squared_error(all_conf_targets, all_conf_preds)
            mae = mean_absolute_error(all_conf_targets, all_conf_preds)
            self.log("val_node_conf_mse", mse, prog_bar=True)
            self.log("val_node_conf_mae", mae, prog_bar=True)

            # Classification metrics
            conf_preds_binary = (all_conf_preds > 0.5).astype(int)
            conf_labels = [0, 1]
            conf_cm = confusion_matrix(all_conf_targets_binary, conf_preds_binary, labels=conf_labels)
            accuracy = (conf_cm.diagonal().sum()) / conf_cm.sum()
            precision = precision_score(all_conf_targets_binary, conf_preds_binary)
            recall = recall_score(all_conf_targets_binary, conf_preds_binary)

            print(f"Node Confidence MSE: {mse:.4f}, MAE: {mae:.4f}")
            print(f"Node Confidence Confusion Matrix:\n{conf_cm}")
            print(f"Node Confidence Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

            self.log("val_node_conf_accuracy", accuracy, prog_bar=True, on_epoch=True)
            self.log("val_node_conf_precision", precision, prog_bar=True, on_epoch=True)
            self.log("val_node_conf_recall", recall, prog_bar=True, on_epoch=True)

        # Node classification metrics
        if len(self.validation_node_preds) > 0:
            all_node_preds = torch.cat(self.validation_node_preds).numpy()
            all_node_targets = torch.cat(self.validation_node_targets).numpy()

            cls_labels = [0, 1, 2, 3]  # Corresponds to node_cls_labels
            node_cm = confusion_matrix(all_node_targets, all_node_preds, labels=cls_labels)

            self.print_matrix_with_labels(node_cm, node_cls_labels,
                                          "Node Classification Validation Confusion Matrix")

            # Convert OrderedDict to list for consistent indexing
            cls_label_names = list(node_cls_labels.keys())
            # Normalize node_cm by its own row sums
            with np.errstate(divide='ignore', invalid='ignore'):
                node_cm_ratio = node_cm.astype('float') / node_cm.sum(axis=1)[:, np.newaxis]
                node_cm_ratio = np.nan_to_num(node_cm_ratio)
            self.print_matrix_with_labels(node_cm_ratio, node_cls_labels,
                                          "Node Classification Validation Confusion Matrix (Ratio Format)")

            # Save heatmap (automatically overwrites previous)
            current_epoch = self.current_epoch  # Assuming you're using PyTorch Lightning
            save_confusion_matrix_heatmap(node_cm_ratio, cls_label_names, current_epoch)

            # Macro-averaged metrics
            node_precision = precision_score(all_node_targets, all_node_preds, average='macro', labels=cls_labels,
                                             zero_division=0)
            node_recall = recall_score(all_node_targets, all_node_preds, average='macro', labels=cls_labels,
                                       zero_division=0)
            node_f1 = f1_score(all_node_targets, all_node_preds, average='macro', labels=cls_labels,
                               zero_division=0)
            print(
                f"\nNode Classification Validation Macro Metrics: F1 score: {node_f1:.4f}, Precision: {node_precision:.4f}, Recall: {node_recall:.4f}\n")

            # Weighted metrics
            weighted_precision = precision_score(all_node_targets, all_node_preds, average='weighted',
                                                 labels=cls_labels, zero_division=0)
            weighted_recall = recall_score(all_node_targets, all_node_preds, average='weighted', labels=cls_labels,
                                           zero_division=0)
            weighted_f1 = f1_score(all_node_targets, all_node_preds, average='weighted', labels=cls_labels,
                                   zero_division=0)
            print(
                f"Node Classification Validation Weighted Metrics: Precision: {weighted_precision:.4f}, Recall: {weighted_recall:.4f}, F1 Score: {weighted_f1:.4f}\n")

            # Per-category metrics
            category_precision = precision_score(all_node_targets, all_node_preds, labels=cls_labels, average=None,
                                                 zero_division=0)
            category_recall = recall_score(all_node_targets, all_node_preds, labels=cls_labels, average=None,
                                           zero_division=0)
            category_f1 = f1_score(all_node_targets, all_node_preds, labels=cls_labels, average=None,
                                   zero_division=0)

            for i, label in enumerate(cls_labels):
                print(
                    f"Category: {cls_label_names[label]}, Precision: {category_precision[i]:.4f}, Recall: {category_recall[i]:.4f}, F1: {category_f1[i]:.4f}")

            self.log("valid/node_precision", node_precision, on_epoch=True)
            self.log("valid/node_recall", node_recall, on_epoch=True)
            self.log("valid/node_f1_score", node_f1, on_epoch=True)

            # plt.figure(figsize=(10, 8))
            # sns.heatmap(node_cm_ratio, annot=True, fmt='.2f',
            #             xticklabels=cls_label_names,
            #             yticklabels=cls_label_names)
            # plt.title('Normalized Confusion Matrix')
            # plt.ylabel('True Label')
            # plt.xlabel('Predicted Label')
            # plt.show()

        # Log epoch metrics
        epoch_metrics = {
            key: torch.stack(values).mean().item()
            for key, values in self.valid_step_outputs.items()
        }
        self.log("valid/node_loss/epoch", epoch_metrics['node_level_loss'], on_epoch=True)
        self.log("valid/node_conf_loss/epoch", epoch_metrics['node_conf_loss'], on_epoch=True)
        self.log("valid/interval_conf_loss/epoch", epoch_metrics['interval_conf_loss'], on_epoch=True)
        self.log("valid/interval_cls_loss/epoch", epoch_metrics['interval_cls_loss'], on_epoch=True)
        self.log("valid/interval_loc_loss/epoch", epoch_metrics['interval_loc_loss'], on_epoch=True)
        self.log("valid/total_loss/epoch", epoch_metrics['total_loss'], on_epoch=True)

        # Clear buffers
        # Clear stored predictions
        self.validation_node_conf_preds.clear()
        self.validation_node_conf_targets.clear()
        self.validation_node_conf_targets_binary.clear()

        self.validation_node_preds.clear()
        self.validation_node_targets.clear()

        self.val_buffer_intervals = {th: pd.DataFrame() for th in self.val_buffer_intervals.keys()}
        self.valid_epoch_node_label_counts.zero_()
    def log_counts(self,counts, prefix, description=""):
        print(f"\n Validation epoch-wide {description} counts")
        for i, count in enumerate(counts.tolist()):
            print(f"{prefix}_{i} number: {count}")
            self.log(f'{prefix}_{i}_count', count)
        counts.zero_()  # 重置计数

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
        # Initial learning rates
        lr_node = 1e-3
        lr_interval = 1e-3

        params = [
            {"params": self.shared_params + self.node_update_conf_cls_params, "lr": lr_node},
            {"params": self.interval_conf_cls_refine_params, "lr": lr_interval}
        ]

        optimizer = torch.optim.Adam(params)
        custom_scheduler = CustomScheduler(
            optimizer,
            milestones_node=[160, 300, 350],
            gamma_node=0.2,
            milestones_interval=[300, 400, 500],
            gamma_interval=0.2
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": custom_scheduler, "interval": "epoch", "name": "custom_scheduler"}
        }
    # Custom collate function to convert spectrograms to Data objects during batch preparation

    def lr_scheduler_step(self, scheduler, metric=None):
        print(f"\n Current Epoch: {self.current_epoch}, Scheduler last_epoch: {scheduler.last_epoch}")
        scheduler.step(epoch=self.current_epoch)


    def apply_time_shift(self,item, audio_dur, train_flag=False, apply_time_shift=False):
        if not (train_flag and item['record_binary_label'].item() == 1 and apply_time_shift):
            return item

        vad_timestamps = item['vad_timestamps']
        if vad_timestamps:
            onsets = [vad['start'] for vad in vad_timestamps]
            offsets = [vad['end'] for vad in vad_timestamps]
            min_onset = min(onsets)
            max_offset = max(offsets)
            max_forward_shift = min_onset
            max_backward_shift = audio_dur - max_offset
        else:
            max_forward_shift = audio_dur / 2
            max_backward_shift = audio_dur / 2

        shift_seconds = random.uniform(-max_forward_shift, max_backward_shift)
        updated_vad = []
        for vad in vad_timestamps:
            new_start = vad['start'] + shift_seconds
            new_end = vad['end'] + shift_seconds
            wrapped_vads = []
            if new_start < 0 or new_end > audio_dur:
                new_start = new_start % audio_dur
                new_end = new_end % audio_dur
                if new_start > new_end:
                    if new_start < audio_dur:
                        wrapped_vad_1 = vad.copy()
                        wrapped_vad_1['start'] = new_start
                        wrapped_vad_1['end'] = audio_dur
                        wrapped_vads.append(wrapped_vad_1)
                    if new_end > 0:
                        wrapped_vad_2 = vad.copy()
                        wrapped_vad_2['start'] = 0
                        wrapped_vad_2['end'] = new_end
                        wrapped_vads.append(wrapped_vad_2)
                else:
                    wrapped_vad = vad.copy()
                    wrapped_vad['start'] = new_start
                    wrapped_vad['end'] = new_end
                    wrapped_vads.append(wrapped_vad)
            else:
                if new_start < new_end:
                    updated_vad_entry = vad.copy()
                    updated_vad_entry['start'] = new_start
                    updated_vad_entry['end'] = new_end
                    wrapped_vads.append(updated_vad_entry)
            updated_vad.extend(wrapped_vads)
        item['vad_timestamps'] = updated_vad

        # Regenerate assignments
        gt_intervals = [(vad['start'], vad['end']) for vad in item['vad_timestamps']]
        gt_labels = [vad['event_label'] for vad in item['vad_timestamps']]
        gt_boxes = torch.tensor(gt_intervals) if gt_intervals else torch.empty((0, 2))
        anchor_boxes = item['anchor_intervals']
        iou_threshold = 0.5
        if len(gt_boxes) == 0 or gt_boxes.dim() < 2:
            assignments = [{'conf': 0.0, 'cls': -1, 'box': [0.0, 0.0]} for _ in range(len(anchor_boxes))]
        else:
            iou_matrix = self.compute_iou_matrix(anchor_boxes, gt_boxes)
            assignments = []
            max_ious, max_gt_indices = iou_matrix.max(dim=1)
            for a_idx in range(len(anchor_boxes)):
                if max_ious[a_idx] >= iou_threshold:
                    gt_idx = max_gt_indices[a_idx].item()
                    gt_label = gt_labels[gt_idx]
                    if gt_label != 'Normal':
                        assignments.append({
                            'conf': max_ious[a_idx].item(),
                            'cls': interval_4labels[gt_label],
                            'box': gt_intervals[gt_idx]
                        })
                    else:
                        assignments.append({
                            'conf': 0.0,
                            'cls': -1,
                            'box': [0.0, 0.0]
                        })
                else:
                    assignments.append({
                        'conf': 0.0,
                        'cls': -1,
                        'box': [0.0, 0.0]
                    })
        item['assignments'] = assignments
        item['conf_targets'] = torch.tensor([a['conf'] for a in assignments], dtype=torch.float)
        item['cls_targets'] = torch.tensor([a['cls'] for a in assignments], dtype=torch.long)
        item['box_targets'] = torch.tensor([a['box'] for a in assignments], dtype=torch.float)

        return item


    def custom_collate(self, batch, train_flag, audio_aug_prob=0.2, time_shift_prob=0.5):
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

        batch_size = len(batch)
        apply_time_shift = random.random() > time_shift_prob if train_flag else False
        apply_general_aug_indices = random.sample(range(batch_size),
                                                  int(batch_size * audio_aug_prob)) if train_flag else []

        processed_batch = []
        for i, item in enumerate(batch):
            # General augmentation already applied in __getitem__
            # Apply time shifting (batch-level) here
            item = self.apply_time_shift(
                item,
                audio_dur=item['audio_duration'].item(),
                train_flag=train_flag,
                apply_time_shift=apply_time_shift
            )
            processed_batch.append(item)

        return {
            'spectrograms': [item['spectrogram'] for item in processed_batch],
            'frame_labels': [item['frame_labels'] for item in processed_batch],
            'c_ex_mixtures': [item['c_ex_mixture'] for item in processed_batch],
            'record_binary_label': [item['record_binary_label'] for item in processed_batch],
            'chest_info': [item['chest_pos'] for item in processed_batch],
            # 'gender_info': [item['gender_info'] for item in processed_batch],
            'vad_timestamps': [item['vad_timestamps'] for item in processed_batch],
            'audio_dur': [item['audio_duration'] for item in processed_batch],
            'anchor_intervals': [item['anchor_intervals'] for item in processed_batch],
            'assignments': [item['assignments'] for item in processed_batch],
            'conf_targets': [item['conf_targets'] for item in processed_batch],
            'cls_targets': [item['cls_targets'] for item in processed_batch],
            'box_targets': [item['box_targets'] for item in processed_batch],
        }





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

        # Wrapper dataset to apply general augmentation randomly

        train_dataset = AugmentWrapper(combined_dataset, audio_aug_prob=0.2)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.hparams["training"]["batch_size"],
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=partial(self.custom_collate, train_flag=True),
            #collate_fn=partial(self.custom_collate, train_flag= False),

            pin_memory=True,
        )

        return train_loader

        # return self.train_dataloader




    def val_dataloader(self):
        #self.val_loader =  torch_geometric.data.DataLoader(
        self.val_loader = DataLoader(
            self.valid_data,
            batch_size=self.hparams["training"]["batch_size_val"],
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            # collate_fn=self.custom_collate
            collate_fn = partial(self.custom_collate, train_flag=False),
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

    # def on_train_batch_end(self, outputs, batch, batch_idx):
    #     # Get the optimizers
    #     optimizer = self.optimizers()  # List of optimizers
    #
    #     # Extract learning rates for each parameter group
    #     lr_node_conf_cls = optimizer[0].param_groups[0]['lr']  # Node confidence/classification
    #     lr_interval_unify = optimizer[1].param_groups[0]['lr']  # Interval unification
    #
    #     # Log learning rates at the step level
    #     self.log('lr_node_conf_cls', lr_node_conf_cls, on_step=True, on_epoch=False)
    #     self.log('lr_interval_unify', lr_interval_unify, on_step=True, on_epoch=False)
    def on_train_epoch_start(self):

        # note,  对损失权重进行调整，
        # current_epoch = self.current_epoch

        optimizer = self.optimizers()  # Single optimizer
        # Log learning rate for each parameter group
        lr_node_interval_conf_cls = optimizer.param_groups[0]['lr']  # Node group
        lr_interval = optimizer.param_groups[1]['lr']  # Interval group
        # lr_shared = optimizer.param_groups[2]['lr']  # shared

        print(f"Epoch {self.current_epoch}: lr_node_interval_conf_cls={lr_node_interval_conf_cls}, "
              f"lr_interval_off={lr_interval}, ")

        self.log('lr_node', lr_node_interval_conf_cls, on_epoch=True)
        self.log('lr_interval', lr_interval, on_epoch=True)



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
# class FocalLossBinary(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
#         """
#         Args:
#             alpha (float): Weighting factor for the positive class.
#             α: Controls the balance between positive and negative classes.
#             A higher    α increases the weight of positive class losses,
#             while a lower α increases the weight of negative class losses.
#
#             α  =α for positive examples (class 1, Abnormal).
#             α  =1−α for negative examples (class 0, Normal)
#             gamma (float): Focusing parameter that adjusts the rate at which easy examples are down-weighted.
#             reduction (str): 'mean', 'sum', or 'none' for the loss reduction method.
#         """
#         super(FocalLossBinary, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction
#
#     def forward(self, inputs, targets):
#         """
#         Args:
#             inputs (Tensor): Raw predictions (logits) of shape (N,) or (N, 1).
#             targets (Tensor): Ground truth binary labels of shape (N,).
#         Returns:
#             Tensor: Focal loss value.
#         """
#         # Ensure inputs are 1D (if necessary)
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
#
#         # Compute binary cross entropy loss without reduction
#         BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
#
#         # Convert logits to probabilities
#         probas = torch.sigmoid(inputs)
#         # For each instance, p_t is the model's estimated probability for the true class
#         p_t = targets * probas + (1 - targets) * (1 - probas)
#
#         # Compute the focal factor
#         focal_factor = (1 - p_t) ** self.gamma
#
#         # Apply alpha weighting: alpha for positive, (1-alpha) for negative examples
#         alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
#
#         # Combine the factors with the BCE loss
#         loss = alpha_factor * focal_factor * BCE_loss
#
#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         else:
#             return loss


from torch.optim.lr_scheduler import _LRScheduler
class CustomScheduler(_LRScheduler):
    def __init__(self, optimizer, milestones_node, gamma_node, milestones_interval, gamma_interval, last_epoch=-1):
        self.milestones_node = milestones_node
        self.gamma_node = gamma_node
        self.milestones_interval = milestones_interval
        self.gamma_interval = gamma_interval
        # Store initial learning rates
        self.initial_lr_node = optimizer.param_groups[0]['lr']
        self.initial_lr_interval = optimizer.param_groups[1]['lr']
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Compute learning rates for the current epoch
        lr_node = self._compute_lr(self.last_epoch, self.milestones_node, self.gamma_node, self.initial_lr_node)
        lr_interval = self._compute_lr(self.last_epoch, self.milestones_interval, self.gamma_interval, self.initial_lr_interval)
        return [lr_node, lr_interval]  # Must return a list matching the number of param groups (2)

    def _compute_lr(self, epoch, milestones, gamma, initial_lr):
        # MultiStepLR logic: count milestones passed and apply gamma
        exponent = sum(epoch >= m for m in milestones)
        return initial_lr * (gamma ** exponent)

    def step(self, epoch=None):
        # Handle epoch manually if provided, otherwise increment last_epoch
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
        # Update learning rates (PyTorch's _LRScheduler will call get_lr and set them)
        super().step()


class AugmentWrapper(Dataset):
    def __init__(self, dataset, audio_aug_prob=0.2):
        self.dataset = dataset
        self.audio_aug_prob = audio_aug_prob
        # If dataset is ConcatDataset, store the underlying datasets
        if isinstance(self.dataset, ConcatDataset):
            self.sub_datasets = []
            for sub_dataset in self.dataset.datasets:
                # Unwrap Subset to get the underlying RespiraGnnSet
                if isinstance(sub_dataset, Subset):
                    self.sub_datasets.append(sub_dataset.dataset)
                else:
                    self.sub_datasets.append(sub_dataset)
            # Compute cumulative lengths for index mapping
            self.cumulative_sizes = self.dataset.cumulative_sizes
        else:
            self.sub_datasets = [self.dataset]
            self.cumulative_sizes = [len(self.dataset)]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        apply_general_aug = random.random() < self.audio_aug_prob
        # Find the correct sub-dataset and adjusted index
        if isinstance(self.dataset, ConcatDataset):
            # Find which dataset this index belongs to
            dataset_idx = 0
            for i, cum_size in enumerate(self.cumulative_sizes):
                if idx < cum_size:
                    dataset_idx = i
                    break
            # Adjust the index for the sub-dataset
            if dataset_idx > 0:
                idx -= self.cumulative_sizes[dataset_idx - 1]
            # If the sub-dataset is a Subset, map the index to the underlying dataset
            if isinstance(self.dataset.datasets[dataset_idx], Subset):
                idx = self.dataset.datasets[dataset_idx].indices[idx]
            # Call the underlying RespiraGnnSet.__getitem__
            return self.sub_datasets[dataset_idx].__getitem__(idx, apply_general_aug=apply_general_aug)
        else:
            # If not ConcatDataset, call directly
            return self.dataset.__getitem__(idx, apply_general_aug=apply_general_aug)


import matplotlib.pyplot as plt
import seaborn as sns
def save_confusion_matrix_heatmap(node_cm_ratio, cls_label_names, epoch, save_dir='confusion_matrices'):
    """Save confusion matrix heatmap and overwrite previous epoch's image"""
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Create figure
    plt.figure(figsize=(10, 8))
    sns.heatmap(node_cm_ratio, annot=True, fmt='.2f',
                xticklabels=cls_label_names,
                yticklabels=cls_label_names,
                cmap='Blues')
    plt.title(f'Normalized Confusion Matrix - Epoch {epoch}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Save with consistent filename (overwrites previous)
    save_path = os.path.join(save_dir, 'latest_confusion_matrix.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()  # Close the figure to free memory

    # Optional: Also save with epoch number for historical tracking
    historical_path = os.path.join(save_dir, f'confusion_matrix_epoch_{epoch}.png')
    plt.savefig(historical_path, bbox_inches='tight', dpi=300)
    plt.close()

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
import numpy as np


