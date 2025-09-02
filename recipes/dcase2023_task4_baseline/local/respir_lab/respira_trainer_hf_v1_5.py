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

        # Initialize the loss function (you can experiment with alpha and gamma)
        self.interval_conf_loss_fn = FocalLossBinary(alpha=0.25, gamma=2.0, reduction='mean')

        # Compute interval class alpha (class weights)
        #  [Crackle, Wheeze, Rhonchi, Stridor]
        # alpha = [0.023, 0.035, 0.256, 0.686] for [Crackle, Wheeze, Rhonchi, Stridor]
        # counts = torch.tensor([1289, 871, 119, 44], dtype=torch.float32)
        counts = torch.tensor([13794, 7027, 3780, 657], dtype=torch.float32)
        interval_alpha = 1.0 / counts  # Inverse of class counts,
        self.interval_cls_alpha =  interval_alpha / interval_alpha.sum()  # Normalize to sum to 1
        self.interval_cls_loss_fn = FocalLoss(alpha=self.interval_cls_alpha, gamma=2.0, reduction='mean')



        #self.vad_loss_fn = nn.BCELoss()  # For node VAD loss,    主导损失， 先完成正常异常二分类
        # Compute pos_weight
        num_neg = 88613
        num_pos = 29682
        node_vad_alpha = [0.20, 0.80]  # Experiment with different weights
        self.node_vad_alpha = torch.tensor(node_vad_alpha, dtype=torch.float32)
        # Initialize Binary Focal Loss for VAD
        self.vad_loss_fn = FocalLoss(alpha= self.node_vad_alpha, gamma=2.0, reduction='mean')


        self.frames_per_node = 5

        # Loss weight parameters
        # self.node_loss_weight_initial = 1.0
        # self.node_loss_weight_initial = 10.0
        # self.node_loss_weight = self.node_loss_weight_initial    #  特征属性

        self.interval_conf_loss_weight =  40.0
        self.interval_cls_loss_weight =   40.0
        self.interval_location_loss_weight = 1.0
        # self.interval_distill_loss_weight = 10.00


        #  创建一个异常的缓冲， 用于专门存储异常的 预测节点；
        self.val_buffer_intervals= {  # 为不同阈值创建多个 DataFrame 缓冲区，用于存储合成数据和测试数据的验证结果。
            conf_th: pd.DataFrame() for conf_th in self.hparams["training"]["val_conf_thresholds"]

        }




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

        # Initialize storage for node label counts per epoch
        self.epoch_node_label_counts = torch.zeros(5, dtype=torch.long)
        self.valid_epoch_node_label_counts = torch.zeros(5, dtype=torch.long)


        self.validation_node_preds = []
        self.validation_node_targets = []



        # Initialize lists to store metrics for each epoch
        self.training_step_outputs = {
            # 'node_level_loss': [],


            "interval_conf_loss": [],
            "interval_cls_loss":[],
            "interval_loc_loss":[],

           # "interval_distill_loss": [],
            'total_loss':[]
        }


        self.valid_step_outputs = {
            #'node_level_loss': [],

            "interval_conf_loss": [],
            "interval_cls_loss":[],
            "interval_loc_loss":[],

            #"interval_distill_loss": [],
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

    def batch_inference_interval(self,
            model_outputs: dict,
            conf_threshold: float = 0.5,
            nms_iou_threshold: float = 0.8
    ):
        """
        model_outputs: {
            "pred_intervals": Tensor,           # [batch_size, total_intervals, 2]
            "pred_intervals_conf_logits": Tensor,   # [batch_size, total_intervals, 1]
            "pred_intervals_cls_logits": Tensor,    # [batch_size, total_intervals, num_classes - 1]
            ...
        }
        Returns: A list of length batch_size, each element is a dict with final intervals, confidences, sub-type IDs.
        """
        final_bounds = model_outputs["pred_intervals"]  # [batch_size, total_intervals, 2]
        interval_conf_logits = model_outputs["pred_intervals_conf_logits"]  # [batch_size, total_intervals, 1]
        interval_cls_logits = model_outputs["pred_intervals_cls_logits"]  # [batch_size, total_intervals, num_classes - 1]

        batch_size = final_bounds.size(0)
        batch_results = []

        for b_idx in range(batch_size):
            # Per-sample predictions (index into batch dimension)
            bounds_b = final_bounds[b_idx]          # [total_intervals, 2]
            conf_b = interval_conf_logits[b_idx]    # [total_intervals, 1]
            cls_b = interval_cls_logits[b_idx]      # [total_intervals, num_classes - 1]

            # Run single-sample inference
            final_b, final_conf_b, final_subtypes_b = self.inference_one_sample(
                bounds_b, conf_b, cls_b,
                conf_threshold=conf_threshold,
                nms_iou_threshold=nms_iou_threshold
            )

            # Store results in a dict
            sample_result = {
                "pred_intervals": final_b,          # [K, 2]
                "pred_interval_confident": final_conf_b,  # [K]
                "pred_interval_cls": final_subtypes_b     # [K]
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
        audios_dur = [duration.clone().detach().to(device) for duration in batch_data['audio_dur']]
        c_ex_mixtures = batch_data['c_ex_mixtures']  # List of audio names (metadata)

        chest_pos = [chest_pos.clone().detach().to(device) for chest_pos in batch_data['chest_info']]
        #gender_info = [gender.clone().detach().to(device) for gender in batch_data['gender_info']]

        # vad_timestamps = [vad_time.clone().detach().to(device) for vad_time in batch_data['vad_timestamps']]
        vad_timestamps =  batch_data['vad_timestamps']

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

        node_predictions = outputs_detection['node_predictions']  # Shape: (total_nodes, num_classes)
        node_labels = outputs_detection['node_labels']  # Shape: (total_nodes,)

        pred_intervals = outputs_detection["pred_intervals"]  # Shape: [batch_size, total_intervals, 2]
        distill_loss = outputs_detection['distill_loss']  # Scalar or None
        interval_conf_logits = outputs_detection[
            "pred_intervals_conf_logits"]  # Shape: [batch_size, total_intervals, 1]
        interval_cls_logits = outputs_detection[
            "pred_intervals_cls_logits"]  # Shape: [batch_size, total_intervals, num_classes - 1]

        batch_size = pred_intervals.size(0)
        total_intervals_per_sample = pred_intervals.size(1)
        # e.g., 39 + 19 + 9 = 67 if num_intervals_per_scale=[39, 19, 9]
        # We'll collect labels for the entire batch to compute the final loss


        # 4. Now you have conf_targets, cls_targets, box_targets for each sample
        #    => you can flatten them or compute losses sample by sample

        # Extract targets directly from batch
        # Extract target lists directly from batch_data
        conf_targets_list = batch_data['conf_targets']
        cls_targets_list = batch_data['cls_targets']
        box_targets_list = batch_data['box_targets']

        # Move tensors to the correct device
        conf_targets_list = [t.to(self.device) for t in conf_targets_list]
        cls_targets_list = [t.to(self.device) for t in cls_targets_list]
        box_targets_list = [t.to(self.device) for t in box_targets_list]


        # Concatenate targets across the batch
        conf_targets_cat = torch.cat(conf_targets_list, dim=0)  # Shape: [total_targets, 1] or [total_targets]
        cls_targets_cat = torch.cat(cls_targets_list, dim=0)  # Shape: [total_targets]
        box_targets_cat = torch.cat(box_targets_list, dim=0)  # Shape: [total_targets, 2]

        # Reshape predictions to match concatenated targets
        # Flatten batch and interval dimensions
        conf_preds_cat = interval_conf_logits.view(-1, 1)  # Shape: [batch_size * total_intervals, 1]
        cls_preds_cat = interval_cls_logits.view(-1, interval_cls_logits.size(
            -1))  # Shape: [batch_size * total_intervals, num_classes - 1]
        all_pred_bounds = pred_intervals.view(-1, 2)  # Shape: [batch_size * total_intervals, 2]

        # Ensure the number of predictions matches the number of targets
        # If targets and predictions have different lengths, you might need alignment logic (e.g., padding or matching)
        assert conf_preds_cat.size(0) == conf_targets_cat.size(0), \
            f"Mismatch: {conf_preds_cat.size(0)} predictions vs {conf_targets_cat.size(0)} targets"

        # 5. Confidence Loss
        # Note: If interval_conf_loss_fn expects logits, no sigmoid is needed here; apply it inside the loss function if required
        conf_loss = self.interval_conf_loss_fn(conf_preds_cat, conf_targets_cat)
        # 6. Sub-type classification => only for intervals where conf=1
        # Foreground mask
        fg_mask = (conf_targets_cat == 1)

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


        # Count occurrences of each label type in the batch
        batch_label_counts = torch.bincount(node_labels, minlength=5)
        self.epoch_node_label_counts += batch_label_counts.to(self.epoch_node_label_counts.device)


        # # Identify abnormal and normal nodes
        # # note, 训练阶段是提取真实值是异常类型的节点，
        # abnormal_node_indices = (node_labels != 0).nonzero(as_tuple=True)[0]
        # normal_node_indices = (node_labels == 0).nonzero(as_tuple=True)[0]
        #
        # # Determine number of normal nodes to sample (e.g., equal to the number of abnormal nodes)
        # num_abnormal_nodes = len(abnormal_node_indices)
        # num_normal_nodes_to_sample = max(64, num_abnormal_nodes // 3)  # Adjust denominator based on desired ratio
        #
        # # Sample normal nodes
        # if len(normal_node_indices) > num_normal_nodes_to_sample:
        #     sampled_normal_indices = normal_node_indices[
        #         torch.randperm(len(normal_node_indices))[:num_normal_nodes_to_sample]]
        # else:
        #     sampled_normal_indices = torch.tensor([], dtype=torch.long, device=device)
        #
        # # Combine indices
        # node_indices_to_use = torch.cat((abnormal_node_indices, sampled_normal_indices))
        #
        # # Ensure indices are unique
        # node_indices_to_use = node_indices_to_use.unique()
        #
        # # Extract predictions and labels for selected nodes
        # node_predictions_selected = node_predictions[node_indices_to_use]
        # node_labels_selected = node_labels[node_indices_to_use]
        #
        # # Compute node-level loss
        # node_loss = self.cls_node_loss_fn(node_predictions_selected, node_labels_selected)
        # node_level_loss = ( self.node_loss_weight *  node_loss  )



        # # Accumulate node-level predictions and labels for confusion matrix
        # self.training_node_preds.append(torch.argmax(node_predictions_selected, dim=1).cpu())
        # self.training_node_targets.append(node_labels_selected.cpu())


        conf_loss            = self.interval_conf_loss_weight * conf_loss
        interval_cls_loss    = self.interval_cls_loss_weight  * interval_cls_loss
        loc_loss             = self.interval_location_loss_weight * loc_loss
        distill_loss         = self.interval_distill_loss_weight * distill_loss

        # Total loss
        total_loss = (
                      conf_loss  + interval_cls_loss +
                      loc_loss
                      + distill_loss
                      )



        # 1. 节点分类损失，
        # 4. 置信度损失， 5. 区间分类损失， 6. 区间定位损失；
        # 7. 区间多样性损失， 8. 定位特征蒸馏损失
        # 这八个损失， 后续需要通过实验， 来验证能否去除；

        # Log individual losses and total loss
        # Log losses
        # self.log("train/node_level_loss", node_level_loss, prog_bar=True)

        self.log("train/interval_conf_loss", conf_loss,prog_bar=True)
        self.log("train/interval_cls_loss", interval_cls_loss,prog_bar=True)
        self.log("train/interval_loc_loss", loc_loss,prog_bar=True)
        #self.log("train/interval_distill_loss", distill_loss,prog_bar=True)

        self.log("train/total_loss", total_loss, prog_bar=True)




        # Store values for epoch-end calculation
        #self.training_step_outputs['node_level_loss'].append(node_level_loss)

        self.training_step_outputs['interval_conf_loss'].append(conf_loss)
        self.training_step_outputs['interval_cls_loss'].append(interval_cls_loss)
        self.training_step_outputs['interval_loc_loss'].append(loc_loss)
        #self.training_step_outputs['interval_distill_loss'].append(distill_loss)

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


            # --- New Adjustment Logic ---
            # Check if node-level F1-score improved
            if node_f1_score > self.best_node_f1:
                self.best_node_f1 = node_f1_score

            # Start adjusting if node-level F1-score reaches 0.80
            if node_f1_score >= 0.98 and not self.adjustment_started:
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
        # Calculate epoch statistics
        epoch_metrics = {
            key: torch.stack(values).mean().item()
            for key, values in self.training_step_outputs.items()
        }

        # Log epoch-level metrics
        #self.log("train/node_loss/epoch",         epoch_metrics['node_level_loss'], on_epoch=True)

        self.log("train/interval_conf_loss/epoch", epoch_metrics['interval_conf_loss'], on_epoch=True)
        self.log("train/interval_cls_loss/epoch",  epoch_metrics['interval_cls_loss'], on_epoch=True)
        self.log("train/interval_loc_loss/epoch",  epoch_metrics['interval_loc_loss'], on_epoch=True)
        # self.log("train/interval_distill_loss/epoch", epoch_metrics['interval_distill_loss'], on_epoch=True)

        self.log("train/total_loss/epoch", epoch_metrics['total_loss'], on_epoch=True)





        # Clear the lists for next epoch
        self.training_node_preds = []
        self.training_node_targets = []

        self.training_node_vad_preds = []
        self.training_node_vad_targets = []

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
        # gender_info = [gender.clone().detach().to(device) for gender in batch_data['gender_info']]
        # vad_timestamps = [vad_time.clone().detach().to(device) for vad_time in batch_data['vad_timestamps']]
        vad_timestamps =  batch_data['vad_timestamps']
        audios_dur = [duration.clone().detach().to(device) for duration in batch_data['audio_dur']]

        anchor_intervals = batch_data["anchor_intervals"]
        assignments = batch_data["assignments"]

        # Prepare batch_data dictionary for the model
        batch_data_device = {
            'spectrograms': spectrograms,
            'frame_labels': frame_labels,

            'c_ex_mixtures':batch_audio_names,
            'vad_timestamps': vad_timestamps,

            'chest_loc': chest_pos,
            # 'genders': gender_info,

            'audio_dur': audios_dur,
            "anchor_intervals": anchor_intervals,
            "assignments": assignments

        }


        ''' ====================== Node Level Stage ===================== '''
        # Perform detection at node level
        outputs_detection = self.sed_net(batch_data_device)

        batch_detection_names = outputs_detection['batch_audio_names']  # List of audio names

        # note, batch_audio_names  需要更新成 batch detection names， 表示只获取预测为异常样本的音频文件；
        filenames_synth = [ x  for x in  batch_detection_names
                        if Path(x).parent == Path(self.hparams["data"]["eval_folder_4k"])
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


        # Prepare a dictionary to hold the decoded predictions for each threshold.
        decoded_abnormal_pred = {}

        # Loop through each confidence threshold.
        for idx, conf_th in enumerate(self.hparams["training"]["val_conf_thresholds"]):
            # If you want to use a corresponding IoU threshold, for example by index:
            iou_th = self.hparams["training"]["val_iou_NMS"][idx]

            # Run inference for the current confidence threshold and IoU threshold.
            batch_results = self.batch_inference_interval(outputs_detection, conf_threshold=conf_th,
                                                          nms_iou_threshold=iou_th)

            # Process batch_results to decode abnormal predictions.
            # You might have a function that converts batch_results into a DataFrame, for example:
            decoded_df = self.decode_results(batch_results, batch_audio_names, interval_4labels)

            # Save the decoded predictions for this threshold.
            decoded_abnormal_pred[conf_th] = decoded_df

            # Optionally, print or log results for debugging.
            # print(f"Results for conf_threshold={conf_th}, iou_threshold={iou_th}:")
            # print(decoded_df.head())

        # Finally, update the buffer by concatenating the new results with previous ones (if any).
        for conf_th in self.val_buffer_intervals.keys():
            self.val_buffer_intervals[conf_th] = pd.concat(
                [self.val_buffer_intervals[conf_th], decoded_abnormal_pred[conf_th]],
                ignore_index=True,
            )

        node_predictions = outputs_detection['node_predictions']  # Shape: (total_nodes, num_classes)
        node_labels = outputs_detection['node_labels']  # Shape: (total_nodes,)


        # Count occurrences of each label type in the batch
        batch_label_counts = torch.bincount(node_labels, minlength=5)
        self.valid_epoch_node_label_counts += batch_label_counts.to(self.valid_epoch_node_label_counts.device)



        pred_intervals = outputs_detection["pred_intervals"]
        # Extract targets directly from batch
        # Extract target lists directly from batch_data
        conf_targets_list = batch_data['conf_targets']
        cls_targets_list = batch_data['cls_targets']
        box_targets_list = batch_data['box_targets']

        # Move tensors to the correct device
        conf_targets_list = [t.to(self.device) for t in conf_targets_list]
        cls_targets_list = [t.to(self.device) for t in cls_targets_list]
        box_targets_list = [t.to(self.device) for t in box_targets_list]

        # Concatenate across the batch
        conf_targets_cat = torch.cat(conf_targets_list, dim=0)
        cls_targets_cat = torch.cat(cls_targets_list, dim=0)
        box_targets_cat = torch.cat(box_targets_list, dim=0)

        pred_intervals = outputs_detection["pred_intervals"]  # Shape: [batch_size, total_intervals, 2]
        distill_loss = outputs_detection['distill_loss']  # Scalar or None
        interval_conf_logits = outputs_detection[
            "pred_intervals_conf_logits"]  # Shape: [batch_size, total_intervals, 1]
        interval_cls_logits = outputs_detection[
            "pred_intervals_cls_logits"]  # Shape: [batch_size, total_intervals, num_classes - 1]

        # Reshape predictions to match concatenated targets
        # Flatten batch and interval dimensions
        conf_preds_cat = interval_conf_logits.view(-1, 1)  # Shape: [batch_size * total_intervals, 1]
        cls_preds_cat = interval_cls_logits.view(-1, interval_cls_logits.size(
            -1))  # Shape: [batch_size * total_intervals, num_classes - 1]
        all_pred_bounds = pred_intervals.view(-1, 2)  # Shape: [batch_size * total_intervals, 2]

        # Ensure the number of predictions matches the number of targets
        # If targets and predictions have different lengths, you might need alignment logic (e.g., padding or matching)
        assert conf_preds_cat.size(0) == conf_targets_cat.size(0), \
            f"Mismatch: {conf_preds_cat.size(0)} predictions vs {conf_targets_cat.size(0)} targets"

        # 5. Confidence Loss => binary cross-entropy with logits, note,   这里在使用 conf pred logit 是否需要使用 sigmoid ，因为后续使用了；
        #conf_loss = F.binary_cross_entropy_with_logits(conf_preds_cat, conf_targets_cat)
        conf_loss =  self.interval_conf_loss_fn(conf_preds_cat, conf_targets_cat)

        # 6. Sub-type classification => only for intervals where conf=1
        # Foreground mask
        fg_mask = (conf_targets_cat == 1)

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

        distill_loss = outputs_detection['distill_loss']

        conf_loss            = self.interval_conf_loss_weight * conf_loss
        interval_cls_loss    = self.interval_cls_loss_weight  * interval_cls_loss
        loc_loss             = self.interval_location_loss_weight * loc_loss
        distill_loss         = self.interval_distill_loss_weight * distill_loss


        self.valid_total_loss =   (
                                   conf_loss + interval_cls_loss +
                                   loc_loss
                                   + distill_loss

                                   )


        # Log individual losses and total loss
        #self.log("valid/node_loss", self.valid_node_loss, prog_bar=True)
        self.log("valid/interval_conf_loss", conf_loss,prog_bar=True)
        self.log("valid/interval_cls_loss", interval_cls_loss,prog_bar=True)
        self.log("valid/interval_loc_loss", loc_loss,prog_bar=True)
        #self.log("valid/interval_distill_loss", distill_loss,prog_bar=True)

        self.log("valid/total_loss", self.valid_total_loss, prog_bar=True)



        #self.valid_step_outputs['node_level_loss'].append(self.valid_node_loss)

        self.valid_step_outputs['interval_conf_loss'].append(conf_loss)
        self.valid_step_outputs['interval_cls_loss'].append(interval_cls_loss)
        self.valid_step_outputs['interval_loc_loss'].append(loc_loss)
        #self.valid_step_outputs['interval_distill_loss'].append(distill_loss)

        self.valid_step_outputs['total_loss'].append( self.valid_total_loss)


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





        #  读取valid 数据集的真实标签和音频时长。
        # 更新后的函数可以用来建立空列表的事件， 由于空列表代表的是正常类型的record ,后续会将其过滤掉；

        DEFAULT_EVENT_SEGMENT_SCORED = (0.0,)  # Adjust based on expected structure
        # Compute node-level event-based metrics
        try:
            cur_thread = self.hparams["training"]["val_conf_thresholds"][0]
            cur_iou_NMS = self.hparams["training"]["val_iou_NMS"][0]
            print(f"\n Using confidient & NMS thread-based to filter intervals, cur_conf_thread:{cur_thread}, iou_NMS:{cur_iou_NMS}")
            node_event_segment_scored = compute_event_based_metrics(
                self.val_buffer_intervals[cur_thread],
                self.hparams["data"]["valid_tsv"],
                save_dir=os.path.join(self.hparams["log_dir"], "pred_interval_event_metrics"),
            )


            cur_thread1 = self.hparams["training"]["val_conf_thresholds"][1]
            cur_iou_NMS1 = self.hparams["training"]["val_iou_NMS"][1]
            print(f"\n Using confidient & NMS thread-based to filter intervals, cur_conf_thread:{cur_thread1},iou_NMS:{cur_iou_NMS1}")
            node_event_segment_scored1 = compute_event_based_metrics(
                self.val_buffer_intervals[cur_thread1],
                self.hparams["data"]["valid_tsv"],
                save_dir=os.path.join(self.hparams["log_dir"], "conf1_pred_interval_event_metrics"),
            )


            cur_thread2 = self.hparams["training"]["val_conf_thresholds"][2]
            cur_iou_NMS2 = self.hparams["training"]["val_iou_NMS"][2]
            print(f"\n Using confidient & NMS thread-based to filter intervals, cur_conf_thread:{cur_thread2},iou_NMS:{cur_iou_NMS2}")
            node_event_segment_scored2 = compute_event_based_metrics(
                self.val_buffer_intervals[cur_thread2],
                self.hparams["data"]["valid_tsv"],
                save_dir=os.path.join(self.hparams["log_dir"], "conf2_pred_interval_event_metrics"),
            )


        except Exception as e:
            print(f"Error in compute_event_based_metrics for node level: {e}")
            node_event_segment_scored = DEFAULT_EVENT_SEGMENT_SCORED

        node_event_thread_f_score = node_event_segment_scored1[0]

        # Update the objective metric
        obj_metric = torch.tensor( node_event_thread_f_score)

        # Update the best model if validation score improves
        val_score =  node_event_thread_f_score

        if val_score > self.best_val_score:
            self.best_val_score = val_score
            self.best_model_weights = deepcopy(self.sed_net.state_dict())
            self.best_ema_weights = deepcopy(self.ema_model.state_dict())

        self.log('best_val_class_wise_score', self.best_val_score, prog_bar=True)
        self.log_counts(self.valid_epoch_node_label_counts, "node_label", "node label")

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
        # self.log("valid/node_loss/epoch", epoch_metrics['node_level_loss'], on_epoch=True)

        self.log("valid/interval_conf_loss/epoch", epoch_metrics['interval_conf_loss'], on_epoch=True)
        self.log("valid/interval_cls_loss/epoch",  epoch_metrics['interval_cls_loss'], on_epoch=True)
        self.log("valid/interval_loc_loss/epoch",  epoch_metrics['interval_loc_loss'], on_epoch=True)
        self.log("valid/interval_loc_loss/epoch", epoch_metrics['interval_distill_loss'], on_epoch=True)
       # self.log("valid/total_loss/epoch", epoch_metrics['total_loss'], on_epoch=True)

        
        # Clear validation buffers for the next epoch
        self.validation_node_preds = []
        self.validation_node_targets = []
        self.validation_edge_preds = []
        self.validation_edge_targets = []
        self.validation_node_vad_preds = []
        self.validation_node_vad_targets = []

        # Clear buffers for node level

        self.val_buffer_intervals = {th: pd.DataFrame() for th in self.val_buffer_intervals.keys()}
        #self.val_buffer_refine_pred = pd.DataFrame()

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

        node_learing_params = (
            list(self.sed_net.node_fea_generator.parameters()) +
            list(self.sed_net.node_vad_proj.parameters()) +
            list(self.sed_net.edge_encoder.parameters()) +
            list(self.sed_net.node_interval_pred.parameters())
            # list(self.sed_net.factor_graph_layer.parameters())

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
            #  "gender_info": [item['gender_info'] for item in batch],
            'vad_timestamps': [item['vad_timestamps'] for item in batch],
            'audio_dur': [item['audio_duration'] for item in batch],

            'anchor_intervals': [item['anchor_intervals'] for item in batch],
            'assignments': [item['assignments'] for item in batch],

            'conf_targets': [item['conf_targets'] for item in batch],
            'cls_targets':[item['cls_targets'] for item in batch],
            'box_targets':[item['box_targets'] for item in batch],
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
class FocalLossBinary(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (float): Weighting factor for the positive class.
            gamma (float): Focusing parameter that adjusts the rate at which easy examples are down-weighted.
            reduction (str): 'mean', 'sum', or 'none' for the loss reduction method.
        """
        super(FocalLossBinary, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (Tensor): Raw predictions (logits) of shape (N,) or (N, 1).
            targets (Tensor): Ground truth binary labels of shape (N,).
        Returns:
            Tensor: Focal loss value.
        """
        # Ensure inputs are 1D (if necessary)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Compute binary cross entropy loss without reduction
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Convert logits to probabilities
        probas = torch.sigmoid(inputs)
        # For each instance, p_t is the model's estimated probability for the true class
        p_t = targets * probas + (1 - targets) * (1 - probas)

        # Compute the focal factor
        focal_factor = (1 - p_t) ** self.gamma

        # Apply alpha weighting: alpha for positive, (1-alpha) for negative examples
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        # Combine the factors with the BCE loss
        loss = alpha_factor * focal_factor * BCE_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

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
