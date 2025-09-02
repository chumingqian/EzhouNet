import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax, remove_self_loops, add_self_loops
from torch_geometric.typing import Adj, OptTensor, Size, OptPairTensor
from torch import Tensor
from typing import Union, Tuple, Optional
from torch_geometric.nn.conv import MessagePassing
# from torch_sparse import SparseTensor
# import torch_sparse
from desed_task.nnet.DCNN_v8_16 import  DynamicFeatureExtractor

# Adjusted GATConv that supports edge attributes (your implementation)
import numpy as np



#  V7-5-7,   使用可学习的缩放形式， 将两者进行拼接；
#  v8-1,  开始引入 d fine 的思想， 对定位的时间损失进行计算。
#  v8-7,  先完成节点分类， 区间分类，区间定位的损失， 蒸馏损失先不考虑进去；
#  v8-7,  直接使用 anchor + offset logit;
#  v8-8,  直接使用 anchor + delta + offset logit; ( not sure work)

#  v8-9,  直接使用 anchor + offset logit;   并且此时， 去除所有其他非区间损失；
# 将节点类型相关的所有损失去除， 只保留区间类型的相关损失；


#  v8-10, 考虑节点损失进去， 但节点边的损失，以及 node vad 相关的损失不考虑进去；
#   之后使用 anchor + offset logit;


#  v8-10_2,  此时的生成预定义的 anchor interval,
#  总共三种尺度， 每种尺度包含不同的 anchor interval 的个数；
#  与之前的不同的是 每种尺度都会从0遍历到1；


# v8-11， anchor interval 的生成移动到 dataset 中去了，
# 此时网络只需要根据预先定义好的 anchor interval 学习其对应的 offset,
#  从而生成最终预测的interval,  然后计算区间的分类与定位损失



# v8-13, 每种尺度使用各自对应的实例化对象进行学习；
# 并且直接只使用一层 refine layer 学习对应的offset，
# 直接作用于 anchor interval 进行学习； set  num_refine_layers = 3,


# v8-13_2, 每种尺度使用各自对应的实例化对象进行学习；
# 并且直接只使用一层 refine layer 学习对应的offset，
# 直接作用于 anchor interval 进行学习； set  num_refine_layers = 1;


# v8-14, 使用3层 refine_layer，
# 每一层    在前一层生成的 start, end 基础上，进行offset 学习；
# 并且此时， 将每一层输出的 start_logit, end_logit 进行保留， 使用最后一层的进行知识蒸馏；

# v8-14_2   使用软置信度的方式； 来设置 anchor interval 对应的真实置信度标签；

# # v8-14_3   使用用 hard confidence ,硬置信度，  但是节点分类不参与计算；

# # v8-16   基于每个 anchor  interval 的间隔生成每个节点特征， 即 anchor node 的方式；
#  去除固定帧数生成节点的方式 ， 此时使用anchor inerval 区间中帧数生成节点，
#  即此时的每个节点特征都是由不等长度的帧数生成的， 且该长度是由该anchor 对应区间中的帧数来决定的；



import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
class Interval_Refine(nn.Module):
    def __init__(self, node_embedding_dim, num_classes=5, num_refine_layers=3,
                 dist_bins_list=[80, 60, 40], kernel_size=5, max_intervals= 70 ):
        #super().__init__()
        super(Interval_Refine, self).__init__()
        self.node_embedding_dim = node_embedding_dim
        self.num_classes = num_classes
        self.kernel_size = kernel_size

        self.num_refine_layers = num_refine_layers
        self.dist_bins = dist_bins_list
        self.num_scales = 3  # Three scales

        self.normal_class_idx = 0
        self.true_duration_mode = False

        # Duration-aware time embedding.
        # Input: [normalized_time, duration_ratio]
        self.time_embed = nn.Sequential(
            nn.Linear(2, node_embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(node_embedding_dim)
        )

        # Learnable Gaussian smoothing.
        # Grouped conv1d is applied to the node prediction scores.
        self.smoothing_conv = nn.Conv1d(
            in_channels=num_classes,
            out_channels=num_classes,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=num_classes,
            bias=False
        )
        self._init_gaussian_weights()



        # Scale-specific refinement layers;
        # Refinement layers for boundary adjustment.
        # Input: [local_feature, center, width, start, end, local_abnormal_feature]
        # (dimension = node_embedding_dim + 5)
        self.refinement_layers = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(node_embedding_dim + 4, 256),
                    nn.ReLU(),
                    nn.Linear(256, 2 * dist_bins_list[i])
                ) for _ in range(num_refine_layers)
            ]) for i in range(self.num_scales)
        ])

        # Scale-specific local feature GRUs;
        # --- Learnable Local Aggregators ---
        # For each candidate interval, we use GRUs to aggregate node features and abnormal scores.
        # self.local_feat_rnns = nn.ModuleList([
        #     nn.GRU(input_size=node_embedding_dim, hidden_size=node_embedding_dim, batch_first=True)
        #     for _ in range(self.num_scales)
        # ])
        #
        # # Scale-specific local abnormal GRUs
        # self.local_abnormal_rnns = nn.ModuleList([
        #     nn.GRU(input_size=1, hidden_size=1, batch_first=True)
        #     for _ in range(self.num_scales)
        # ])


        self.weight_params = nn.ParameterList([
            nn.Parameter(torch.linspace(-0.5, 0.5, dist_bins_list[i]))
            for i in range(self.num_scales)
        ])

        # Scale-specific confidence heads;
        # NEW: Confidence head for normal (0) vs. abnormal (1)
        # We'll feed the same local feature vector into this head
        self.interval_conf_heads = nn.ModuleList([
            nn.Linear(node_embedding_dim , 1) for _ in range(self.num_scales)
        ])

        # Scale-specific classification heads;
        # We'll feed it [local_feat, local_abnormal_feature], shape => (node_embedding_dim + 1)
        #  not include normal class;
        self.interval_class_heads = nn.ModuleList([
            nn.Linear(node_embedding_dim , self.num_classes - 1) for _ in range(self.num_scales)
        ])



    def _init_gaussian_weights(self):
        """Initialize smoothing convolution with a fixed Gaussian kernel."""
        kernel = self._get_gaussian_kernel()
        with torch.no_grad():
            # For grouped conv, assign the same kernel to each channel.
            for i in range(self.smoothing_conv.weight.shape[0]):
                self.smoothing_conv.weight[i, 0, :] = kernel

    def _get_gaussian_kernel(self):
        """Returns a fixed 1D Gaussian kernel, e.g. [0.06136, 0.24477, 0.38774, 0.24477, 0.06136]."""
        kernel = torch.tensor([0.06136, 0.24477, 0.38774, 0.24477, 0.06136], dtype=torch.float32)
        return kernel

    def _empty_output(self, device):
        return {
            "final_bounds": torch.zeros((0, 2), device=device),
            "interval_mask": torch.zeros(0, dtype=torch.bool, device=device),
            "num_intervals": torch.tensor(0, device=device)
        }

    def forward(
            self,   # one sample node information;  not batch node sample
            node_embeddings: Tensor,  # shape (num_nodes, node_embedding_dim)
            time_positions: Tensor,  # shape (num_nodes, 1) normalized [0,1]
            node_pred: Tensor,  # shape (num_nodes, num_node_classes)
            audio_len: Tensor,  # the duaration of  this audio
            distill_loss: bool= True ,
            duration_ratio: float = None,
            cur_anchor_intervals: dict ={},
            num_intervals_per_scale=None

    ):
        """
        Returns a dict with:
          - "final_bounds": (max_intervals, 2) refined [start, end] in [0,1].
          - "interval_cls_logits": (max_intervals, num_interval_classes) classification logits.
          - "num_intervals": number of intervals (always max_intervals here).
        """
        device = node_embeddings.device


        if num_intervals_per_scale is None:
            raise ValueError("num_intervals_per_scale must be provided as a list of integers [n1, n2, n3]")

        # print(type(cur_anchor_intervals))  # Should print <class 'dict'>
        # print(cur_anchor_intervals)  # See what’s inside the dictionary
        # Split node_embeddings and anchor_intervals by scale
        node_embeddings_list = torch.split(node_embeddings, num_intervals_per_scale, dim=0)
        anchor_intervals_list = torch.split(cur_anchor_intervals, num_intervals_per_scale, dim=0)


        # Lists to collect results from all scales
        final_bounds_list = []
        interval_conf_logits_list = []
        interval_cls_logits_list = []

        distill_loss_value = 0
        # Process each scale independently
        for scale_idx in range(self.num_scales):
            local_feat = node_embeddings_list[scale_idx]  # Node embeddings for this scale
            anchor_intervals = anchor_intervals_list[scale_idx]
            num_intervals_scale = num_intervals_per_scale[scale_idx]

            if num_intervals_scale == 0:
                continue

            # Select scale-specific components

            refinement_layers = self.refinement_layers[scale_idx]
            weight_params = self.weight_params[scale_idx]
            interval_conf_head = self.interval_conf_heads[scale_idx]
            interval_class_head = self.interval_class_heads[scale_idx]

            # 1. Duration-aware time embedding (shared across scales)
            # 3. Compute masks for this scale's intervals
            time_pos_1d = time_positions.squeeze(-1) * audio_len.item()

            starts = anchor_intervals[:, 0].clone()
            ends = anchor_intervals[:, 1].clone()

            # 6. Refine intervals for this scale
            centers = (starts + ends) / 2
            widths = ends - starts

            candidate_layer_outputs = []
            #  here each update  are all based on original start and end;
            for layer in refinement_layers:
                query = torch.cat([
                    local_feat,
                    centers.unsqueeze(-1) / audio_len.item(),
                    widths.unsqueeze(-1) / audio_len.item(),
                    starts.unsqueeze(-1) / audio_len.item(),
                    ends.unsqueeze(-1) / audio_len.item(),
                ], dim=1)


                logits = layer(query)

                # here keep each layers's logit used for distillation  for the laber;
                start_logits, end_logits = logits.chunk(2, dim=1)
                candidate_layer_outputs.append((start_logits, end_logits))

                start_offsets = (F.softmax(start_logits, dim=1) * weight_params).sum(dim=1)
                end_offsets = (F.softmax(end_logits, dim=1) * weight_params).sum(dim=1)

                # starts = (starts + start_offsets).clamp(0, 1)
                # ends = (ends + end_offsets).clamp(0, 1)

                starts = (starts + start_offsets)
                ends = (ends + end_offsets)

            # 7. Final bounds and predictions for this scale
            final_bounds_scale = torch.stack([starts, ends], dim=1)

            interval_conf_logits_scale = interval_conf_head(local_feat)
            interval_cls_logits_scale = interval_class_head(local_feat)

            # Collect results for this scale
            final_bounds_list.append(final_bounds_scale)
            interval_conf_logits_list.append(interval_conf_logits_scale)
            interval_cls_logits_list.append(interval_cls_logits_scale)

            # Distillation
            teacher_start, teacher_end = candidate_layer_outputs[-1]
            for student_start, student_end in candidate_layer_outputs[:-1]:
                loss_start = F.kl_div(
                    F.log_softmax(student_start, dim=-1),
                    F.softmax(teacher_start, dim=-1),
                    reduction='batchmean'
                )
                loss_end = F.kl_div(
                    F.log_softmax(student_end, dim=-1),
                    F.softmax(teacher_end, dim=-1),
                    reduction='batchmean'
                )
                distill_loss_value += (loss_start + loss_end)  # 三种尺度下， 单个样本蒸馏损失的总和；


        # 8. Concatenate results from all scales
        final_bounds = torch.cat(final_bounds_list, dim=0)
        interval_conf_logits = torch.cat(interval_conf_logits_list, dim=0)
        interval_cls_logits = torch.cat(interval_cls_logits_list, dim=0)


        return {
            "final_bounds": final_bounds,  # (max_intervals, 2)
            "distill_loss": distill_loss_value if distill_loss else None,

            "interval_conf_logits": interval_conf_logits,  # (max_intervals, 1) NEW
            "interval_cls_logits": interval_cls_logits,  # (max_intervals, num_interval_classes)
        }


def adjust_vad_timestamps(vad_intervals, frame_times):
    adjusted_vad_intervals = []
    for interval in vad_intervals:
        start = float (interval['start'])
        end = float(interval['end'])
        # Find the closest frame time to start
        start_idx = np.searchsorted(frame_times, start, side='left')
        if start_idx >= len(frame_times):
            start_idx = len(frame_times) - 1
        elif start_idx > 0 and (frame_times[start_idx] - start) > (start - frame_times[start_idx - 1]):
            start_idx -= 1
        start_time = frame_times[start_idx]

        # Find the closest frame time to end
        end_idx = np.searchsorted(frame_times, end, side='left')
        if end_idx >= len(frame_times):
            end_idx = len(frame_times) - 1
        elif end_idx > 0 and (frame_times[end_idx] - end) > (end - frame_times[end_idx - 1]):
            end_idx -= 1
        end_time = frame_times[end_idx]

        adjusted_vad_intervals.append({'start': start_time, 'end': end_time})
    return adjusted_vad_intervals



# Function to check if a chunk is within any VAD interval
def is_chunk_within_vad(chunk_start_time, chunk_end_time, vad_intervals):
    for interval in vad_intervals:
        vad_start = interval['start']
        vad_end = interval['end']
        # Check for overlap
        if chunk_end_time >= vad_start and chunk_start_time <= vad_end:
            return 1
    return 0

# Function to check if an edge is within VAD intervals
def is_edge_within_vad(src_time, dst_time, vad_intervals):
    src_within_vad = is_chunk_within_vad(src_time[0], src_time[1], vad_intervals)
    dst_within_vad = is_chunk_within_vad(dst_time[0], dst_time[1], vad_intervals)
    return src_within_vad and dst_within_vad

class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=2):
        super(GATLayer, self).__init__()
        self.gat = GATConv(in_channels, out_channels, heads=heads, concat=False, edge_dim=16)

    def forward(self, x, edge_index, edge_attr):
        out, attn_weights = self.gat(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        return out, attn_weights

class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GATModel, self).__init__()
        self.gat1 = GATLayer(in_channels, hidden_channels)
        self.gat2 = GATLayer(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        device = next(self.parameters()).device

        x, edge_index, edge_attr = x.to(device), edge_index.to(device), edge_attr.to(device)

        x, attn_weights1 = self.gat1(x, edge_index, edge_attr)
        x = torch.relu(x)
        x, attn_weights2 = self.gat2(x, edge_index, edge_attr)
        return x, (attn_weights1, attn_weights2)


import torch
import torch.nn as nn
import torch.nn.functional as F



class Anchor_node_update_Module(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 node_num_classes, num_scales=3,
                 num_refine_layers = 3,
                 dist_bins_list=[80, 60, 40]
                 ):

        super(Anchor_node_update_Module, self).__init__()
        # self.gat_model = GATModel(in_channels, hidden_channels, out_channels)
        # Fully Connected Layer for Node Classification

        # Three GAT models, one per scale
        self.gat_models = nn.ModuleList([
            GATModel(in_channels, hidden_channels, out_channels) for _ in range(num_scales)
        ])


        self.node_classifier = nn.Linear(out_channels, node_num_classes)
        self.interval_refine = Interval_Refine(
            node_embedding_dim= out_channels,
            num_refine_layers= num_refine_layers,
            dist_bins_list = dist_bins_list
        )

    def forward(self, all_scale_graphs, audio_dur, anchor_intervals  ): # pass the batch data

        device = next(self.parameters()).device
        # If edge labels exist
        num_audios = len(all_scale_graphs)
        num_scales = len(self.gat_models)

        node_predictions_list = []
        intervals_list = []
        bt_distill_loss = []

        interval_conf_list  = []
        interval_cls_list   = []


        # self.method_name()
        audio_dur = torch.tensor(audio_dur, dtype=torch.float)
        mean_duration = audio_dur.mean()  # Mean of all audio durations in the batch
        duration_ratios = audio_dur / mean_duration  # Shape: [num_audios]

        for i in range(num_audios):

            scale_embeddings = []
            for scale_idx in range(num_scales):
                cur_graph = all_scale_graphs[i][scale_idx].to(device)
                # Update nodes with scale-specific GAT
                embeddings, attn_weights = self.gat_models[scale_idx](cur_graph.x, cur_graph.edge_index, cur_graph.edge_attr)
                scale_embeddings.append(embeddings)



            # Combine embeddings from all scales
            node_embeddings = torch.cat(scale_embeddings, dim=0)  # Shape: (67, out_channels)
            cur_sample_anchor_intervals = anchor_intervals[i]
            # Apply GAT
            # node_embeddings, attn_weights = self.gat_model(cur_graph.x, cur_graph.edge_index, cur_graph.edge_attr)

            # Node classification
            node_preds = self.node_classifier(node_embeddings)


            # ---- 4. Compute Time Positions ----
            # Assume nodes are ordered temporally; normalize time to [0, 1]
            num_nodes = node_embeddings.size(0)
            time_positions = torch.linspace(0, 1, num_nodes, device=device).unsqueeze(-1)  # (num_nodes, 1)

            # print(f"node_pred shape before time_aware_fdr: {node_preds.shape if node_preds is not None else 'None'}")

            # Note, ----  learn interval Boundaries's  offset parameter ----
            event_output = self.interval_refine( # the instance of class EnhancedTimeAwareFDR
                node_embeddings=node_embeddings,
                time_positions=time_positions,
                node_pred=node_preds,
                audio_len=audio_dur[i],
                # duration_ratio  = cur_duration_ratio,  # You need to provide this if you have it
                cur_anchor_intervals = cur_sample_anchor_intervals,
                num_intervals_per_scale = [39,19, 9] # each scale's  num_anchor_intervals
            )  # Dummy batch index (all 0s)

            # Extract final bounds and mask
            cur_pred_interval = event_output["final_bounds"]  # (max_intervals, 2)
            cur_pred_interval_conf = event_output["interval_conf_logits"]
            cur_pred_interval_cls  = event_output["interval_cls_logits"]
            cur_distill_loss = event_output["distill_loss"]

            interval_conf_list.append(cur_pred_interval_conf)
            interval_cls_list.append(cur_pred_interval_cls)
            intervals_list.append(cur_pred_interval)
            bt_distill_loss.append(cur_distill_loss)

            node_predictions_list.append(node_preds)


        # Concatenate node predictions and labels
        node_predictions = torch.cat(node_predictions_list, dim=0)

        # After processing all audios:
        if bt_distill_loss:
            bt_distill_loss = sum(bt_distill_loss) / len(bt_distill_loss)

        else:
            bt_distill_loss = torch.tensor(0.0, device=device)

        return [node_predictions,
                intervals_list,
                interval_conf_list,
                interval_cls_list,
                bt_distill_loss,
                ]

from torch.nn.utils.rnn import pad_sequence

class GraphRespiratory(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes,
                 freq_bin = 84,
                 n_input_ch=3,
                 activation="relu",
                 conv_dropout=0,
                 kernel=[3, 3, 3, 3, 3],  # Extended for more layers
                 pad=[1, 1, 1, 1, 1],  # "same" padding
                 stride=[1, 1, 1, 1, 1],  # No downsampling
                 n_filt=[16, 64, 128, 256, 512],  # More channels
                 pooling=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],  # No downsampling
                 normalization="batch",
                 n_basis_kernels=4,
                 DY_layers=[0, 1, 1, 1, 1],  # Match n_filt length
                 temperature=31,
                 pool_dim='freq',
                 node_fea_dim= 512,
                 frame_hop= 128,  # Added frame_hop and fs
                 fs=8000,       # Sampling frequency
                 include_gender= False,
                 include_location= False
                 ):
        super(GraphRespiratory, self).__init__()
        self.frame_hop = frame_hop
        self.fs = fs

        self.num_scales = 3
        self.num_intervals_per_scale = [39, 19, 9]  # Fixed number of intervals per scale



        #  Feature Generator (CNN)
        self.spec_fea_generator = DynamicFeatureExtractor(n_input_ch,
                                                          activation=activation,
                                                          conv_dropout=conv_dropout,
                                                          kernel=kernel,
                                                          pad=pad,
                                                          stride=stride,
                                                          n_filt=n_filt,
                                                          pooling=pooling,
                                                          normalization=normalization,
                                                          n_basis_kernels=n_basis_kernels,
                                                          DY_layers=DY_layers,
                                                          temperature=temperature,
                                                          pool_dim=pool_dim,
                                                          stage="detect",
                                                          node_fea_dim=node_fea_dim)  # Define your CNN here



        # Scale-specific RNNs
        self.anchor_feat_generator = nn.ModuleList([
            nn.GRU(input_size=node_fea_dim, hidden_size=node_fea_dim, num_layers=2, batch_first=True)
            for _ in range(self.num_scales)
        ])
        self.include_gender = include_gender
        self.include_location = include_location

        # Determine the number of additional features
        num_additional_features = 0  # For node_vad_flags_tensor
        if self.include_gender:
            num_additional_features += 2  # Gender one-hot encoding size
        if self.include_location:
            num_additional_features += 4  # Location one-hot encoding size

        # Adjust in_channels accordingly
        adjusted_in_channels = node_fea_dim + num_additional_features

        # Edge feature encoder: from raw edge_attr (scalar) to learnable embedding
        edge_feature_dim = 16 # if you change this , the GAT layer,edge_dim=2 should also be changed ;
        self.edge_encoder = nn.Linear(1024, edge_feature_dim)
        self.anchor_interval_pred = Anchor_node_update_Module(adjusted_in_channels, hidden_channels, out_channels, node_num_classes=num_classes)



    def forward(self, batch_data,):
        device = next(self.parameters()).device


        # Extract data from batch_data
        spectrograms = batch_data['spectrograms']
        c_ex_mixtures = batch_data['c_ex_mixtures']

        # Extract gender and location information
        genders = batch_data['genders']  # List or tensor of length batch_size
        locations = batch_data['chest_loc']  # List or tensor of length batch_size

        audio_durations = batch_data['audio_dur']
        anchor_intervals = batch_data["anchor_intervals"]

        batch_indices = []

        # Step 1: Print the shapes for debugging
        # for i, spec in enumerate(spectrograms):
        #     print(f"Spectrogram {i} shape: {spec.shape}")
        # Step 2: Transpose the spectrograms to make the frames dimension the first dimension
        # From (3, frames, 84) to (frames, 3, 84)
        spectrograms_trans = [spec.transpose(0, 1) for spec in spectrograms]

        # Step 3: Pad spectrograms to the maximum frame length
        padded_spectrograms = pad_sequence(spectrograms_trans, batch_first=True, padding_value=0)
        # Shape after pad_sequence: (batch_size, max_frames, 3, 84) = (12, 961, 3, 84)

        # Step 4: Transpose back to the desired shape (batch, channels, frames, freq)
        padded_spectrograms = padded_spectrograms.permute(0, 2, 1, 3)
        # Shape: (12, 3, 961, 84)

        # Verify the final shape
        #print(f"Final shape: {padded_spectrograms.shape}")

        # (bt, frames, node_embedding  )
        bt_spec_feature  = self.spec_fea_generator(padded_spectrograms)  # Process in parallel
        batch_size = bt_spec_feature.size(0)

        # Step 2: Create masks
        # original_lengths = [577, 961, 577, 577, 961, 577, 961, 961, 577, 577, 577, 577]
        # masks = [torch.ones(3, length, 84) for length in original_lengths]
        # padded_masks = pad_sequence(masks, batch_first=True, padding_value=0)
        # # Shape: (12, 3, 961, 84)

        # Initialize a list to store scale-specific intervals for each sample
        scale_intervals = []
        # Iterate over each sample's anchor intervals
        for sample_intervals in anchor_intervals:

            # Split into starts and ends by indexing the second dimension
            starts = sample_intervals[:, 0]  # Shape: (67,)
            ends = sample_intervals[:, 1]  # Shape: (67,)
            # Split the starts and ends into three scales based on num_intervals_per_scale

            scale_starts = [
                starts[:39],  # First 39 intervals for scale 1
                starts[39:58],  # Next 19 intervals for scale 2 (indices 39 to 57)
                starts[58:67]  # Last 9 intervals for scale 3 (indices 58 to 66)
            ]

            scale_ends = [
                ends[:39],  # First 39 intervals for scale 1
                ends[39:58],  # Next 19 intervals for scale 2
                ends[58:67]  # Last 9 intervals for scale 3
            ]

            # Combine starts and ends for each scale into tuples
            sample_scale_intervals = [
                (scale_starts[0], scale_ends[0]),  # Scale 1: 39 intervals
                (scale_starts[1], scale_ends[1]),  # Scale 2: 19 intervals
                (scale_starts[2], scale_ends[2])  # Scale 3: 9 intervals
            ]

            # Append the scale-specific intervals for this sample
            scale_intervals.append(sample_scale_intervals)



        # Step 2: Parallel chunk extraction and RNN processing per scale
        all_node_features = []
        for scale_idx in range(self.num_scales):
            # Collect chunks for this scale across all samples
            chunks = []
            sequence_lengths = []
            for sample_idx in range(batch_size):
                spec = bt_spec_feature[sample_idx]
                # Assuming anchor_intervals[sample_idx] is a list of lists, one per scale

                scale_anchors = scale_intervals[sample_idx][scale_idx]

                starts, ends = scale_anchors  # Unpack the tuple into starts and ends tensors
                for start_time, end_time in zip(starts, ends):  # Iterate over pairs

                    start_frame = int(start_time * self.fs / self.frame_hop)
                    end_frame = int(end_time * self.fs / self.frame_hop)

                    # here should  automatic ignore the padded frames;
                    start_frame = max(0, min(start_frame, spec.shape[0] - 1))
                    end_frame = max(start_frame + 1, min(end_frame, spec.shape[0]))

                    chunk = spec[start_frame:end_frame, :]  # Shape: (seq_len, node_fea_dim)
                    chunks.append(chunk)
                    sequence_lengths.append(chunk.size(0))
            # processed  for  batch  samples  under  one  scale ;
            # Sort by sequence length for packing
            sorted_indices = sorted(range(len(chunks)), key=lambda i: sequence_lengths[i], reverse=True)
            sorted_chunks = [chunks[i] for i in sorted_indices]
            sorted_lengths = [sequence_lengths[i] for i in sorted_indices]

            # Pad sequences
            max_len = max(sorted_lengths)
            padded_chunks = torch.stack([
                F.pad(chunk, (0, 0, 0, max_len - chunk.size(0))) for chunk in sorted_chunks
            ], dim=0)  # Shape: (batch * num_intervals, max_len, node_fea_dim)

            # Pack sequences
            packed_chunks = pack_padded_sequence(padded_chunks, lengths=sorted_lengths, batch_first=True)

            # Process with scale-specific RNN
            _, h_n = self.anchor_feat_generator[scale_idx](packed_chunks)
            node_features = h_n[-1]  # Shape: (batch * num_intervals, node_fea_dim)

            # Unsort to original order
            unsorted_indices = torch.argsort(torch.tensor(sorted_indices, device=device))
            node_features = node_features[unsorted_indices]

            # Reshape to (batch, num_intervals_per_scale, node_fea_dim)
            node_features = node_features.view(batch_size, self.num_intervals_per_scale[scale_idx], -1)
            all_node_features.append(node_features)



        # Result: scale_intervals is a list of 12 elements, each a list of 3 tuples
        # Each tuple contains start and end tensors for a specific scale

            # Step 3: Concatenate node features from all scales
        node_features = torch.cat(all_node_features, dim=1)  # Shape: (batch, total_intervals, node_fea_dim)

        # Step 4: Include gender and location info (if applicable)
        if self.include_gender or self.include_location:
            batch_node_features = []
            for sample_idx in range(batch_size):
                sample_features = node_features[sample_idx]
                extra_features = []
                if self.include_gender:
                    gender_one_hot = F.one_hot(torch.tensor(genders[sample_idx], dtype=torch.long, device=device),
                                               num_classes=2).float()
                    extra_features.append(gender_one_hot.repeat(sample_features.size(0), 1))
                if self.include_location:
                    location_one_hot = F.one_hot(
                        torch.tensor(locations[sample_idx], dtype=torch.long, device=device), num_classes=4).float()
                    extra_features.append(location_one_hot.repeat(sample_features.size(0), 1))
                if extra_features:
                    sample_features = torch.cat([sample_features] + extra_features, dim=1)
                batch_node_features.append(sample_features)
            node_features = torch.stack(batch_node_features, dim=0)




        # Step 5: Graph construction and prediction
        all_scale_graphs = []
        for sample_idx in range(batch_size):

            sample_features = node_features[sample_idx]
            sample_anchors = anchor_intervals[sample_idx]  # List of lists, one per scale

            # anchor_splits = [torch.tensor(anchors, device=device) for anchors in sample_anchors]
            anchor_splits = torch.split(sample_anchors, self.num_intervals_per_scale, dim=0)
            feature_splits = torch.split(sample_features, self.num_intervals_per_scale, dim=0)
            scale_graphs = []


            for scale_idx, (scale_anchors, scale_features) in enumerate(zip(anchor_splits, feature_splits)):
                    # Create edges within this scale
                    num_nodes = scale_features.size(0)
                    edge_index = []
                    for i in range(num_nodes):
                        for j in range(i + 1, num_nodes):
                            if scale_anchors[i][1] >= scale_anchors[j][0] and scale_anchors[j][1] >= \
                                    scale_anchors[i][0]:
                                edge_index.append([i, j])
                                edge_index.append([j, i])

                    edge_index = torch.tensor(edge_index,
                                              dtype=torch.long).t().contiguous() if edge_index else torch.empty(
                        (2, 0), dtype=torch.long)

                    # Generate edge features (if applicable)
                    edge_attr = None
                    if edge_index.size(1) > 0 and hasattr(self, 'edge_encoder'):
                        src, dst = edge_index
                        combined = torch.cat([scale_features[src], scale_features[dst]], dim=1)
                        edge_attr = self.edge_encoder(combined)

                    # Create graph for this scale
                    graph = Data(x=scale_features, edge_index=edge_index, edge_attr=edge_attr)
                    scale_graphs.append(graph)

            all_scale_graphs.append(scale_graphs)

            # Batch all graphs together
            # batch_graph = Batch.from_data_list(all_graphs).to(device)

            # Node classification and interval prediction
        pred_output = self.anchor_interval_pred(all_scale_graphs, audio_durations, anchor_intervals)

        node_predictions = pred_output[0]
        pred_intervals = pred_output[1]
        pred_interval_conf = pred_output[2]
        pred_interval_cls = pred_output[3]
        distill_loss = pred_output[4]



        # print(f"node_predictions shape: {node_predictions.shape}")  # Should output [sum(N_i), num_classes]

        # Prepare outputs for the node predictions
        outputs = {
            'node_predictions': node_predictions,
            'pred_intervals': pred_intervals,

            'pred_intervals_conf_logits': pred_interval_conf,
            'pred_intervals_cls_logits': pred_interval_cls,

            'distill_loss': distill_loss,


            'batch_indices': batch_indices,
            'batch_audio_names': c_ex_mixtures  # This should correspond to the samples in detection
        }

        return outputs





# Example instantiation:
# model = GraphRespiratory(in_channels=512 + 1, hidden_channels=128, out_channels=128, num_classes=7, n_input_ch=1)



# Example instantiation:
# model = GNNRespiraModel(in_channels=64, hidden_channels=128, out_channels=128, num_classes=5)
# This model can then be passed to a PyTorch Lightning Trainer.

from torch_geometric.data import Data, Batch
import random

if __name__ == "__main__":
        pass




"""

Option 2 (Using src_time and dst_time):选项 2（使用src_time和dst_time ）：

If you prefer a more precise calculation that checks the actual 
time intervals of the nodes against the VAD intervals, you can use the is_edge_within_vad function.
如果您希望进行更精确的计算，根据 VAD 间隔检查节点的实际时间间隔，则可以使用is_edge_within_vad函数。


This function considers whether the time intervals of both the source 
and destination nodes overlap with any VAD intervals.
该功能考虑源节点和目的节点的时间间隔是否与任何VAD间隔重叠。

This approach might be beneficial if you have overlapping VAD intervals or 
if the node VAD flags do not capture the timing nuances you're interested in.
如果您有重叠的 VAD 间隔，或者节点 VAD 标志未捕获您感兴趣的时序细微差别，则此方法可能会很有用。


"""

