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
from desed_task.nnet.DCNN_v3_4 import  DynamicFeatureExtractor

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

#  v8-14_3   使用用 hard confidence ,硬置信度，  但是节点分类不参与计算；


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
                    nn.Linear(node_embedding_dim + 5, 256),
                    nn.ReLU(),
                    nn.Linear(256, 2 * dist_bins_list[i])
                ) for _ in range(num_refine_layers)
            ]) for i in range(self.num_scales)
        ])

        # Scale-specific local feature GRUs;
        # --- Learnable Local Aggregators ---
        # For each candidate interval, we use GRUs to aggregate node features and abnormal scores.
        self.local_feat_rnns = nn.ModuleList([
            nn.GRU(input_size=node_embedding_dim, hidden_size=node_embedding_dim, batch_first=True)
            for _ in range(self.num_scales)
        ])

        # Scale-specific local abnormal GRUs
        self.local_abnormal_rnns = nn.ModuleList([
            nn.GRU(input_size=1, hidden_size=1, batch_first=True)
            for _ in range(self.num_scales)
        ])


        self.weight_params = nn.ParameterList([
            nn.Parameter(torch.linspace(-0.5, 0.5, dist_bins_list[i]))
            for i in range(self.num_scales)
        ])

        # Scale-specific confidence heads;
        # NEW: Confidence head for normal (0) vs. abnormal (1)
        # We'll feed the same local feature vector into this head
        self.interval_conf_heads = nn.ModuleList([
            nn.Linear(node_embedding_dim + 1, 1) for _ in range(self.num_scales)
        ])

        # Scale-specific classification heads;
        # We'll feed it [local_feat, local_abnormal_feature], shape => (node_embedding_dim + 1)
        #  not include normal class;
        self.interval_class_heads = nn.ModuleList([
            nn.Linear(node_embedding_dim + 1, self.num_classes - 1) for _ in range(self.num_scales)
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

         # Split anchor intervals by scale,  into  three scale;
        split_indices = torch.cumsum(torch.tensor(num_intervals_per_scale, device=device), dim=0)
        anchor_intervals_list = torch.split(cur_anchor_intervals, num_intervals_per_scale, dim=0)

        # Lists to collect results from all scales
        final_bounds_list = []
        interval_conf_logits_list = []
        interval_cls_logits_list = []

        distill_loss_value = 0
        # Process each scale independently
        for scale_idx in range(self.num_scales):
            anchor_intervals = anchor_intervals_list[scale_idx]
            num_intervals_scale = num_intervals_per_scale[scale_idx]

            if num_intervals_scale == 0:
                continue

            # Select scale-specific components
            local_feat_rnn = self.local_feat_rnns[scale_idx]
            local_abnormal_rnn = self.local_abnormal_rnns[scale_idx]
            refinement_layers = self.refinement_layers[scale_idx]
            weight_params = self.weight_params[scale_idx]
            interval_conf_head = self.interval_conf_heads[scale_idx]
            interval_class_head = self.interval_class_heads[scale_idx]

            # 1. Duration-aware time embedding (shared across scales)
            # expanded_ratio = torch.full_like(time_positions,
            #                                  duration_ratio) if duration_ratio is not None else torch.ones_like(
            #     time_positions)
            # time_feat = self.time_embed(torch.cat([time_positions, expanded_ratio], dim=-1))
            # x = node_embeddings + time_feat

            x = node_embeddings

            # 2. Learnable smoothing of node_pred (shared across scales)
            smoothed_pred = F.softmax(self.smoothing_conv(node_pred.T).T, dim=-1)
            abnormal_scores = 1 - smoothed_pred[:, self.normal_class_idx]

            # 3. Compute masks for this scale's intervals
            time_pos_1d = time_positions.squeeze(-1) * audio_len.item()
            starts = anchor_intervals[:, 0].clone()
            ends = anchor_intervals[:, 1].clone()
            in_interval_mask = (time_pos_1d.unsqueeze(0) >= starts.unsqueeze(1)) & \
                               (time_pos_1d.unsqueeze(0) <= ends.unsqueeze(1))  # (num_intervals_scale, num_nodes)

            # 4. Extract local features for this scale
            local_x_list = []
            local_abnormal_seq_list = []
            sequence_lengths = []

            for i in range(num_intervals_scale):
                mask = in_interval_mask[i]
                if mask.sum() > 0:
                    local_x = x[mask]
                    local_abnormal = abnormal_scores[mask].unsqueeze(-1)
                else:
                    local_x = torch.zeros((1, self.node_embedding_dim), device=device)
                    local_abnormal = torch.zeros((1, 1), device=device)
                local_x_list.append(local_x)
                local_abnormal_seq_list.append(local_abnormal)
                sequence_lengths.append(local_x.size(0))

            # Sort by sequence length for packing
            sorted_indices = sorted(range(num_intervals_scale), key=lambda i: sequence_lengths[i], reverse=True)
            sorted_local_x = [local_x_list[i] for i in sorted_indices]
            sorted_local_abnormal = [local_abnormal_seq_list[i] for i in sorted_indices]
            sorted_lengths = [sequence_lengths[i] for i in sorted_indices]

            # Pad and pack sequences
            max_len = max(sorted_lengths)# pad_local_x: (num_scale_anchor_intervals,  max_length_for_local_x, node_emb_dim)
            padded_local_x = torch.stack([F.pad(seq, (0, 0, 0, max_len - seq.size(0))) for seq in sorted_local_x],
                                         dim=0)
            padded_local_abnormal = torch.stack(
                [F.pad(seq, (0, 0, 0, max_len - seq.size(0))) for seq in sorted_local_abnormal], dim=0)
            packed_local_x = pack_padded_sequence(padded_local_x, lengths=sorted_lengths, batch_first=True)
            packed_local_abnormal = pack_padded_sequence(padded_local_abnormal, lengths=sorted_lengths,
                                                         batch_first=True)

            # 5. Process through scale-specific GRUs
            _, h_local = local_feat_rnn(packed_local_x)
            _, h_local_abnormal = local_abnormal_rnn(packed_local_abnormal)

            # Unsort to original order ,# (num_scale_anchor_intervals, node_embed)
            local_feat = h_local.squeeze(0)[torch.argsort(torch.tensor(sorted_indices, device=device))]
            local_abnormal_feature = h_local_abnormal.squeeze(0).squeeze(-1)[
                torch.argsort(torch.tensor(sorted_indices, device=device))]

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
                    local_abnormal_feature.unsqueeze(-1)
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
            local_feat_full = torch.cat([local_feat, local_abnormal_feature.unsqueeze(-1)], dim=1)
            interval_conf_logits_scale = interval_conf_head(local_feat_full)
            interval_cls_logits_scale = interval_class_head(local_feat_full)

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



class Node_Interval_cls_Module(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 node_num_classes, edge_num_classes=2,
                 num_refine_layers = 3,
                 dist_bins_list=[80, 60, 40]
                 ):

        super(Node_Interval_cls_Module, self).__init__()
        self.gat_model = GATModel(in_channels, hidden_channels, out_channels)
        # Fully Connected Layer for Node Classification
        self.node_classifier = nn.Linear(out_channels, node_num_classes)
        self.interval_refine = Interval_Refine(
            node_embedding_dim= out_channels,
            num_refine_layers= num_refine_layers,
            dist_bins_list = dist_bins_list
        )

    def forward(self, data, audio_dur, anchor_intervals  ): # pass the batch data
        device = data.x.device

        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        edge_attr = data.edge_attr.to(device)
        batch_indices = data.batch.to(device)
        node_labels = data.y.to(device)

        # If edge labels exist
        edge_labels = data.edge_y.to(device) if hasattr(data, 'edge_y') else None
        num_audios = batch_indices.max().item() + 1

        node_predictions_list = []
        all_node_labels = []


        intervals_list = []
        bt_distill_loss = []

        interval_conf_list  = []
        interval_cls_list   = []


        # self.method_name()
        audio_dur = torch.tensor(audio_dur, dtype=torch.float)
        mean_duration = audio_dur.mean()  # Mean of all audio durations in the batch
        duration_ratios = audio_dur / mean_duration  # Shape: [num_audios]

        for i in range(num_audios):
            mask = (batch_indices == i)
            cur_nodes = x[mask]
            cur_node_labels = node_labels[mask]


            cur_duration_ratio = duration_ratios[i]  # Scalar for current audio
            cur_duration = audio_dur[i]

            cur_sample_anchor_intervals = anchor_intervals[i]

            # Get the global indices of the nodes for the current audio
            cur_nodes_indices = mask.nonzero(as_tuple=False).view(-1)

            # Filter edges that belong to the current audio
            edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
            cur_edge_index0 = edge_index[:, edge_mask]
            cur_edge_attr0 = edge_attr[edge_mask]

            # Map global indices to local indices
            mapping = {global_idx.item(): local_idx for local_idx, global_idx in enumerate(cur_nodes_indices)}
            cur_edge_index = torch.stack(
                [torch.tensor([mapping[idx.item()] for idx in edge], dtype=torch.long, device=device)
                 for edge in cur_edge_index0],
                dim=1
            ).t()

            # Extract edge labels for the current audio if available
            if edge_labels is not None:
                cur_edge_labels = edge_labels[edge_mask]
            else:
                cur_edge_labels = None

            # Create Data object for the current audio subgraph
            cur_graph = Data(x=cur_nodes, edge_index=cur_edge_index, edge_attr=cur_edge_attr0, y=cur_node_labels)
            if cur_edge_labels is not None:
                cur_graph.edge_y = cur_edge_labels

            # Apply GAT
            node_embeddings, attn_weights = self.gat_model(cur_graph.x, cur_graph.edge_index, cur_graph.edge_attr)

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
                audio_len = cur_duration,
                duration_ratio  = cur_duration_ratio,  # You need to provide this if you have it
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
            all_node_labels.append(cur_node_labels)




        # Concatenate node predictions and labels
        node_predictions = torch.cat(node_predictions_list, dim=0)
        node_labels = torch.cat(all_node_labels, dim=0)
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



class GraphRespiratory(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes,
                 n_input_ch,
                 activation="Relu",
                 conv_dropout=0,
                 kernel=[3, 3, 3],
                 pad=[1, 1, 1],
                 stride=[1, 1, 1],
                 n_filt=[16, 64, 128],  # n channels
                 pooling=[(1, 4), (1, 4), (1, 4)],
                 normalization="batch",
                 n_basis_kernels=4,
                 DY_layers=[0, 1, 1, 1, 1, 1, 1],
                 temperature=31,
                 pool_dim='freq',
                 node_fea_dim= 256,
                 node_vad_dim = 256,
                 frame_hop= 128,  # Added frame_hop and fs
                 fs=8000,       # Sampling frequency
                 include_gender= False,
                 include_location= False
                 ):
        super(GraphRespiratory, self).__init__()
        self.frame_hop = frame_hop
        self.fs = fs
        # Node Feature Generator (CNN)
        self.node_fea_generator = DynamicFeatureExtractor(n_input_ch,
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

        # use for time stamp generator
        # self.timestamp_generator = DynamicTimestampGenerator(node_fea_dim, num_classes=num_classes)
        # use for  subgraph builder
        # self.subgraph_builder = EventSubgraphBuilder()
        # # use for  refine the boundary;
        # self.dfine_refiner = FDR(num_classes)

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

        # self.node_vad_thread = 0.5


        self.node_vad_proj =  nn.Sequential( nn.Linear(node_fea_dim, 128),
                                            # Example hidden layer
                                             nn.ReLU(),
                                             nn.Dropout(0.5),
                                             nn.Linear(128, 2)
                                                )


        # Edge feature encoder: from raw edge_attr (scalar) to learnable embedding
        edge_feature_dim = 16 # if you change this , the GAT layer,edge_dim=2 should also be changed ;
        self.edge_encoder = nn.Linear(4, edge_feature_dim)
        self.node_interval_pred = Node_Interval_cls_Module(adjusted_in_channels, hidden_channels, out_channels, node_num_classes=num_classes)

        # Factor graph layer for enforcing node-edge consistency (You must define this layer)
        # self.factor_graph_layer = FactorGraphLayer(node_num_classes=num_classes, edge_num_classes=2, num_iterations=2, gamma=1.0)

    def forward(self, batch_data):
        device = next(self.parameters()).device


        # Extract data from batch_data
        spectrograms = batch_data['spectrograms']
        frame_labels = batch_data['frame_labels']

        c_ex_mixtures = batch_data['c_ex_mixtures']
        vad_timestamps = batch_data['vad_timestamps']

        # Extract gender and location information
        genders = batch_data['genders']  # List or tensor of length batch_size
        locations = batch_data['chest_loc']  # List or tensor of length batch_size

        audio_durations = batch_data['audio_dur']
        anchor_intervals = batch_data["anchor_intervals"]

        # audio_durations = batch_data['audio_durations']


        # Initialize lists
        all_chunks = []
        all_chunk_labels = []

        batch_indices = []
        num_nodes_per_sample = []

        chunk_times_all = []

        node_gender_all = []
        node_location_all = []

        # for sample_idx, (spec, labels, vad_intervals, audio_dur) in enumerate(
        #         zip(spectrograms, frame_labels, vad_timestamps, audio_durations)):

        for sample_idx, (spec, labels, vad_intervals,audio_dur) in enumerate(
                    zip(spectrograms, frame_labels, vad_timestamps, audio_durations)):
            gender = genders[sample_idx]  # 0 or 1
            location = locations[sample_idx]  # 0 to 3

            n_frames = spec.shape[1]
            num_nodes = 0

            # Define chunk size
            chunk_size = 5  # Adjust as needed
            # Get frame times
            # frame_times = np.arange(n_frames + 1) * self.frame_hop / self.fs
            frame_times = np.arange(n_frames ) * self.frame_hop / self.fs
            # Adjust VAD timestamps
            adjusted_vad_intervals = adjust_vad_timestamps(vad_intervals, frame_times)

            # 2) Build a boolean mask: frame_vad_intervals[i] = True if frame i is within an abnormal interval
            frame_vad_intervals = np.zeros(n_frames, dtype=bool)
            for  interval in adjusted_vad_intervals:
                # Mark frames whose start-time is in [start, end)
                # You could also check the center of each frame if preferred.
                start = interval["start"]
                end   = interval["end"]
                in_interval = (frame_times >= start) & (frame_times < end)
                frame_vad_intervals |= in_interval  # combine intervals



            # 1.  ----------------  gen the node label, node vad label-------------     Process spectrogram into fixed-size chunks
            chunk_times = []

            for j in range(0, n_frames - chunk_size + 1, chunk_size):
                chunk = spec[:, j:j + chunk_size, :]
                all_chunks.append(chunk)

                # Compute node labels for the chunk
                label_chunk = labels[j:j + chunk_size, :]  # Adjusted indexing for shape (chunk_size, 7)

                # Check if this chunk overlaps any abnormal frames
                chunk_vad_slice = frame_vad_intervals[j : j + chunk_size]
                if not chunk_vad_slice.any():
                    # If the chunk is entirely outside the abnormal intervals, assign normal label = 0
                    node_label = 0
                else:
                    # If the chunk is within abnormal intervals, force an abnormal label
                    node_label = self.get_node_label(
                        label_chunk,
                        force_abnormal=True  # <--- new argument
                    )


                # node_label = self.get_node_label(label_chunk)
                all_chunk_labels.append(node_label)

                # Compute chunk start and end times
                chunk_start_frame = j
                chunk_end_frame = j + chunk_size - 1
                chunk_start_time = frame_times[chunk_start_frame]
                chunk_end_time = frame_times[chunk_end_frame]
                chunk_times.append((chunk_start_time, chunk_end_time)) #   记录每个chunk的 起始，终止时间。


                # Append gender and location info
                node_gender_all.append(gender)
                node_location_all.append(location)

                num_nodes += 1

                # Append the sample index for this node
                batch_indices.append(sample_idx)

            # Handle the last chunk if necessary
            last_chunk_end = (n_frames - chunk_size + 1) + chunk_size - 1
            if last_chunk_end < n_frames - 1:
                chunk = spec[:, n_frames - chunk_size:, :]
                all_chunks.append(chunk)

                # Compute node labels for the chunk
                # label_chunk = labels[:, n_frames - chunk_size:]
                label_chunk = labels[n_frames - chunk_size:, :]
                chunk_vad_slice = frame_vad_intervals[n_frames - chunk_size:]
                if not chunk_vad_slice.any():
                    node_label = 0
                else:
                    node_label = self.get_node_label(label_chunk, force_abnormal=True)
                all_chunk_labels.append(node_label)
                # node_label = self.get_node_label(label_chunk)
                all_chunk_labels.append(node_label)

                # Compute chunk start and end times
                chunk_start_frame = n_frames - chunk_size
                chunk_end_frame = n_frames - 1
                chunk_start_time = frame_times[chunk_start_frame]
                chunk_end_time = frame_times[chunk_end_frame]
                chunk_times.append((chunk_start_time, chunk_end_time))


                # Append gender and location info
                node_gender_all.append(gender)
                node_location_all.append(location)

                num_nodes += 1
                # Append the sample index for this node
                batch_indices.append(sample_idx)

            num_nodes_per_sample.append(num_nodes)
            chunk_times_all.extend(chunk_times)

        if not all_chunks:
            raise ValueError("No valid chunks generated in the batch.")

        # Stack all chunks and process them together
        all_chunks = [torch.tensor(chunk, dtype=torch.float32) if not isinstance(chunk, torch.Tensor) else chunk for
                      chunk in all_chunks]
        all_chunks_tensor = torch.stack(all_chunks)


        # -------------2. Generate node features----------------------
        #  node_features: (total_nodes, feature_dim=256), node_vad: (total_nodes, feature_dim=256)
        node_features  = self.node_fea_generator(all_chunks_tensor)
        node_vad_fea = self.node_vad_proj(node_features)  # tensors: (total_nodes, fea_dim=2)


        # Stack node labels,       # Convert batch_indices to tensor
        node_labels = torch.tensor(all_chunk_labels, dtype=torch.long, device=node_features.device)
        batch_indices = torch.tensor(batch_indices, dtype=torch.long, device=node_features.device)

        # Include gender information if enabled
        if self.include_gender:
            node_gender_tensor = torch.tensor(node_gender_all, dtype=torch.long, device=node_features.device)
            node_gender_one_hot = F.one_hot(node_gender_tensor, num_classes=2).float()
            node_features = torch.cat([node_features, node_gender_one_hot], dim=1)

        # Include location information if enabled
        if self.include_location:
            node_location_tensor = torch.tensor(node_location_all, dtype=torch.long, device=node_features.device)
            node_location_one_hot = F.one_hot(node_location_tensor, num_classes=4).float()
            node_features = torch.cat([node_features, node_location_one_hot], dim=1)

        #------------------------- Create graphs for each sample-------------------
        all_graphs = []
        all_edge_attr = []
        start_idx = 0
        node_offset = 0
        for num_nodes in num_nodes_per_sample:
            end_idx = start_idx + num_nodes

            sample_node_features = node_features[start_idx:end_idx]
            sample_node_labels = node_labels[start_idx:end_idx]      # 代表当前样本上每个节点的标签；
            sample_chunk_times = chunk_times_all[start_idx:end_idx]  # 每个节点，所对应的5帧，所处于的时间戳区间，
            sample_node_vad_fea = node_vad_fea[start_idx:end_idx]

            # Create edge index
            num_nodes_sample = sample_node_features.size(0)
            if num_nodes_sample > 1:
                edge_index = torch.stack([
                    torch.arange(num_nodes_sample - 1, dtype=torch.long, device=node_features.device),
                    torch.arange(1, num_nodes_sample, dtype=torch.long, device=node_features.device)
                ], dim=0)



                # Generate edge attributes and labels
                edge_attrs = []
                edge_labels = []
                for src, dst in zip(edge_index[0], edge_index[1]):
                    src_time = sample_chunk_times[src.item()]
                    dst_time = sample_chunk_times[dst.item()]

                    # Determine if edge is within vad interval，
                    #  must do not use this,
                    # this  actually  leaks  the label information to the model;
                    # because it leaks  the time info to generate the  features
                    # edge_within_vad = is_edge_within_vad(src_time, dst_time, adjusted_vad_intervals)

                    # Ground truth edge label:
                    # Abnormal if either node is abnormal
                    # node abnormal: node_label != 0
                    src_label = sample_node_labels[src]
                    dst_label = sample_node_labels[dst]
                    edge_is_abnormal = (src_label != 0 or dst_label != 0 )
                    edge_label = 1 if edge_is_abnormal else 0
                    edge_labels.append(edge_label)

                    # Initial edge attribute (scalar), for example just the edge_within_vad as a float
                    # edge_attr_val = float(edge_within_vad)
                    # edge_attrs.append([edge_attr_val])  #  这里边的属性不能依靠 时间戳信息生成；

                #edge_attr = torch.tensor(edge_attrs, dtype=torch.float32, device=device)
                edge_labels = torch.tensor(edge_labels, dtype=torch.long, device=device)

                # Generate edge_features based on connected node features
                src_node_vad = sample_node_vad_fea[edge_index[0]]  # [E, feature_dim=32]
                dst_node_vad = sample_node_vad_fea[edge_index[1]]
                combined_edge_features = torch.cat([src_node_vad, dst_node_vad],
                                                   dim=1)  # [E, 2 * dim=1] (191, 2*1)

                # Encode edge attributes to learnable embeddings
                edge_features = self.edge_encoder(combined_edge_features)  # 这里边的属性，直接使用一个可学习参数；


            else:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
                edge_attr = torch.empty((0, 1), dtype=torch.float32, device=device)
                edge_features = edge_attr  # no edges
                edge_labels = torch.empty((0,), dtype=torch.long, device=device)

            # Create graph data object
            graph_data = Data(x=sample_node_features, edge_index=edge_index,
                              edge_attr=edge_features,
                              y=sample_node_labels)

            # print(f"Encoded edge_attr shape: {graph_data.edge_attr.shape}")  # Should output [num_edges, 1]

            # Store edge labels in the graph data
            graph_data.edge_y = edge_labels

            all_graphs.append(graph_data)
            start_idx = end_idx

        # Batch all graphs together,
        batch_graph = Batch.from_data_list(all_graphs).to(all_chunks_tensor.device)

        # ------------------- batch Node-  Classification-----------------
        pred_output = self.node_interval_pred(batch_graph, audio_durations, anchor_intervals )

        node_predictions =  pred_output[0]
        pred_intervals = pred_output[1]
        pred_interval_conf = pred_output[2]
        pred_interval_cls = pred_output[3]

        distill_loss = pred_output[4]

        # print(f"node_predictions shape: {node_predictions.shape}")  # Should output [sum(N_i), num_classes]

        # Prepare outputs for the node predictions
        outputs = {
            'node_predictions': node_predictions,
            'node_labels': node_labels,


            'pred_intervals': pred_intervals,
            'pred_intervals_conf_logits': pred_interval_conf,
            'pred_intervals_cls_logits': pred_interval_cls,

            'distill_loss': distill_loss,

            'batch_edge_index': batch_graph.edge_index,
            'batch_graph':batch_graph,

            'batch_indices': batch_indices,
            'batch_audio_names': c_ex_mixtures  # This should correspond to the samples in detection
        }

        return outputs


    def get_node_label(self,
                       frames,
                       normal_label=0,
                       abnormal_threshold=0.2,
                       force_abnormal=False):
        """
        Returns a single integer label for the chunk of frames.

        If force_abnormal=True, we do not allow a normal label to be returned;
        we pick the most likely abnormal label. If no abnormal frames exist,
        it will fall back to normal_label=0 (or you can define another strategy).
        """
        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()

        # frames shape: (n_chunk_frames, num_classes)
        frames_label = frames.argmax(axis=1)  # index of the max per-frame
        total_frames = len(frames_label)
        num_classes = frames.shape[1]

        counts = np.bincount(frames_label, minlength=num_classes)
        majority_label = counts.argmax()
        majority_count = counts[majority_label]

        # If we want to force an abnormal label, pick from among labels != normal_label
        if force_abnormal:
            # Filter out normal label
            # Only keep labels with a positive count and != normal_label
            abnormal_counts = [
                (lbl, cnt) for lbl, cnt in enumerate(counts)
                if lbl != normal_label and cnt > 0
            ]
            if not abnormal_counts:
                # No abnormal frames at all -> fallback to normal_label
                return normal_label

            max_count = max(abnormal_counts, key=lambda x: x[1])[1]
            max_labels = [lbl for (lbl, cnt) in abnormal_counts if cnt == max_count]

            if len(max_labels) == 1:
                return max_labels[0]
            else:
                # Tie-break by earliest occurrence
                earliest_indices = {}
                for lbl in max_labels:
                    # frames_label == lbl won't be empty because cnt>0
                    idx_array = np.where(frames_label == lbl)[0]
                    earliest_indices[lbl] = idx_array[0]
                selected_label = min(earliest_indices.items(), key=lambda x: x[1])[0]
                return selected_label
        else:
            # Original logic if we do not force abnormal
            if majority_label == normal_label:
                # Identify abnormal labels exceeding threshold
                significant_abnormals = [
                    (label, count) for label, count in enumerate(counts)
                    if label != normal_label and count > abnormal_threshold * total_frames
                ]
                if significant_abnormals:
                    # Find the abnormal label(s) with the highest ratio
                    max_count = max(significant_abnormals, key=lambda x: x[1])[1]
                    max_labels = [lbl for lbl, cnt in significant_abnormals if cnt == max_count]
                    if len(max_labels) == 1:
                        return max_labels[0]
                    else:
                        # Tie-break by earliest occurrence
                        earliest_indices = {
                            lbl: np.where(frames_label == lbl)[0][0] for lbl in max_labels
                        }
                        selected_label = min(earliest_indices.items(), key=lambda x: x[1])[0]
                        return selected_label
                else:
                    return normal_label
            else:
                return majority_label


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

