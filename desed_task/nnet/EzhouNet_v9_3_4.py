from torch_geometric.nn import GATConv, global_mean_pool
from torch import Tensor
from desed_task.nnet.DCNN_v3_4   import  DynamicFeatureExtractor

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


# # v8-13_3,  基于8-13-2， 纠正focal loss 的 alpha 权重比率；
# 并且减少 数据增强的比率；


# # v8-13_4,  基于8-13-3，  实现 self.gat model 并行化；
#  self.anchor_interval  暂时没有实现并行化计算，
# 因为此时在 spr数据集中，音频的长度，以及每个样本中节点的个数是不同的；


# # v8-18, self.gat_model  并行化；
# 1. refine layer,  interval confidence, inteval cls
# 这些层增加隐藏层 以及dropout ,  提高模型的学习容量， 并且降低过拟合；
# 2. 节点和区间参数的使用两个不同的学习率进行分开优化， 并对区间使用余弦退火算法
# 3. 将定位损失添加 L1 距离损失作为可选项；




#  v9-1-1,  修改先验区间的持续时间为 ， 并且保证尺度2的个数最多，
# Scale 1: 0.5s, Scale 2: 0.8s, Scale 3: 1.5s
# 对于粗粒度的 区间，同样分配更多的bins，  从而可以进行更大范围的偏移学习；
# 即较粗的间隔需要更多的偏移容量才能有效地细化。



# 从 20 帧一个节点回退到 5 帧一个节点；


import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
class Interval_Refine(nn.Module):
    def __init__(self, node_embedding_dim, num_classes=5, num_refine_layers=1,
                 dist_bins_list=None, kernel_size=5,  ):
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
        # Input: [local_feature, center, width,  local_abnormal_feature]
        # (dimension = node_embedding_dim + 3)

        self.node_embed_adaptive = nn.Sequential(
            nn.Linear(node_embedding_dim, node_embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(node_embedding_dim)
        )


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

        # self.local_abnormal_rnns = nn.ModuleList([
        #     nn.GRU(input_size=1, hidden_size=16, num_layers=1, batch_first=True)
        #     for _ in range(self.num_scales)
        # ])
        # for rnn in self.local_abnormal_rnns:
        #     nn.init.orthogonal_(rnn.weight_hh_l0)
        #     nn.init.orthogonal_(rnn.weight_ih_l0)
        #     nn.init.zeros_(rnn.bias_hh_l0)
        #     nn.init.zeros_(rnn.bias_ih_l0)
        #
        # for rnn in self.local_abnormal_rnns:
        #     for name, param in rnn.named_parameters():
        #         print(f"{name}: {param.shape}")


        # self.weight_params = nn.ParameterList([
        #     nn.Parameter(torch.linspace(-0.5, 0.5, dist_bins_list[i]))
        #     for i in range(self.num_scales)
        # ])

        self.weight_params = nn.ParameterList([
            nn.Parameter(torch.linspace(-1.0, 1.0, dist_bins_list[i]))
            for i in range(self.num_scales)
        ])

        # Common head for localization logits, confidence, classification
        self.interval_unify_head =  nn.ModuleList([
                nn.Sequential(
                    nn.Linear(node_embedding_dim + 3, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),  # Add dropout for regularization
                    nn.Linear(256, 256),  # Additional layer for more capacity
                    nn.ReLU(),
                    nn.Dropout(0.2),  # Add dropout
                    nn.Linear(256, 2 * dist_bins_list[i] + 1 + (self.num_classes - 1)))
            for i in range(self.num_scales)
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

        if not torch.is_tensor(cur_anchor_intervals):
            cur_anchor_intervals = torch.tensor(cur_anchor_intervals, device=device)
        assert sum(num_intervals_per_scale) == cur_anchor_intervals.size(
            0), f"Interval count mismatch: {sum(num_intervals_per_scale)} vs {cur_anchor_intervals.size(0)}"

        if num_intervals_per_scale is None:
            raise ValueError("num_intervals_per_scale must be provided as a list of integers [n1, n2, n3]")

         # Split anchor intervals by scale,  into  three scale;
        split_indices = torch.cumsum(torch.tensor(num_intervals_per_scale, device=device), dim=0)
        anchor_intervals_list = torch.split(cur_anchor_intervals, num_intervals_per_scale, dim=0)

        node_embed_adaptive =  self.node_embed_adaptive(node_embeddings)


        # Lists to collect results from all scales
        final_bounds_list = []
        interval_conf_logits_list = []
        interval_cls_logits_list = []

        # print("node_embeddings shape:", node_embeddings.shape)
        # print("cur_anchor_intervals shape:", cur_anchor_intervals.shape)
        # print("num_intervals_per_scale:", num_intervals_per_scale)
        # Process each scale independently
        for scale_idx in range(self.num_scales):


            anchor_intervals = anchor_intervals_list[scale_idx]
            num_intervals_scale = num_intervals_per_scale[scale_idx]

            if num_intervals_scale == 0:
                continue

            # Select scale-specific components
            local_feat_rnn = self.local_feat_rnns[scale_idx]
            local_abnormal_rnn = self.local_abnormal_rnns[scale_idx]
            #refinement_layers = self.refinement_layers[scale_idx]
            unify_head = self.interval_unify_head[scale_idx]
            weight_params = self.weight_params[scale_idx]


            # Cosine positional encoding with scaling
            scale_factor = 0.05  # Adjustable
            freqs = torch.linspace(0, 10, self.node_embedding_dim, device=device)  # Shape: (node_embedding_dim,)
            temporal_encoding = scale_factor * torch.sin(time_positions * freqs)  # Shape: (num_nodes, node_embedding_dim)

            # x = node_embeddings + temporal_encoding  # Shape: (num_nodes, node_embedding_dim)

            x = node_embed_adaptive + temporal_encoding  # Shape: (num_nodes, node_embedding_dim)

            # print("\n Node embed adaptive mean:", node_embed_adaptive.abs().mean().item())
            # print("Temporal x encoding mean:", x.abs().mean().item())

            # 2. Learnable smoothing of node_pred (shared across scales)
            smoothed_pred = F.softmax(self.smoothing_conv(node_pred.T).T, dim=-1)
            abnormal_scores = smoothed_pred[:, self.normal_class_idx] # note, here node abnormal scores = node confidence;

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

            shared_feature = torch.cat([
                local_feat,
                local_abnormal_feature.unsqueeze(-1),
                centers.unsqueeze(-1) / audio_len.item(),
                widths.unsqueeze(-1) / audio_len.item()
            ], dim=1)


            # Common head for localization logits, interval confidence, classification
            uni_output_logit = unify_head(shared_feature)

            start_logits = uni_output_logit[:, :self.dist_bins[scale_idx]]
            end_logits =uni_output_logit[:, self.dist_bins[scale_idx]:2 * self.dist_bins[scale_idx]]

            interval_conf_logits_scale = uni_output_logit[:, 2 * self.dist_bins[scale_idx]]
            interval_cls_logits_scale = uni_output_logit[:, 2 * self.dist_bins[scale_idx] + 1:]

            # Compute offsets
            start_offsets = (F.softmax(start_logits, dim=1) * weight_params).sum(dim=1)
            end_offsets = (F.softmax(end_logits, dim=1) * weight_params).sum(dim=1)

            # starts = starts + start_offsets
            # ends = ends + end_offsets

            starts = torch.clamp(starts + start_offsets, 0, audio_len.item())
            ends = torch.clamp(ends + end_offsets, 0, audio_len.item())


            # 7. Final bounds and predictions for this scale
            final_bounds_scale = torch.stack([starts, ends], dim=1)


            # Collect results for this scale
            final_bounds_list.append(final_bounds_scale)
            interval_conf_logits_list.append(interval_conf_logits_scale)
            interval_cls_logits_list.append(interval_cls_logits_scale)

            # print(f"Scale {scale_idx} anchor_intervals shape:", anchor_intervals_list[scale_idx].shape)
            # print(f"Scale {scale_idx} local_feat shape:", local_feat.shape)
            # print(f"Scale {scale_idx} conf_logits shape:", interval_conf_logits_scale.shape)
            # print(f"Scale {scale_idx} cls_logits shape:", interval_cls_logits_scale.shape)
        # 8. Concatenate results from all scales
        final_bounds = torch.cat(final_bounds_list, dim=0)
        interval_conf_logits = torch.cat(interval_conf_logits_list, dim=0)
        interval_cls_logits = torch.cat(interval_cls_logits_list, dim=0)

        # print("Final interval_conf_logits shape:", interval_conf_logits.shape)
        # print("Final interval_cls_logits shape:", interval_cls_logits.shape)

        distill_loss_value = 0

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
                 node_num_classes,
                 num_refine_layers = 1,
                 dist_bins_list=[80, 60, 80]
                 ):

        super(Node_Interval_cls_Module, self).__init__()
        self.gat_model = GATModel(in_channels, hidden_channels, out_channels)
        # Fully Connected Layer for Node Classification
        # self.node_classifier = nn.Linear(out_channels, node_num_classes)
        self.node_class_heads =  nn.Sequential(
                nn.Linear(out_channels, 64),  # Hidden layer
                nn.ReLU(),
                nn.Dropout(0.2),  # Regularization
                nn.Linear(64, node_num_classes)  # Output layer
            )



        self.interval_refine = Interval_Refine(
            node_embedding_dim= out_channels,
            num_refine_layers= num_refine_layers,
            dist_bins_list = dist_bins_list
        )

    def forward(self, data, audio_dur, anchor_intervals, fs, frame_hop  ): # pass the batch data
        device = data.x.device

        x = data.x.to(device)
        edge_index = data.edge_index.to(device)

        # Assuming 'data' is your batched graph object
        batch = data.batch
        edge_index = data.edge_index

        # Check if each edge connects nodes from the same sample
        same_batch = batch[edge_index[0]] == batch[edge_index[1]]
        if same_batch.all():

            print("All edges are within the same sample. Your edge_index is correct for batched samples.")
        else:
            print("Some edges connect nodes from different samples. Your edge_index is incorrect for batched samples.")


        edge_attr = data.edge_attr.to(device)
        batch_indices = data.batch.to(device)
        node_labels = data.y.to(device)

        # If edge labels exist
        edge_labels = data.edge_y.to(device) if hasattr(data, 'edge_y') else None
        num_audios = batch_indices.max().item() + 1

        # Compute node embeddings & node predictions  for the entire batch
        node_embeddings, attn_weights = self.gat_model(x, edge_index, edge_attr) # (batch_nodes, node_embeddings )
        node_preds = self.node_class_heads(node_embeddings) # (batch_nodes, normal confidence + 4 abnormal cls)



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
        num_frames_per_audio =  audio_dur * fs / frame_hop  # need to check here;


        for i in range(num_audios):
            mask = (batch_indices == i)

            cur_node_embeddings = node_embeddings[mask]
            cur_node_preds = node_preds[mask]
            cur_node_labels = node_labels[mask]


            cur_duration_ratio = duration_ratios[i]  # Scalar for current audio
            cur_duration = audio_dur[i]
            cur_sample_anchor_intervals = anchor_intervals[i]

            num_frames = int(num_frames_per_audio[i])  # Number of spectrogram frames for this audio

            # ---- 4. Compute Time Positions ----
            # Assume nodes are ordered temporally; normalize time to [0, 1]
            num_nodes = mask.sum().item()
            node_indices = torch.arange(num_nodes, device=device) * 5  # Every 5 frames
            if num_nodes > 1 and node_indices[-1] > num_frames - 5:
                node_indices[-1] = num_frames - 2.5  # Center of last 5 frames
            time_positions = node_indices / num_frames  # Normalize to [0, 1]
            time_positions = time_positions.unsqueeze(-1)  # Shape: (num_nodes, 1)
            # print(f"node_pred shape before time_aware_fdr: {node_preds.shape if node_preds is not None else 'None'}")

            # Note, ----  learn interval Boundaries's  offset parameter ----
            event_output = self.interval_refine( # the instance of class EnhancedTimeAwareFDR
                node_embeddings= cur_node_embeddings ,          #node_embeddings,
                time_positions = time_positions,
                node_pred      = cur_node_preds,
                audio_len      = cur_duration,
                duration_ratio = cur_duration_ratio,  # You need to provide this if you have it
                cur_anchor_intervals = cur_sample_anchor_intervals,
                num_intervals_per_scale = [15, 40, 15] #[39,19, 9] # each scale's  num_anchor_intervals
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



            node_predictions_list.append(cur_node_preds)
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
                 frame_hop= 128,  # Added frame_hop and fs
                 fs=8000,       # Sampling frequency
                 include_gender= False,
                 include_location= False
                 ):
        super(GraphRespiratory, self).__init__()
        self.frame_hop = frame_hop
        self.fs = fs

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



        # In GraphRespiratory or wherever node_fea_generator is defined
        print(f"Conv0 input channels: {self.node_fea_generator.dynamic_cnn.conv0.in_channels}")

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



    def forward(self, batch_data, level):
        device = next(self.parameters()).device

        if level == "node":
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
            all_chunk_type_labels = []
            all_chunk_confidences = []
            batch_indices = []
            num_nodes_per_sample = []
            chunk_times_all = []
            node_gender_all = []
            node_location_all = []

            for sample_idx, (spec, labels, vad_intervals, audio_dur) in enumerate(
                    zip(spectrograms, frame_labels, vad_timestamps, audio_durations)):
                gender = genders[sample_idx]
                location = locations[sample_idx]
                n_frames = spec.shape[1]
                num_nodes = 0

                frame_times = np.arange(n_frames) * self.frame_hop / self.fs
                adjusted_vad_intervals = adjust_vad_timestamps(vad_intervals, frame_times)

                frame_vad_intervals = np.zeros(n_frames, dtype=bool)
                for interval in adjusted_vad_intervals:
                    start = interval["start"]
                    end = interval["end"]
                    in_interval = (frame_times >= start) & (frame_times < end)
                    frame_vad_intervals |= in_interval


                chunk_size = 5
                stride = 5   # 50%  overlap
                chunk_times = []
                for j in range(0, n_frames - chunk_size + 1, stride):
                    chunk = spec[:, j:j + chunk_size, :]
                    all_chunks.append(chunk)
                    label_chunk = labels[j:j + chunk_size, :]

                    # Generate confidence and type labels
                    confidence, type_label = self.get_node_labels(label_chunk, chunk_size)
                    all_chunk_confidences.append(confidence)
                    all_chunk_type_labels.append(type_label)

                    chunk_start_frame = j
                    chunk_end_frame = j + chunk_size - 1
                    chunk_start_time = frame_times[chunk_start_frame]
                    chunk_end_time = frame_times[chunk_end_frame]
                    chunk_times.append((chunk_start_time, chunk_end_time))

                    node_gender_all.append(gender)
                    node_location_all.append(location)
                    num_nodes += 1
                    batch_indices.append(sample_idx)

                # Handle the last chunk if necessary
                last_chunk_end = (n_frames - chunk_size + 1 - stride)
                if  n_frames - chunk_size > last_chunk_end  and  n_frames % stride != 0:
                    chunk = spec[:, n_frames - chunk_size:, :]
                    all_chunks.append(chunk)
                    label_chunk = labels[n_frames - chunk_size:, :]

                    confidence, type_label = self.get_node_labels(label_chunk, chunk_size)
                    all_chunk_confidences.append(confidence)
                    all_chunk_type_labels.append(type_label)

                    chunk_start_frame = n_frames - chunk_size
                    chunk_end_frame = n_frames - 1
                    chunk_start_time = frame_times[chunk_start_frame]
                    chunk_end_time = frame_times[chunk_end_frame]
                    chunk_times.append((chunk_start_time, chunk_end_time))

                    node_gender_all.append(gender)
                    node_location_all.append(location)
                    num_nodes += 1
                    batch_indices.append(sample_idx)

                num_nodes_per_sample.append(num_nodes)
                chunk_times_all.extend(chunk_times)

            if not all_chunks:
                raise ValueError("No valid chunks generated in the batch.")

            all_chunks = [torch.tensor(chunk, dtype=torch.float32) if not isinstance(chunk, torch.Tensor) else chunk for chunk in all_chunks]
            all_chunks_tensor = torch.stack(all_chunks)

            node_features = self.node_fea_generator(all_chunks_tensor)
            node_vad_fea = self.node_vad_proj(node_features)

            node_type_labels = torch.tensor(all_chunk_type_labels, dtype=torch.long, device=node_features.device)
            node_confidences = torch.tensor(all_chunk_confidences, dtype=torch.float, device=node_features.device)

            # valid_mask = node_type_labels >= 0
            # valid_labels = node_type_labels[valid_mask]
            # # Apply bincount with minlength=5
            # a1 = torch.bincount(valid_labels, minlength=5)
            # print(a1)
            #
            # # Create histogram with bins (e.g., 5 bins from 0 to 1)
            # num_bins = 5
            # a2 = torch.histc(node_confidences, bins=num_bins, min=0.0, max=1.0)
            # print(a2)


            batch_indices = torch.tensor(batch_indices, dtype=torch.long, device=node_features.device)

            if self.include_gender:
                node_gender_tensor = torch.tensor(node_gender_all, dtype=torch.long, device=node_features.device)
                node_gender_one_hot = F.one_hot(node_gender_tensor, num_classes=2).float()
                node_features = torch.cat([node_features, node_gender_one_hot], dim=1)

            if self.include_location:
                node_location_tensor = torch.tensor(node_location_all, dtype=torch.long, device=node_features.device)
                node_location_one_hot = F.one_hot(node_location_tensor, num_classes=4).float()
                node_features = torch.cat([node_features, node_location_one_hot], dim=1)

            all_graphs = []
            start_idx = 0
            for num_nodes in num_nodes_per_sample:
                end_idx = start_idx + num_nodes
                sample_node_features = node_features[start_idx:end_idx]
                sample_node_type_labels = node_type_labels[start_idx:end_idx]
                sample_chunk_times = chunk_times_all[start_idx:end_idx]
                sample_node_vad_fea = node_vad_fea[start_idx:end_idx]

                if num_nodes > 1:
                    edge_index = torch.stack([
                        torch.arange(num_nodes - 1, dtype=torch.long, device=node_features.device),
                        torch.arange(1, num_nodes, dtype=torch.long, device=node_features.device)
                    ], dim=0)
                    edge_labels = []
                    for src, dst in zip(edge_index[0], edge_index[1]):
                        src_label = sample_node_type_labels[src]
                        dst_label = sample_node_type_labels[dst]
                        edge_is_abnormal = (src_label != -1 or dst_label != -1)
                        edge_label = 1 if edge_is_abnormal else 0
                        edge_labels.append(edge_label)

                    edge_labels = torch.tensor(edge_labels, dtype=torch.long, device=device)
                    src_node_vad = sample_node_vad_fea[edge_index[0]]
                    dst_node_vad = sample_node_vad_fea[edge_index[1]]
                    combined_edge_features = torch.cat([src_node_vad, dst_node_vad], dim=1)
                    edge_features = self.edge_encoder(combined_edge_features)
                else:
                    edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
                    edge_features = torch.empty((0, 1), dtype=torch.float32, device=device)
                    edge_labels = torch.empty((0,), dtype=torch.long, device=device)

                graph_data = Data(x=sample_node_features, edge_index=edge_index,
                                edge_attr=edge_features, y=sample_node_type_labels)
                graph_data.edge_y = edge_labels
                all_graphs.append(graph_data)
                start_idx = end_idx

            batch_graph = Batch.from_data_list(all_graphs).to(all_chunks_tensor.device)
            pred_output = self.node_interval_pred(batch_graph, audio_durations, anchor_intervals, self.fs, self.frame_hop)

            node_predictions = pred_output[0]
            pred_intervals = pred_output[1]
            pred_interval_conf = pred_output[2]
            pred_interval_cls = pred_output[3]
            distill_loss = pred_output[4]

            # print(f"node_predictions shape: {node_predictions.shape}")  # Should output [sum(N_i), num_classes]

            # Prepare outputs for the node predictions
            outputs = {
                'node_predictions': node_predictions,
                # 'node_labels': node_labels,

                'node_type_labels': node_type_labels,
                'node_confidences': node_confidences,

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



    def get_node_labels(self, frames, chunk_size, normal_label=0):
        """
        Generate confidence and type labels for a chunk of frames.
        Abnormal type labels are mapped from 1-4 to 0-3.

        Args:
            frames (np.ndarray): Shape (5, num_classes), one-hot encoded frame labels.
            normal_label (int): Index for normal frames, default is 0.

        Returns:
            confidence (float): Ratio of abnormal frames to total frames (0 to 1).
            type_label (int): Abnormal type (0-3) if confidence > 0, else -1.
        """

        # Convert frames to NumPy if it's a PyTorch tensor
        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()

        if frames.shape[0] !=  chunk_size:
            raise ValueError(f"Expected {chunk_size}  frames per chunk.")

        frame_labels = frames.argmax(axis=1)  # Shape: (5,), values in {0, 1, 2, 3, 4}
        abnormal_mask = frame_labels != normal_label
        num_abnormal = np.sum(abnormal_mask)
        total_frames = len(frame_labels)
        confidence = num_abnormal / total_frames

        if num_abnormal > 0:
            # Map abnormal frame labels (1-4) to (0-3)
            abnormal_labels = frame_labels[abnormal_mask] - 1  # e.g., [1, 2] -> [0, 1]
            unique, counts = np.unique(abnormal_labels, return_counts=True)
            max_count = np.max(counts)
            candidates = unique[counts == max_count]
            if len(candidates) == 1:
                type_label = int(candidates[0])
            else:
                earliest_idx = total_frames
                selected_label = None
                for label in candidates:
                    # Find first occurrence of the original frame label (label + 1)
                    idx = np.where(frame_labels == (label + 1))[0][0]
                    if idx < earliest_idx:
                        earliest_idx = idx
                        selected_label = label
                type_label = int(selected_label)
        else:
            type_label = -1

        return confidence, type_label

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

