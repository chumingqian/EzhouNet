
from torch_geometric.nn import GATConv, global_mean_pool

from torch import Tensor
from desed_task.nnet.DCNN_v3_4 import  DynamicFeatureExtractor

# Adjusted GATConv that supports edge attributes (your implementation)
import numpy as np




# v1 使用3层 refine_layer，
# 每一层    在前一层生成的 start, end 基础上，进行offset 学习；
# 并且此时， 将每一层输出的 start_logit, end_logit 进行保留， 使用最后一层的进行知识蒸馏；


#  hf_lab1_1:   使用用 hard confidence ,硬置信度，  但是节点分类不参与计算；

# hf_lab1_2:   self.gat  实现并行化计算
# hf_lab1_3:   self.gat, self.anchor_interval  同时实现并行化计算，
# 因为此时在新的数据集中，音频的长度，以及每个样本中节点的个数是相同的；

# hf_lab1_4:   节点分类参与 损失计算；


#  v3-1,  修改先验区间的持续时间为 ， 并且保证尺度2的个数最多，
# Scale 1: 0.5s, Scale 2: 0.8s, Scale 3: 1.5s
# 对于粗粒度的 区间，同样分配更多的bins，  从而可以进行更大范围的偏移学习；
# 即较粗的间隔需要更多的偏移容量才能有效地细化。


#  v3-1,



import torch

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
class Batch_Interval_Refine(nn.Module):
    def __init__(self, node_embedding_dim, num_classes=5, num_refine_layers=1,
                 dist_bins_list= None, kernel_size=5 ):
        #super().__init__()
        super(Batch_Interval_Refine, self).__init__()
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
            self,   # Now is batch sample node information;  not a single sample
            node_embeddings: torch.Tensor,  # [num_audios, num_nodes, emb_dim]
            time_positions: torch.Tensor,  # [num_audios, num_nodes, 1]
            node_pred: torch.Tensor,  # [num_audios, num_nodes, num_classes]
            audio_len: torch.Tensor,  # scalar

            cur_anchor_intervals: torch.Tensor,  # [num_audios, num_intervals, 2]
            distill_loss: bool = True,
            duration_ratio: float = None,
            num_intervals_per_scale: list = None

    ):
        """
        # Inputs:
        # - node_embeddings: [num_audios, num_nodes, emb_dim]
        # - time_positions: [num_audios, num_nodes, 1]
        # - node_pred: [num_audios, num_nodes, num_classes]
        # - audio_len: scalar
        # - duration_ratio: scalar
        # - cur_anchor_intervals: [num_audios, num_intervals, 2]
        # - num_intervals_per_scale: list of ints (e.g., [39, 19, 9])

        Returns a dict with:
          - "final_bounds": (max_intervals, 2) refined [start, end] in [0,1].
          - "interval_cls_logits": (max_intervals, num_interval_classes) classification logits.
          - "num_intervals": number of intervals (always max_intervals here).
        """
        device = node_embeddings.device
        num_audios, num_nodes, _ = node_embeddings.shape
        batch_size, num_nodes, node_embedding_dim = node_embeddings.shape  # (batch_size, 94, node_embedding_dim)

        print(f"node_embeddings shape: {node_embeddings.shape}")
        print(f"time_positions shape: {time_positions.shape}")

        # note,  debug here
        if time_positions.shape[1] != num_nodes:
            if time_positions.shape[1] < num_nodes:
                padding = torch.zeros(batch_size, num_nodes - time_positions.shape[1], 1, device=device)
                time_positions = torch.cat([time_positions, padding], dim=1)
            else:
                time_positions = time_positions[:, :num_nodes, :]


        if not torch.is_tensor(cur_anchor_intervals):
            cur_anchor_intervals = torch.tensor(cur_anchor_intervals, device=device)
        assert sum(num_intervals_per_scale) == cur_anchor_intervals.size(1), f"Interval count mismatch: {sum(num_intervals_per_scale)} vs {cur_anchor_intervals.size(1)}"

        if num_intervals_per_scale is None:
            raise ValueError("num_intervals_per_scale must be provided as a list of integers [n1, n2, n3]")

         # Split anchor intervals by scale,  into  three scale;
        anchor_intervals_list = torch.split(cur_anchor_intervals, num_intervals_per_scale, dim=1)

        # Lists to collect results from all scales
        final_bounds_list = []
        interval_conf_logits_list = []
        interval_cls_logits_list = []


        #node_embed_adaptive =  self.node_embed_adaptive(node_embeddings)

        # Process each scale independently
        for scale_idx in range(self.num_scales):# [num_audios, num_intervals_scale, 2]
            anchor_intervals = anchor_intervals_list[scale_idx]
            num_intervals_scale = num_intervals_per_scale[scale_idx]

            if num_intervals_scale == 0:
                continue

            # Select scale-specific components
            local_feat_rnn = self.local_feat_rnns[scale_idx]
            local_abnormal_rnn = self.local_abnormal_rnns[scale_idx]

            unify_head = self.interval_unify_head[scale_idx]
            weight_params = self.weight_params[scale_idx]


            # Batched temporal encoding
            scale_factor = 0.05
            freqs = torch.linspace(0, 10, self.node_embedding_dim, device=device).view(1, 1,-1)  # (1, 1, node_embedding_dim)
            temporal_encoding = scale_factor * torch.sin(time_positions * freqs)  # (batch_size, num_nodes, node_embedding_dim)

            x = node_embeddings + temporal_encoding  # (batch_size, num_nodes, node_embedding_dim)


            # 2. Learnable smoothing of node_pred
            node_pred_permuted = node_pred.permute(0, 2, 1)  # [num_audios, num_classes, num_nodes]
            smoothed_pred_permuted = self.smoothing_conv(node_pred_permuted)
            smoothed_pred = smoothed_pred_permuted.permute(0, 2, 1)  # [num_audios, num_nodes, num_classes]

            smoothed_pred = F.softmax(smoothed_pred, dim=-1)
            abnormal_scores = smoothed_pred[:, :, self.normal_class_idx]  # [num_audios, num_nodes]

            # 3. Compute masks for this scale's intervals
            starts = anchor_intervals[:, :, 0].unsqueeze(2)  # (batch_size, num_intervals_scale, 1)
            ends = anchor_intervals[:, :, 1].unsqueeze(2)  # (batch_size, num_intervals_scale, 1)
            #time_pos_1d = time_positions.squeeze(-1).unsqueeze(1)  # [num_audios, 1, num_nodes]


            time_pos_1d = time_positions.squeeze(-1).unsqueeze(1) * audio_len  # (batch_size, 1, num_nodes)
            in_interval_mask = (time_pos_1d >= starts) & (
                        time_pos_1d <= ends)  # [num_audios, num_intervals_scale, num_nodes]



            # 4. Extract local features for this scale
            # Extract local features
            local_x_list = []
            local_abnormal_list = []
            sequence_lengths = []
            for b in range(batch_size):
                for i in range(num_intervals_scale):
                    mask = in_interval_mask[b, i]  # (num_nodes)
                    if mask.sum() > 0:
                        local_x = x[b, mask]  # (seq_len, node_embedding_dim)
                        local_abnormal = abnormal_scores[b, mask].unsqueeze(-1)  # (seq_len, 1)
                    else:
                        local_x = torch.zeros((1, self.node_embedding_dim), device=device)
                        local_abnormal = torch.zeros((1, 1), device=device)
                    local_x_list.append(local_x)
                    local_abnormal_list.append(local_abnormal)
                    sequence_lengths.append(local_x.size(0))

            # Sort by sequence length for packing
            # Pad and pack sequences
            sorted_indices = sorted(range(len(sequence_lengths)), key=lambda k: sequence_lengths[k],  reverse=True)
            sorted_local_x = [local_x_list[k] for k in sorted_indices]
            sorted_local_abnormal = [local_abnormal_list[k] for k in sorted_indices]
            sorted_lengths = [sequence_lengths[k] for k in sorted_indices]

            max_len = max(sorted_lengths)
            padded_local_x = torch.stack(
                [F.pad(seq, (0, 0, 0, max_len - seq.size(0))) for seq in sorted_local_x], dim=0)
            padded_local_abnormal = torch.stack(
                [F.pad(seq, (0, 0, 0, max_len - seq.size(0))) for seq in sorted_local_abnormal], dim=0)
            packed_local_x = pack_padded_sequence(padded_local_x, lengths=sorted_lengths, batch_first=True)
            packed_local_abnormal = pack_padded_sequence(padded_local_abnormal, lengths=sorted_lengths,
                                                         batch_first=True)
            # 5. Process through scale-specific GRUs
            _, h_local = local_feat_rnn(packed_local_x)
            _, h_local_abnormal = local_abnormal_rnn(packed_local_abnormal)

            # Unsort to original order
            unsort_indices = torch.argsort(torch.tensor(sorted_indices, device=device))
            local_feat = h_local.squeeze(0)[unsort_indices]  # [total_intervals, hidden_dim]
            local_abnormal_feature = h_local_abnormal.squeeze(0).squeeze(-1)[ unsort_indices]  # [total_intervals]

            # Reshape to [num_audios, num_intervals_scale, ...]
            total_intervals = num_audios * num_intervals_scale
            local_feat = local_feat.view(num_audios, num_intervals_scale, -1)
            local_abnormal_feature = local_abnormal_feature.view(num_audios, num_intervals_scale)

            # 6. Refine intervals for this scale
            starts = anchor_intervals[:, :, 0].clone()  # [num_audios, num_intervals_scale]
            ends = anchor_intervals[:, :, 1].clone()  # [num_audios, num_intervals_scale]

            centers = (starts + ends) / 2  # (batch_size, num_intervals_scale)
            widths = ends - starts  # (batch_size, num_intervals_scale)


            shared_feature = torch.cat([
                local_feat,
                local_abnormal_feature.unsqueeze(-1),
                centers.unsqueeze(-1) / audio_len.item(),
                widths.unsqueeze(-1) / audio_len.item()
            ], dim=2)  # (batch_size, num_intervals_scale, node_embedding_dim + 3)

            # Common head for localization logits, interval confidence, classification
            uni_output_logit = unify_head(shared_feature) # (batch_size, num_intervals_scale, output_dim)

            start_logits = uni_output_logit[:, :, :self.dist_bins[scale_idx]]
            end_logits = uni_output_logit[:, :, self.dist_bins[scale_idx]:2 * self.dist_bins[scale_idx]]
            interval_conf_logits_scale = uni_output_logit[:, :, 2 * self.dist_bins[scale_idx]]
            interval_cls_logits_scale = uni_output_logit[:, :, 2 * self.dist_bins[scale_idx] + 1:]
            # Compute offsets

            # Compute offsets
            start_offsets = (F.softmax(start_logits, dim=-1) * weight_params).sum(
                dim=-1)  # (batch_size, num_intervals_scale)
            end_offsets = (F.softmax(end_logits, dim=-1) * weight_params).sum(
                dim=-1)  # (batch_size, num_intervals_scale)
            starts = torch.clamp(starts + start_offsets, 0, audio_len)
            ends = torch.clamp(ends + end_offsets, 0, audio_len)

            # 7. Final bounds and predictions for this scale
            final_bounds_scale = torch.stack([starts, ends], dim=-1)  # [num_audios, num_intervals_scale, 2]

            # Collect results
            final_bounds_list.append(final_bounds_scale)
            interval_conf_logits_list.append(interval_conf_logits_scale)
            interval_cls_logits_list.append(interval_cls_logits_scale)



        # 8. Concatenate results from all scales
        final_bounds = torch.cat(final_bounds_list, dim=1)  # [num_audios, total_intervals, 2]
        interval_conf_logits = torch.cat(interval_conf_logits_list, dim=1)  # [num_audios, total_intervals, 1]
        interval_cls_logits = torch.cat(interval_cls_logits_list,
                                        dim=1)  # [num_audios, total_intervals, num_classes - 1]

        distill_loss_value =0
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
                 dist_bins_list=[80, 80, 80]
                 ):

        super(Node_Interval_cls_Module, self).__init__()
        self.gat_model = GATModel(in_channels, hidden_channels, out_channels)
        # Fully Connected Layer for Node Classification
        self.node_class_heads =  nn.Sequential(
                nn.Linear(out_channels, 64),  # Hidden layer
                nn.ReLU(),
                nn.Dropout(0.2),  # Regularization
                nn.Linear(64, node_num_classes)  # Output layer
            )

        self.interval_refine = Batch_Interval_Refine(
            node_embedding_dim= out_channels,
            num_refine_layers= num_refine_layers,
            dist_bins_list = dist_bins_list
        )


    def forward(self, data, audio_dur, anchor_intervals, fs, frame_hop   ): # pass the batch data
        device = data.x.device

        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        edge_attr = data.edge_attr.to(device)         # If edge labels exist
        edge_labels = data.edge_y.to(device) if hasattr(data, 'edge_y') else None


        # Assuming 'data' is your batched graph object
        batch = data.batch
        edge_index = data.edge_index

        # Check if each edge connects nodes from the same sample
        same_batch = batch[edge_index[0]] == batch[edge_index[1]]
        # if same_batch.all():
        #
        #     print("All edges are within the same sample. Your edge_index is correct for batched samples.")
        # else:
        #     print("Some edges connect nodes from different samples. Your edge_index is incorrect for batched samples.")


        edge_attr = data.edge_attr.to(device)
        batch_indices = data.batch.to(device)
        node_labels = data.y.to(device)
        num_audios = batch_indices.max().item() + 1


        # Compute node embeddings for the entire batch
        node_embeddings, attn_weights = self.gat_model(x, edge_index, edge_attr)
        node_preds = self.node_class_heads(node_embeddings)

        # Uniform audio duration (assuming all are the same)
        audio_dur = torch.tensor(audio_dur, dtype=torch.float, device=device)
        num_frames = int((audio_dur[0] * fs / frame_hop).item())  # Same for all audios

        # Calculate number of nodes per audio (assuming fixed chunk size of 5 frames)
        chunk_size = 5
        num_nodes_per_audio = (num_frames - chunk_size) // chunk_size + 1  # 93

        # Generate time_positions for one audio and replicate across batch
        node_indices = torch.arange(num_nodes_per_audio,
                                    device=device) * chunk_size + chunk_size / 2  # Center of each chunk
        time_positions_single = node_indices / num_frames  #shape: (num_nodes_per_audio =93,),   Normalize to [0, 1],
        time_positions_batch = time_positions_single.unsqueeze(0).expand(num_audios, -1).unsqueeze(
            -1)  # Shape: (num_audios, num_nodes_per_audio, 1)


        # Reshape to include batch dimension (same node count per sample)
        num_nodes_per_sample = node_embeddings.size(0) // num_audios # num nodes = 94;
        node_embeddings = node_embeddings.view(num_audios, num_nodes_per_sample, -1)
        node_preds = node_preds.view(num_audios, num_nodes_per_sample, -1)

        #   note, here  is the  root reason  of that  can not parallsize ;
        #  althoug each audio has the same  length,
        #  but it actully  has  different frames,  and this wil generate differ number nodes;
        # # 188;188;188;188; 187; 187; 188;188;187;
        # but,  i can  allocate the  each audio frames before  parallize it ,
        # how  should  i deal  with that;



        # Fixed audio length (all are 15s)
        audio_len = audio_dur[0]  # Scalar, same for all

        # Batch anchor intervals (assuming consistent number of intervals)
        anchor_intervals_tensor = torch.stack([torch.tensor(ai, device=device) for ai in anchor_intervals])
        num_intervals_per_scale = [15, 40, 15]  #[39, 19, 9]  # Example scales

        # Call interval_refine with batched inputs
        event_output = self.interval_refine(
            node_embeddings=node_embeddings,  # [num_audios, num_nodes, emb_dim]
            time_positions=time_positions_batch,  # [num_audios, num_nodes, 1]
            node_pred=node_preds,  # [num_audios, num_nodes, num_classes]
            audio_len=audio_len,  # Scalar
            duration_ratio=1.0,  # Same duration across samples
            cur_anchor_intervals=anchor_intervals_tensor,  # [num_audios, num_intervals, 2]
            num_intervals_per_scale=num_intervals_per_scale
        )

        # Prepare outputs
        node_predictions = node_preds.view(-1, node_preds.size(-1))
        node_labels = node_labels.view(-1)





        return [node_predictions,
                event_output["final_bounds"],  # [num_audios, num_intervals, 2]
                event_output["interval_conf_logits"],  # [num_audios, num_intervals, 1]
                event_output["interval_cls_logits"],  # [num_audios, num_intervals, num_classes - 1]
                event_output["distill_loss"]  # Scalar or per-sample
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
                                             nn.Linear(128, 32)
                                                )


        # Edge feature encoder: from raw edge_attr (scalar) to learnable embedding
        edge_feature_dim = 16 # if you change this , the GAT layer,edge_dim=2 should also be changed ;
        self.edge_encoder = nn.Linear(32*2, edge_feature_dim)
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
        #genders = batch_data['genders']  # List or tensor of length batch_size
        locations = batch_data['chest_loc']  # List or tensor of length batch_size

        audio_durations = batch_data['audio_dur']
        anchor_intervals = batch_data["anchor_intervals"]

        # audio_durations = batch_data['audio_durations']


        # Initialize lists
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
            stride = 5  # 50%  overlap
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

                # node_gender_all.append(gender)
                node_location_all.append(location)
                num_nodes += 1
                batch_indices.append(sample_idx)

            # Handle the last chunk if necessary
            last_chunk_end = (n_frames - chunk_size + 1 - stride)
            if n_frames - chunk_size > last_chunk_end and n_frames % stride != 0:
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

                # node_gender_all.append(gender)
                node_location_all.append(location)
                num_nodes += 1
                batch_indices.append(sample_idx)

            num_nodes_per_sample.append(num_nodes)
            chunk_times_all.extend(chunk_times)

        if not all_chunks:
            raise ValueError("No valid chunks generated in the batch.")

        all_chunks = [torch.tensor(chunk, dtype=torch.float32) if not isinstance(chunk, torch.Tensor) else chunk for
                      chunk in all_chunks]
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


        # Batch all graphs together,
        batch_graph = Batch.from_data_list(all_graphs).to(all_chunks_tensor.device)

        # ------------------- batch Node-  Classification-----------------
        pred_output = self.node_interval_pred(batch_graph, audio_durations, anchor_intervals,self.fs, self.frame_hop)

        node_predictions =  pred_output[0]
        pred_intervals = pred_output[1]
        pred_interval_conf = pred_output[2]
        pred_interval_cls = pred_output[3]

        distill_loss = pred_output[4]

        # print(f"node_predictions shape: {node_predictions.shape}")  # Should output [sum(N_i), num_classes]

        # Prepare outputs for the node predictions
        outputs = {
            'node_predictions': node_predictions,

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

