
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

# hf_lab3_9:  调整为双向 Bigru,  从 gru 中 获取 logit,
# 然后与独立的 weight param 相乘获得 start, end offsets;


import torch

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
class Interval_Refine(nn.Module):
    def __init__(self, node_embedding_dim, num_classes=5, num_refine_layers=1,
                 dist_bins_list=None, kernel_size=5):
        super(Interval_Refine, self).__init__()
        self.node_embedding_dim = node_embedding_dim
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.num_refine_layers = num_refine_layers
        self.dist_bins = dist_bins_list
        self.num_scales = 3  # Three scales
        self.normal_class_idx = 0

        # Learnable Gaussian smoothing
        self.smoothing_conv = nn.Conv1d(
            in_channels=num_classes,
            out_channels=num_classes,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=num_classes,
            bias=False
        )
        self._init_gaussian_weights()

        # Scale-specific local feature BiGRUs,  note  here, change to bigru;
        self.local_feat_rnns = nn.ModuleList([
            nn.GRU(input_size=node_embedding_dim, hidden_size=node_embedding_dim, batch_first=True, bidirectional=True)
            for _ in range(self.num_scales)
        ])

        # Scale-specific local abnormal BiGRUs
        self.local_abnormal_rnns = nn.ModuleList([
            nn.GRU(input_size=1, hidden_size=1, batch_first=True, bidirectional=True)
            for _ in range(self.num_scales)
        ])

        # Separate weight parameters for start and end offsets
        self.start_weight_params = nn.ParameterList([
            nn.Parameter(torch.linspace(-20.0, 20.0, dist_bins_list[i]))
            for i in range(self.num_scales)
        ])
        self.end_weight_params = nn.ParameterList([
            nn.Parameter(torch.linspace(-20.0, 20.0, dist_bins_list[i]))
            for i in range(self.num_scales)
        ])
        #


        # self.start_weight_params = nn.ParameterList([
        #     nn.Parameter(torch.linspace(-0.50, 0.50, dist_bins_list[i]))
        #     for i in range(self.num_scales)
        # ])
        # self.end_weight_params = nn.ParameterList([
        #     nn.Parameter(torch.linspace(-0.50, 0.50, dist_bins_list[i]))
        #     for i in range(self.num_scales)
        # ])


        # Heads for start and end logits
        # self.start_heads = nn.ModuleList([
        #     nn.Linear(node_embedding_dim, dist_bins_list[i])
        #     for i in range(self.num_scales)
        # ])
        # self.end_heads = nn.ModuleList([
        #     nn.Linear(node_embedding_dim, dist_bins_list[i])
        #     for i in range(self.num_scales)
        # ])
        #
        # # Heads for confidence and classification
        # self.conf_cls_heads = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(2 * node_embedding_dim + 3, 256),
        #         nn.ReLU(),
        #         nn.Dropout(0.2),
        #         nn.Linear(256, 64),
        #         nn.ReLU(),
        #         nn.Dropout(0.2),
        #         nn.Linear(64, 1 + (num_classes - 1))
        #     )
        #     for i in range(self.num_scales)
        # ])



        # Unified head for all predictions
        self.unify_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * node_embedding_dim + 3, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 2 * dist_bins_list[i] + 1 + (num_classes - 1))
            )
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

    def forward(   # change  back  to  process  one  sample
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
        assert sum(num_intervals_per_scale) == cur_anchor_intervals.size(0), "Interval count mismatch"

        split_indices = torch.cumsum(torch.tensor(num_intervals_per_scale, device=device), dim=0)
        anchor_intervals_list = torch.split(cur_anchor_intervals, num_intervals_per_scale, dim=0)

        final_bounds_list = []
        interval_conf_logits_list = []
        interval_cls_logits_list = []

        for scale_idx in range(self.num_scales):
            anchor_intervals = anchor_intervals_list[scale_idx]
            num_intervals_scale = num_intervals_per_scale[scale_idx]
            if num_intervals_scale == 0:
                continue

            local_feat_rnn = self.local_feat_rnns[scale_idx]
            local_abnormal_rnn = self.local_abnormal_rnns[scale_idx]


            start_weight_params = self.start_weight_params[scale_idx]
            end_weight_params = self.end_weight_params[scale_idx]

            # Temporal encoding
            scale_factor = 0.05
            freqs = torch.linspace(0, 10, self.node_embedding_dim, device=device)
            temporal_encoding = scale_factor * torch.sin(time_positions * freqs)
            x = node_embeddings + temporal_encoding

            # Smoothed predictions
            smoothed_pred = F.softmax(self.smoothing_conv(node_pred.T).T, dim=-1)
            abnormal_scores = smoothed_pred[:, self.normal_class_idx]

            # Compute masks
            time_pos_1d = time_positions.squeeze(-1) * audio_len.item()
            starts = anchor_intervals[:, 0].clone()
            ends = anchor_intervals[:, 1].clone()

            in_interval_mask = (time_pos_1d.unsqueeze(0) >= starts.unsqueeze(1)) & \
                               (time_pos_1d.unsqueeze(0) <= ends.unsqueeze(1))

            # Extract local sequences
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

            # Sort by sequence length
            sorted_indices = sorted(range(num_intervals_scale), key=lambda i: sequence_lengths[i], reverse=True)
            sorted_local_x = [local_x_list[i] for i in sorted_indices]
            sorted_local_abnormal = [local_abnormal_seq_list[i] for i in sorted_indices]
            sorted_lengths = [sequence_lengths[i] for i in sorted_indices]

            # Pad and pack
            max_len = max(sorted_lengths)
            padded_local_x = torch.stack([F.pad(seq, (0, 0, 0, max_len - seq.size(0))) for seq in sorted_local_x], dim=0)
            padded_local_abnormal = torch.stack([F.pad(seq, (0, 0, 0, max_len - seq.size(0))) for seq in sorted_local_abnormal], dim=0)
            packed_local_x = pack_padded_sequence(padded_local_x, lengths=sorted_lengths, batch_first=True)
            packed_local_abnormal = pack_padded_sequence(padded_local_abnormal, lengths=sorted_lengths, batch_first=True)

            # Process through BiGRUs
            _, h_local = local_feat_rnn(packed_local_x)  # h_local: (2, batch, hidden_size)
            _, h_local_abnormal = local_abnormal_rnn(packed_local_abnormal)  # h_local_abnormal: (2, batch, 1)

            # Unsort to original order
            unsort_indices = torch.argsort(torch.tensor(sorted_indices, device=device))
            h_local = h_local[:, unsort_indices, :]
            h_local_abnormal = h_local_abnormal[:, unsort_indices, :]

            # Extract forward and backward hidden states
            h_forward = h_local[0, :, :]  # Forward final hidden state
            h_backward = h_local[1, :, :]  # Backward final hidden state

            # Confidence and classification
            local_feat = torch.cat([h_forward, h_backward], dim=1)  # (batch, 2*hidden_size)
            local_abnormal_feature = (h_local_abnormal[0, :, 0] + h_local_abnormal[1, :, 0]) / 2  # Average of directions

            centers = (starts + ends) / 2
            widths = ends - starts



            shared_feature = torch.cat([
                local_feat,
                local_abnormal_feature.unsqueeze(-1),
                centers.unsqueeze(-1) / audio_len.item(),
                widths.unsqueeze(-1) / audio_len.item()
            ], dim=1)

            # Unified prediction
            unify_head = self.unify_heads[scale_idx]
            unify_output = unify_head(shared_feature)


            start_logits = unify_output[:, :self.dist_bins[scale_idx]]
            end_logits = unify_output[:, self.dist_bins[scale_idx]:2 * self.dist_bins[scale_idx]]

            interval_conf_logits_scale = unify_output[:, 2 * self.dist_bins[scale_idx]]
            interval_cls_logits_scale = unify_output[:, 2 * self.dist_bins[scale_idx] + 1:]

            # Compute offsets
            start_offsets = (F.softmax(start_logits, dim=1) * start_weight_params).sum(dim=1)
            end_offsets = (F.softmax(end_logits, dim=1) * end_weight_params).sum(dim=1)

            starts = torch.clamp(starts + start_offsets, 0, audio_len.item())
            ends = torch.clamp(ends + end_offsets, 0, audio_len.item())


            # Collect results
            final_bounds_scale = torch.stack([starts, ends], dim=1)
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
                 dist_bins_list=[80, 60, 40]
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

        self.interval_refine = Interval_Refine(
            node_embedding_dim= out_channels,
            num_refine_layers= num_refine_layers,
            dist_bins_list = dist_bins_list
        )


    def forward(self, data, audio_dur, anchor_intervals, fs, frame_hop, chunk_size, stride ): # pass the batch data
        device = data.x.device

        x = data.x.to(device)


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
        node_embeddings, attn_weights = self.gat_model(x, edge_index, edge_attr) # node embeddings (2253, 256  )
        node_preds = self.node_class_heads(node_embeddings)  # 2253



        node_predictions_list = []
        all_node_labels = []

        intervals_list = []
        bt_distill_loss = []

        interval_conf_list  = []
        interval_cls_list   = []



        # Uniform audio duration (assuming all are the same)
        audio_dur = torch.tensor(audio_dur, dtype=torch.float, device=device)
        mean_duration = audio_dur.mean()  # Mean of all audio durations in the batch
        duration_ratios = audio_dur / mean_duration  # Shape: [num_audios]
        num_frames_per_audio =  audio_dur * fs / frame_hop  # need to check here; 12

        #   note, here  is the  root reason  of that  can not parallsize ;
        #  althoug each audio has the same  length,
        #  but it actully  has  different frames,  and this wil generate differ number nodes;
        # # 188;188;188;188; 187; 187; 188;188;187;
        # but,  i can  allocate the  each audio frames before  parallize it ,
        # how  should  i deal  with that;



        for i in range(num_audios):
            mask = (batch_indices == i)

            cur_node_embeddings = node_embeddings[mask]
            cur_node_preds = node_preds[mask]
            cur_node_labels = node_labels[mask]

            cur_duration_ratio = duration_ratios[i]  # Scalar for current audio
            cur_duration = audio_dur[i]
            cur_sample_anchor_intervals = anchor_intervals[i]

            num_frames = int(num_frames_per_audio[i])  # 937 Number of spectrogram frames for this audio

            # ---- 4. Compute Time Positions ----
            # Assume nodes are ordered temporally; normalize time to [0, 1]
            num_nodes = mask.sum().item() # 188;188;188;188; 187; 187; 188;188;187;

            #node_indices = torch.arange(num_nodes, device=device) * 5  # Every 5 frames
            # if num_nodes > 1 and node_indices[-1] > num_frames - 5:
            #     node_indices[-1] = num_frames - 2.5  # Center of last 5 frames
            # time_positions = node_indices / num_frames  # num nodes = 193,  Normalize to [0, 1]
            #
            # chunk_size, stride = 25, 5
            chunk_centers = torch.arange(num_nodes, device=device) * stride + (chunk_size / 2)
            time_positions = chunk_centers / num_frames

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
                num_intervals_per_scale = [39,19, 9]  # [15, 40, 15] #[39,19, 9] # each scale's  num_anchor_intervals
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
                 include_location=  False  # True  # note, only use Hf data put on  the triger, if steth_scope no posi Information;
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


        # self.node_class_heads =  nn.Sequential(
        #         nn.Linear(adjusted_in_channels, 256),  # Hidden layer
        #         nn.ReLU(),
        #         nn.Dropout(0.2),
        #         nn.Linear(256, 128),
        #         nn.ReLU(),
        #         nn.Dropout(0.2),
        #         nn.Linear(128, num_classes)  # Output layer
        #     )

        self.node_edge_proj =  nn.Sequential( nn.Linear(node_fea_dim, 128),
                                            # Example hidden layer
                                             nn.ReLU(),
                                             nn.Dropout(0.5),
                                             nn.Linear(128, 2)
                                                )


        # Edge feature encoder: from raw edge_attr (scalar) to learnable embedding
        edge_feature_dim = 16 # if you change this , the GAT layer,edge_dim=2 should also be changed ;
        self.edge_encoder = nn.Linear( 2 *2, edge_feature_dim)
        self.node_interval_pred = Node_Interval_cls_Module(node_fea_dim, hidden_channels, out_channels, node_num_classes=num_classes)

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

        chunk_size = 5
        stride = 5  # 0%  overlap

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
        
        
        if self.include_gender:
            node_gender_tensor = torch.tensor(node_gender_all, dtype=torch.long, device=node_features.device)
            node_gender_one_hot = F.one_hot(node_gender_tensor, num_classes=2).float()
            node_features = torch.cat([node_features, node_gender_one_hot], dim=1)

        if self.include_location:
            node_location_all = [t.item() for t in node_location_all]
            node_location_tensor = torch.tensor(node_location_all, dtype=torch.long, device=node_features.device)
            node_location_one_hot = F.one_hot(node_location_tensor, num_classes=4).float()
            posi_fea = self.position_linear(node_location_one_hot)
            node_features = torch.cat([node_features, posi_fea], dim=1)

        # node_edge_fea = self.node_edge_proj(node_features)
        # node_preds = self.node_class_heads(node_features)  # (batch_nodes, normal confidence + 4 abnormal cls)
        # # directly use node edge fea as the node preds
        # node_edge_fea = node_preds

        node_vad_fea = self.node_edge_proj(node_features)

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



        all_graphs = []
        start_idx = 0
        for num_nodes in num_nodes_per_sample:
            end_idx = start_idx + num_nodes
            sample_node_features = node_features[start_idx:end_idx]
            sample_node_type_labels = node_type_labels[start_idx:end_idx]
            sample_chunk_times = chunk_times_all[start_idx:end_idx]
            sample_node_edge_fea = node_vad_fea[start_idx:end_idx]

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
                src_node_vad = sample_node_edge_fea[edge_index[0]]
                dst_node_vad = sample_node_edge_fea[edge_index[1]]
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
        # pred_output = self.node_interval_pred(batch_graph, audio_durations, anchor_intervals,  node_preds, self.fs, self.frame_hop)
        pred_output = self.node_interval_pred(batch_graph, audio_durations, anchor_intervals, self.fs, self.frame_hop,
                                              chunk_size, stride)

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

