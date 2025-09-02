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
from desed_task.nnet.DCNN_v3_3 import  DynamicFeatureExtractor

# Adjusted GATConv that supports edge attributes (your implementation)
import numpy as np


# v6-5-4, 将vad 的时间戳信息编码到 node 的特征中，以及图的边缘属性中；
# v6-5-5, 增加采集时的胸腔位置信息，以及性别信息；；

# v7-1-1,  开始使用边的预测信息
# v7-5-4,  引入门控机制；

# V7-5-5,   引入门控机制，  但是只保留门控的特征， 作为后续图网络输入的特征；
# V7-5-6,   引入门控机制， 将门控特征与原始的特征，使用残差的思想进行相加， 后续图网络输入的特征；

# V7-5-7,    使用可学习的缩放形式， 将两者进行拼接；
# v8-1,   开始引入 d fine 的思想， 对定位的时间损失进行计算。


import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeAwareFDR(nn.Module):
    def __init__(self, node_embedding_dim, max_intervals=5, num_refine_layers=3, dist_bins=100):
        super().__init__()
        self.node_embedding_dim = node_embedding_dim
        self.max_intervals = max_intervals
        self.num_refine_layers = num_refine_layers
        self.dist_bins = dist_bins

        # Time embedding layer
        self.time_embed = nn.Linear(1, node_embedding_dim)

        # Cross-node attention
        self.attn = nn.MultiheadAttention(node_embedding_dim, num_heads=4, batch_first=True)

        # Interval count predictor
        self.interval_counter = nn.Sequential(
            nn.Linear(node_embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, max_intervals),
            nn.Softmax(dim=-1)
        )

        # Boundary refinement layers
        self.refinement_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(node_embedding_dim + 2, 128),  # +2 for current start/end
                nn.ReLU(),
                nn.Linear(128, 2 * dist_bins)
            )
            for _ in range(num_refine_layers)

        ])

        # Learnable offset bins
        self.weight_params = nn.Parameter(torch.linspace(-0.5, 0.5, dist_bins))

        # Query projection layer to fix dimension mismatch
        self.query_proj = nn.Linear(node_embedding_dim + 2, node_embedding_dim)

    def forward(self, node_embeddings, time_positions, batch_indices):
        # Time embedding
        time_features = self.time_embed(time_positions)
        x = node_embeddings + time_features  # (num_nodes, emb_dim) = node_fea + time fea

        # Cross-node attention
        attn_output, _ = self.attn(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
        attn_output = attn_output.squeeze(0)  # (num_nodes, emb_dim)

        # Predict number of intervals， 在节点维度上进行平均；
        pooled = attn_output.mean(dim=0)  # (emb_dim)
        interval_probs = self.interval_counter(pooled)  # (max_intervals)， 为什么这里设置了各个数量的概率和为1；
        num_intervals = interval_probs.argmax() + 1


        # Mask invalid intervals,# Compute interval mask once
        interval_mask = torch.arange(self.max_intervals, device=x.device) < num_intervals


        # Initialize boundaries based on time positions
        t_min = time_positions.min()
        t_max = time_positions.max()
        intervals = torch.linspace(t_min, t_max, self.max_intervals + 1, device=x.device)
        # init_bounds = torch.rand(self.max_intervals, 2, device=x.device)  # Placeholder， 随机初始化5个边界框，属于0-1 范围内；

        init_bounds = torch.stack([intervals[:-1], intervals[1:]], dim=-1)  # (max_intervals, 2)
        current_start = init_bounds[:, 0]
        current_end = init_bounds[:, 1]

        # Iterative refinement
        for layer in self.refinement_layers:
            for i in range(self.max_intervals):
                # Concatenate current boundaries with node features
                query = torch.cat([
                    attn_output, # (num_nodes, embed_dim),
                    current_start[i].expand(attn_output.size(0), 1), #(num_nodes, 1)
                    current_end[i].expand(attn_output.size(0), 1)  # (num_nodes, 1)
                ], dim=-1)  # query: (num_nodes, embed_dim + 2)


                # Project query back to original embedding dimension
                query_projected = self.query_proj(query)  # (num_nodes, emb_dim)

                # Predict offsets
                logits = layer(query)  # (num_nodes, 2 * dist_bins)
                start_logits, end_logits = logits.chunk(2, dim=-1)

                # Compute offsets
                start_offset = (F.softmax(start_logits, dim=-1) * self.weight_params).sum(-1)
                end_offset = (F.softmax(end_logits, dim=-1) * self.weight_params).sum(-1)

                # Aggregate using attention weights
                _, attn_weights = self.attn(
                    query_projected.unsqueeze(0), # Projected to correct dimension
                    attn_output.unsqueeze(0),
                    attn_output.unsqueeze(0) )

                current_start[i] += (attn_weights.squeeze() * start_offset).sum()
                current_end[i] += (attn_weights.squeeze() * end_offset).sum()


        final_bounds = torch.stack([current_start, current_end], dim=-1)

        return {
            "final_bounds": final_bounds,
            "interval_mask": interval_mask
        }


class DynamicTimestampGenerator(nn.Module):
    def __init__(self, input_dim,num_classes, max_events=5, num_bins=100):
        super().__init__()
        self.max_events = max_events
        self.num_bins = num_bins

        # Temporal Context Encoder
        self.context_encoder = nn.Sequential(
            nn.Conv1d(input_dim, 256, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(256, 128, 5, padding=2),
            nn.ReLU()
        )

        # Dynamic Event Slot Attention
        self.event_slots = nn.Parameter(torch.randn(max_events, 128))
        self.slot_norm = nn.LayerNorm(128)

        # Multi-head Proposal Generator
        self.proposal_heads = nn.ModuleDict({
            'existence': nn.Linear(128, 1),
            'distribution': nn.Linear(128, 2 * num_bins),  # Start/end distributions
            'type': nn.Linear(128, num_classes)
        })

        # Learnable temperature for soft event counting
        self.temp = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # x: [B, Freq, Time]
        B = x.size(0)
        context = self.context_encoder(x)  # [B, 128, Time]

        # Cross-attention between event slots and temporal context
        slots = self.slot_norm(self.event_slots.unsqueeze(0).repeat(B, 1, 1))  # [B, 5, 128]
        context = context.permute(0, 2, 1)  # [B, Time, 128]

        # Event-Temporal Attention
        attn_weights = F.softmax(
            torch.matmul(slots, context.transpose(1, 2)) / self.temp,  # [B, 5, Time]
            dim=-1
        )
        attended = torch.matmul(attn_weights, context)  # [B, 5, 128]

        # Dynamic Proposal Generation
        outputs = {
            'existence': torch.sigmoid(self.proposal_heads['existence'](attended)),  # [B,5,1]
            'distributions': F.softmax(
                self.proposal_heads['distribution'](attended).view(B, 5, 2, self.num_bins),
                dim=-1
            ),  # [B,5,2,100]
            'type_logits': self.proposal_heads['type'](attended)  # [B,5,C]
        }

        return outputs


#  Event-Adaptive Subgraph Construction
# class EventSubgraphBuilder(nn.Module):
#     def __init__(self, node_dim=256, num_bins=100):
#         super().__init__()
#         # D-FINE Components
#         self.offset_weights = nn.Parameter(torch.linspace(-0.5, 0.5, num_bins))
#         self.distill_proj = nn.Linear(node_dim, num_bins)
#
#         # Type Consistency Module
#         self.type_consistency = nn.Sequential(
#             nn.Linear(node_dim + num_classes, 128),
#             nn.ReLU(),
#             nn.Linear(128, num_classes)
#         )
#
#     def build_subgraphs(self, node_features, time_stamps, proposals):
#         """ Dynamic subgraph construction with type consistency """
#         subgraphs = []
#         for b in range(proposals['existence'].size(0)):
#             # Decode temporal distributions
#             start_bins = proposals['distributions'][b, :, 0]  # [5,100]
#             end_bins = proposals['distributions'][b, :, 1]  # [5,100]
#
#             # Convert to time offsets
#             start_offsets = (start_bins * self.offset_weights).sum(-1)
#             end_offsets = (end_bins * self.offset_weights).sum(-1)
#
#             # Get valid events (existence > 0.5)
#             valid_mask = proposals['existence'][b].squeeze(-1) > 0.5
#             valid_events = zip(
#                 start_offsets[valid_mask],
#                 end_offsets[valid_mask],
#                 proposals['type_logits'][b][valid_mask]
#             )
#
#             # Build subgraphs per valid event
#             for start, end, type_logit in valid_events:
#                 # 1. Temporal Node Selection
#                 time_mask = (time_stamps[:, 1] > start) & (time_stamps[:, 0] < end)
#                 sg_nodes = node_features[time_mask]
#
#                 if len(sg_nodes) == 0:
#                     continue
#
#                 # 2. Type Consistency Projection
#                 sg_type = type_logit.argmax()
#                 type_emb = F.one_hot(sg_type, num_classes).float()
#                 projected_nodes = self.type_consistency(
#                     torch.cat([sg_nodes, type_emb.expand(len(sg_nodes), -1)], dim=1)
#                 # Feature-Type Alignment: Forces node features to align with subgraph type through concatenated type embeddings.
#                 # 特征类型对齐：通过连接类型嵌入使节点特征与子图类型对齐。
#                 #
#                 # Class-Specific Processing: Enables different feature transformations for normal vs. abnormal breath sounds.
#                 # 类特定处理：为正常与异常呼吸音启用不同的特征变换。
#
#
#                 # 3. Temporal Ordering
#                 sorted_idx = torch.argsort(time_stamps[time_mask, 0])
#                 ordered_nodes = projected_nodes[sorted_idx]
#
#                 # 4. Edge Construction
#                 edges = self.create_temporal_edges(sorted_idx)
#
#                 subgraphs.append({
#                     'nodes': ordered_nodes,
#                     'edges': edges,
#                     'type': sg_type,
#                     'time_range': (start, end),
#                     'distributions': (start_bins, end_bins)
#                 })
#
#         return subgraphs
#
#     def create_temporal_edges(self, sorted_idx):
#         """ Connect consecutive temporal nodes """
#         src = sorted_idx[:-1]
#         dst = sorted_idx[1:]
#         return torch.stack([src, dst], dim=0)




# Type-aware node projection
# projected_nodes = self.type_consistency(
#     torch.cat([sg_nodes, type_emb.expand(...)], dim=1))
#
# # Combines refined boundaries with type information
# context = torch.cat([refined['final_bounds'], type_emb], dim=1)
# cls_logits = self.fusion_net(context)



def adjust_vad_timestamps(vad_intervals, frame_times):
    adjusted_vad_intervals = []
    for interval in vad_intervals:
        start = interval['start']
        end = interval['end']
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


class FactorGraphLayer(nn.Module):
    def __init__(self, node_num_classes, edge_num_classes, num_iterations=2, gamma=1.0):
        super(FactorGraphLayer, self).__init__()
        self.node_num_classes = node_num_classes
        self.edge_num_classes = edge_num_classes
        self.num_iterations = num_iterations
        self.gamma = gamma

        # Learnable parameters: initialize them as ones for simplicity
        # node_factor_weights: shape (node_num_classes, edge_num_classes)
        # edge_factor_weights: shape (edge_num_classes, node_num_classes)
        self.node_factor_weights = nn.Parameter(torch.ones(node_num_classes, edge_num_classes))
        self.edge_factor_weights = nn.Parameter(torch.ones(edge_num_classes, node_num_classes))

    def forward(self, node_logits, edge_logits, batch_graph):
        # node_logits: (N, node_num_classes)
        # edge_logits: (E, edge_num_classes)
        # batch_graph: a Batch object from PyG with edge_index, etc.

        # Convert logits to probabilities
        node_probs = F.softmax(node_logits, dim=-1)  # (N, node_num_classes)
        edge_probs = F.softmax(edge_logits, dim=-1)  # (E, edge_num_classes)

        edge_index = batch_graph.edge_index
        src, dst = edge_index

        for _ in range(self.num_iterations):
            node_abnormal_prob = 1.0 - node_probs[:, 0]  # prob node is abnormal
            edge_node_abnormal = torch.max(node_abnormal_prob[src], node_abnormal_prob[dst])

            # Update edges based on nodes
            # For edges: Adjust abnormal class probabilities using edge_factor_weights
            # We want to scale the adjustment by how strongly node classes influence edges.
            edge_probs_new = edge_probs.clone()
            if self.edge_num_classes > 1:
                abnormal_class_slice = edge_probs[:, 1:]  # (E, edge_num_classes-1)
                abnormal_mass = abnormal_class_slice.sum(dim=-1, keepdim=True)  # (E,1)

                # Compute a weighting factor. For simplicity, we take the average of relevant weights.
                # We have two nodes per edge, we can combine their classes' expected influence.
                # But since we only have probabilities here, let's do a simple weighting:
                # Use edge_factor_weights corresponding to each node class and take an expectation weighted by node_probs.
                # This can get complex. For demonstration, let's just use a learned scalar from the weights:

                # A simple approach:
                # Combine node->edge influence by averaging across node classes weighted by node probabilities
                # This is a simplistic approximation.
                # node_weights_for_edge: (E, node_num_classes) from src and dst
                # For simplicity, just use a single scalar from edge_factor_weights, like a mean:
                avg_edge_factor = self.edge_factor_weights[1:, 1:].mean()  # focusing on abnormal classes
                # Incorporate this factor:
                edge_probs_new[:, 1:] = edge_probs[:, 1:] + self.gamma * edge_node_abnormal.unsqueeze(
                    1) * abnormal_class_slice * avg_edge_factor
                edge_probs_new = edge_probs_new / edge_probs_new.sum(dim=-1, keepdim=True)

            edge_probs = edge_probs_new

            # Update nodes based on edges
            edge_abnormal_prob = 1.0 - edge_probs[:, 0]
            node_edge_abnormal_sum = torch.zeros_like(node_abnormal_prob)
            node_degree = torch.zeros_like(node_abnormal_prob)
            node_edge_abnormal_sum.index_add_(0, src, edge_abnormal_prob)
            node_edge_abnormal_sum.index_add_(0, dst, edge_abnormal_prob)
            node_degree.index_add_(0, src, torch.ones_like(edge_abnormal_prob))
            node_degree.index_add_(0, dst, torch.ones_like(edge_abnormal_prob))

            node_edge_abnormal_mean = node_edge_abnormal_sum / (node_degree + 1e-6)

            node_probs_new = node_probs.clone()
            if self.node_num_classes > 1:
                abnormal_node_slice = node_probs[:, 1:]
                # Similarly, incorporate node_factor_weights
                # Average the node_factor_weights for abnormal node classes
                avg_node_factor = self.node_factor_weights[1:, 1:].mean()
                node_probs_new[:, 1:] = node_probs[:, 1:] + self.gamma * node_edge_abnormal_mean.unsqueeze(
                    1) * abnormal_node_slice * avg_node_factor
                node_probs_new = node_probs_new / node_probs_new.sum(dim=-1, keepdim=True)

            node_probs = node_probs_new

        # Convert back to logits
        node_logits_refined = torch.log(node_probs + 1e-9)
        edge_logits_refined = torch.log(edge_probs + 1e-9)

        return node_logits_refined, edge_logits_refined


class Node_Edge_cls_Module(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 node_num_classes, edge_num_classes=2,
                 max_intervals = 5,
                 num_refine_layers = 3,
                 dist_bins = 100
                 ):

        super(Node_Edge_cls_Module, self).__init__()
        self.gat_model = GATModel(in_channels, hidden_channels, out_channels)
        # Fully Connected Layer for Node Classification
        self.node_classifier = nn.Linear(out_channels, node_num_classes)

        self.node_vad_classifier = nn.Linear(out_channels, 2)
        self.node_embedding_reduce = nn.Sequential(
            nn.Linear(out_channels, 4),
            nn.BatchNorm1d(4),
            nn.ReLU()
        )
        # Fully Connected Layer for Edge Classification
        self.edge_classifier = nn.Linear( 16+8, edge_num_classes)

        self.time_aware_fdr = TimeAwareFDR(
            node_embedding_dim= out_channels,
            max_intervals= max_intervals,
            num_refine_layers= num_refine_layers,
            dist_bins= dist_bins
        )




    def forward(self, data):
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
        edge_predictions_list = []
        all_edge_labels = [] if edge_labels is not None else None

        node_vad_pred_list = []

        event_bounds_list = []


        for i in range(num_audios):
            mask = (batch_indices == i)
            cur_nodes = x[mask]
            cur_node_labels = node_labels[mask]

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
            node_pred = self.node_classifier(node_embeddings)
            node_vad_pred = self.node_vad_classifier(node_embeddings)

            # ---- 4. Compute Time Positions ----
            # Assume nodes are ordered temporally; normalize time to [0, 1]
            num_nodes = node_embeddings.size(0)
            time_positions = torch.linspace(0, 1, num_nodes, device=device).unsqueeze(-1)  # (num_nodes, 1)

            # ---- 5. Predict Event Boundaries ----
            event_output = self.time_aware_fdr(
                node_embeddings=node_embeddings,
                time_positions=time_positions,
                batch_indices=torch.zeros(num_nodes, dtype=torch.long, device=device) )  # Dummy batch index (all 0s)

            # Extract final bounds and mask
            final_bounds = event_output["final_bounds"]  # (max_intervals, 2)
            interval_mask = event_output["interval_mask"]  # (max_intervals,)
            valid_bounds = final_bounds[interval_mask]  # (num_true_intervals, 2)


            node_predictions_list.append(node_pred)
            node_vad_pred_list.append(node_vad_pred)
            all_node_labels.append(cur_node_labels)
            event_bounds_list.append(valid_bounds)

            node_embeddings_reduced = self.node_embedding_reduce(node_embeddings)
            # Edge classification
            if cur_graph.edge_index.size(1) > 0:
                src, dst = cur_graph.edge_index

                # Concatenate edge_attr with reduced node embeddings
                edge_embeddings = torch.cat([
                    cur_graph.edge_attr,  # [num_edges, 16]
                    node_embeddings_reduced[src],  # [num_edges, 4]
                    node_embeddings_reduced[dst]  # [num_edges, 4]
                ], dim=1)  # [num_edges, 16+8]

                edge_pred = self.edge_classifier(edge_embeddings)
                edge_predictions_list.append(edge_pred)
                if cur_edge_labels is not None:
                    all_edge_labels.append(cur_edge_labels)
            else:
                # Handle the case with no edges
                # Just skip since no edges to classify
                pass

        # Concatenate node predictions and labels
        node_predictions = torch.cat(node_predictions_list, dim=0)
        node_vad_pred_batch = torch.cat(node_vad_pred_list, dim=0)

        node_labels = torch.cat(all_node_labels, dim=0)

        # Concatenate edge predictions and labels if present
        if edge_predictions_list:
            edge_predictions = torch.cat(edge_predictions_list, dim=0)
        else:
            # No edges at all (unlikely), return empty
            edge_predictions = torch.empty((0, 2), device=device)

        if all_edge_labels is not None and all_edge_labels:
            edge_labels = torch.cat(all_edge_labels, dim=0)
        else:
            edge_labels = None

        return node_predictions,  edge_predictions, node_vad_pred_batch, event_bounds_list






# Assuming DynamicFeatureExtractor is defined elsewhere
# from desed_task.nnet.DCNN_v3 import DynamicFeatureExtractor

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


        self.node_vad_proj =  nn.Sequential( nn.Linear(node_vad_dim, 128),
                                            # Example hidden layer
                                             nn.ReLU(),
                                             nn.Dropout(0.5),
                                             nn.Linear(128, 2)
                                                )


        # Define alpha as a learnable parameter
        self.alpha = nn.Parameter(torch.ones(1, node_fea_dim))  # Learnable parameter


        # Edge feature encoder: from raw edge_attr (scalar) to learnable embedding
        edge_feature_dim = 16 # if you change this , the GAT layer,edge_dim=2 should also be changed ;
        self.edge_encoder = nn.Linear(4, edge_feature_dim)
        self.node_edge_cls = Node_Edge_cls_Module(adjusted_in_channels, hidden_channels, out_channels, node_num_classes=num_classes)

        # Factor graph layer for enforcing node-edge consistency (You must define this layer)
        self.factor_graph_layer = FactorGraphLayer(node_num_classes=num_classes, edge_num_classes=2, num_iterations=2, gamma=1.0)

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

            # audio_durations = batch_data['audio_durations']
            batch_size = len(spectrograms)

            # Initialize lists
            all_chunks = []
            all_chunk_labels = []

            batch_indices = []
            num_nodes_per_sample = []

            chunk_times_all = []
            node_vad_flags_all = []

            node_gender_all = []
            node_location_all = []

            # for sample_idx, (spec, labels, vad_intervals, audio_dur) in enumerate(
            #         zip(spectrograms, frame_labels, vad_timestamps, audio_durations)):

            for sample_idx, (spec, labels, vad_intervals,) in enumerate(
                        zip(spectrograms, frame_labels, vad_timestamps)):
                gender = genders[sample_idx]  # 0 or 1
                location = locations[sample_idx]  # 0 to 3

                n_frames = spec.shape[1]
                num_nodes = 0

                # Define chunk size
                chunk_size = 5  # Adjust as needed
                # Get frame times
                frame_times = np.arange(n_frames + 1) * self.frame_hop / self.fs
                # Adjust VAD timestamps
                adjusted_vad_intervals = adjust_vad_timestamps(vad_intervals, frame_times)

                # Process spectrogram into fixed-size chunks
                chunk_times = []
                node_vad_flags = []

                for j in range(0, n_frames - chunk_size + 1, chunk_size):
                    chunk = spec[:, j:j + chunk_size, :]
                    all_chunks.append(chunk)

                    # Compute node labels for the chunk
                    # label_chunk = labels[:, j:j + chunk_size]
                    label_chunk = labels[j:j + chunk_size, :]  # Adjusted indexing for shape (chunk_size, 7)
                    node_label = self.get_node_label(label_chunk)
                    all_chunk_labels.append(node_label)

                    # Compute chunk start and end times
                    chunk_start_frame = j
                    chunk_end_frame = j + chunk_size - 1
                    chunk_start_time = frame_times[chunk_start_frame]
                    chunk_end_time = frame_times[chunk_end_frame]
                    chunk_times.append((chunk_start_time, chunk_end_time)) #   记录每个chunk的 起始，终止时间。

                    # Determine if chunk is within VAD intervals
                    node_within_vad = is_chunk_within_vad(chunk_start_time, chunk_end_time, adjusted_vad_intervals)
                    node_vad_flags.append(node_within_vad)  # 用于揭示每个节点 是否存在于真实的vad 区间上；

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
                    label_chunk = labels[:, n_frames - chunk_size:]
                    node_label = self.get_node_label(label_chunk)
                    all_chunk_labels.append(node_label)

                    # Compute chunk start and end times
                    chunk_start_frame = n_frames - chunk_size
                    chunk_end_frame = n_frames - 1
                    chunk_start_time = frame_times[chunk_start_frame]
                    chunk_end_time = frame_times[chunk_end_frame]
                    chunk_times.append((chunk_start_time, chunk_end_time))

                    # Determine if chunk is within VAD intervals
                    node_within_vad = is_chunk_within_vad(chunk_start_time, chunk_end_time, adjusted_vad_intervals)
                    node_vad_flags.append(node_within_vad)

                    # Append gender and location info
                    node_gender_all.append(gender)
                    node_location_all.append(location)

                    num_nodes += 1
                    # Append the sample index for this node
                    batch_indices.append(sample_idx)

                num_nodes_per_sample.append(num_nodes)
                chunk_times_all.extend(chunk_times)
                node_vad_flags_all.extend(node_vad_flags)

            if not all_chunks:
                raise ValueError("No valid chunks generated in the batch.")

            # Stack all chunks and process them together
            all_chunks = [torch.tensor(chunk, dtype=torch.float32) if not isinstance(chunk, torch.Tensor) else chunk for
                          chunk in all_chunks]
            all_chunks_tensor = torch.stack(all_chunks)

            # Generate node features
            #  node_features: (total_nodes, feature_dim=256), node_vad: (total_nodes, feature_dim=256)
            node_features, node_vad = self.node_fea_generator(all_chunks_tensor)


            # should compute loss  with  node_vad_flags_all: (total_nodes ),
            # note, sigmoid output
            node_vad_logit =  self.node_vad_proj(node_vad)  # tensors: (total_nodes, fea_dim=2)
            node_vad_prob = F.softmax(node_vad_logit, dim=1)  # Shape: (total_nodes, 2)


            # Extract 'normal' class probability for gating
            normal_prob = node_vad_prob[:, 0].unsqueeze(1)  # Shape: (total_nodes, 1)

            # Apply gating mechanism: element-wise multiplication
            gated_node_features = node_features * normal_prob  # Shape: (total_nodes, 256)


            # Add original node_features and gated_node_features
            # node_features = gated_node_features + node_features  # Shape: (total_nodes, feature_dim)
            scaled_features = self.alpha * gated_node_features + (1 - self.alpha) * node_features

            # Nonlinear transformation
            node_features = torch.relu(scaled_features)
            # if self.training:
            #     # print(f"\n it's in the trianing stage, Not using vad thread ")
            #     # Use continuous probabilities during training
            #     node_features = torch.cat([ gated_node_features, ], dim=1)
            # else:
            #     #print(f"\n it's in the inference stage, using vad thread:{self.node_vad_thread}")
            #     # Apply thresholding during inference
            #     # node_vad_judge = (node_vad_logit > self.node_vad_thread).float()
            #     # node_features = torch.cat([node_vad_judge,node_vad, node_features,], dim=1)
            #     node_features = torch.cat([ gated_node_features, ], dim=1)


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

            # Create graphs for each sample
            all_graphs = []
            all_edge_attr = []
            start_idx = 0
            node_offset = 0
            for num_nodes in num_nodes_per_sample:
                end_idx = start_idx + num_nodes
                sample_node_features = node_features[start_idx:end_idx]
                sample_node_labels = node_labels[start_idx:end_idx]      # 代表当前样本上每个节点的标签；
                sample_chunk_times = chunk_times_all[start_idx:end_idx]  # 每个节点，所对应的5帧，所处于的时间戳区间，
                sample_node_vad_flags = node_vad_flags_all[start_idx:end_idx] # 该节点是否处于真实的 vad 区间中；

                sample_node_vad = node_vad_logit[start_idx:end_idx]

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
                        edge_within_vad = is_edge_within_vad(src_time, dst_time, adjusted_vad_intervals)

                        # Ground truth edge label:
                        # Abnormal if either node is abnormal or edge is within abnormal interval
                        # node abnormal: node_label != 0
                        src_label = sample_node_labels[src]
                        dst_label = sample_node_labels[dst]
                        edge_is_abnormal = (src_label != 0 or dst_label != 0 or edge_within_vad)
                        edge_label = 1 if edge_is_abnormal else 0
                        edge_labels.append(edge_label)

                        # Initial edge attribute (scalar), for example just the edge_within_vad as a float
                        # edge_attr_val = float(edge_within_vad)
                        # edge_attrs.append([edge_attr_val])  #  这里边的属性不能依靠 时间戳信息生成；

                    #edge_attr = torch.tensor(edge_attrs, dtype=torch.float32, device=device)
                    edge_labels = torch.tensor(edge_labels, dtype=torch.long, device=device)

                    # Generate edge_features based on connected node features
                    src_node_vad = sample_node_vad[edge_index[0]]  # [E, feature_dim=32]
                    dst_node_vad = sample_node_vad[edge_index[1]]  # [E, feature_dim]
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

            # Node-Level Classification
            node_predictions, edge_predictions, node_emb_vad_pred = self.node_edge_cls(batch_graph)
            # note, Factor Graph Layer for consistency refinement
            node_predictions_refined, edge_predictions_refined = self.factor_graph_layer(node_predictions, edge_predictions, batch_graph)


            # now shape is [sum(E_i), 1]

            # Prepare outputs for the node predictions
            outputs = {
                'node_predictions': node_predictions,
                'node_labels': node_labels,

                'edge_predictions': edge_predictions, # note, it's wrong should be edge_predictions ;
                'edge_labels': torch.cat([g.edge_y for g in all_graphs], dim=0) if len(all_graphs) > 0 else torch.empty( 0, dtype=torch.long, device=device),

                'node_pred_refine': node_predictions_refined,
                'edge_pred_refine': edge_predictions_refined,

                'node_vad_falg': node_vad_flags_all,
                'node_vad_logit': node_vad_logit,   # 用于衡量节点是否位于 vad 中的属性， 与 node_vad_falg 形成loss;
                'node_emb_vad': node_emb_vad_pred,

                'batch_edge_index': batch_graph.edge_index,
                'batch_graph':batch_graph,

                'batch_indices': batch_indices,
                'batch_audio_names': c_ex_mixtures  # This should correspond to the samples in detection
            }

        return outputs

    def get_node_label(self,frames, normal_label=0, abnormal_threshold=0.2):
        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()

        # frames shape: (n_frames, num_classes)
        frames_label = frames.argmax(axis=1)  # Get the label index for each frame

        total_frames = len(frames_label)
        num_classes = frames.shape[1]
        counts = np.bincount(frames_label, minlength=num_classes)

        majority_label = counts.argmax()
        majority_count = counts[majority_label]

        if majority_label == normal_label:
            # Identify abnormal labels exceeding the threshold
            significant_abnormals = [
                (label, count) for label, count in enumerate(counts)
                if label != normal_label and count > abnormal_threshold * total_frames
            ]

            if significant_abnormals:
                # Find the abnormal label(s) with the highest ratio (count)
                max_count = max(significant_abnormals, key=lambda x: x[1])[1]
                max_labels = [label for label, count in significant_abnormals if count == max_count]

                if len(max_labels) == 1:
                    # Only one label has the highest count
                    return max_labels[0]
                else:
                    # Multiple labels have the same max count
                    # Select the one that appears earlier in the frames
                    earliest_indices = {
                        label: np.where(frames_label == label)[0][0] for label in max_labels
                    }
                    # Select label with earliest occurrence (lowest frame index)
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

