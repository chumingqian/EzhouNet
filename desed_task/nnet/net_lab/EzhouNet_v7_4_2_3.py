

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
from desed_task.nnet.DCNN_v3 import  DynamicFeatureExtractor
# Adjusted GATConv that supports edge attributes (your implementation)
import numpy as np

from torch_geometric.data import Data
from torch_geometric.utils import subgraph
#, connected_components
import networkx as nx

# v6-5-4, 将vad 的时间戳信息编码到 node 的特征中，以及图的边缘属性中；
# v6-5-5, 增加采集时的胸腔位置信息，以及性别信息；；

# v7-1-1,  开始使用边的预测信息

# v7-2-2,  20 frames  作为一个节点，进行特征生成
# v7-2-3,  10 frames  作为一个节点，进行特征生成
#  7-4-1,  丢弃 将时间戳的信息引入到模型中的方式， 这种方式提前泄漏了时间信息；

#  7-4-2,  重新引入时间戳信息， 作为边的 真实值标签；


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
        self.gat = GATConv(in_channels, out_channels, heads=heads, concat=False, edge_dim=256)

    def forward(self, x, edge_index, edge_attr):
        # print("GATLayer in_channels:", self.gat.in_channels)
        # print(f"x shape in GATLayer: {x.shape}")
        # print(f"edge_index shape in GATLayer: {edge_index.shape}")
        # print(f"edge_attr shape in GATLayer: {edge_attr.shape}")

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



from torch_geometric.nn import GINEConv, EdgeConv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv


class FactorGraphLayer(nn.Module):
    def __init__(self, node_num_classes, edge_num_classes, hidden_dim=32, num_iterations=2, gamma=1.0):
        super(FactorGraphLayer, self).__init__()
        self.num_iterations = num_iterations
        self.gamma = gamma

        # Define MLP for initial node feature projection
        self.node_initial_proj = nn.Linear(node_num_classes, hidden_dim)

        # Define GINEConv layers for nodes with updated MLP input dimensions
        self.node_conv = GINEConv(nn=nn.Sequential(
            nn.Linear( hidden_dim, hidden_dim),  # Updated input size
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ))

        # Define MLP for edge feature transformation
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Define attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4)

        # Projection layers to ensure correct dimensions
        self.attn_proj = nn.Linear(2 * hidden_dim, hidden_dim)

        # Classification heads
        self.node_classification_head = nn.Linear(hidden_dim, node_num_classes)
        self.edge_classification_head = nn.Linear(hidden_dim, edge_num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, node_logits, edge_logits, batch_graph):
        # Initialize node and edge features
        node_feats = self.node_initial_proj(node_logits)  # (N, hidden_dim)
        edge_feats = edge_logits  # (E, edge_num_classes)

        # Transform edge features
        edge_feats = self.edge_mlp(edge_feats)  # (E, hidden_dim)

        edge_index = batch_graph.edge_index
        src, dst = edge_index

        for iteration in range(self.num_iterations):
            # Interaction between node and edge features
            combined_node_feats = torch.cat([node_feats[src], node_feats[dst]], dim=1)  # (E, 2 * hidden_dim)

            # Project combined features to hidden_dim
            combined_node_feats = self.attn_proj(combined_node_feats)  # (E, hidden_dim)

            # Apply attention
            combined_node_feats = combined_node_feats.unsqueeze(1)  # (E, 1, hidden_dim)
            attn_output, _ = self.attention(combined_node_feats, combined_node_feats, combined_node_feats)
            interaction_message = attn_output.squeeze(1)  # (E, hidden_dim)

            # Update edge features with interaction message
            edge_feats = F.relu(edge_feats + self.gamma * interaction_message)  # (E, hidden_dim)

            # Update node features using GINEConv with updated edge features
            node_feats = F.relu(self.node_conv(node_feats, edge_index, edge_feats))  # (N, hidden_dim)

        # Final transformations to logits
        node_logits_refined = self.node_classification_head(node_feats)  # (N, node_num_classes)
        edge_logits_refined = self.edge_classification_head(edge_feats)  # (E, edge_num_classes)

        return node_logits_refined, edge_logits_refined

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv
import torch_scatter


# --- Define the Edge Attention Layers ---

"""

Node Processing: Use GATConv exclusively for updating node embeddings.节点处理：专门使用GATConv来更新节点嵌入。
Edge Processing: Use EdgeAttentionUpdater or similar mechanisms for edge embedding updates.

The GATConv layer from torch_geometric is designed for node feature aggregation, not for edge feature updates. 
Its primary function is to update node embeddings based on their neighbors' features. 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeAttentionUpdater(nn.Module):
    def __init__(self, node_embedding_dim, edge_embedding_dim):
        super(EdgeAttentionUpdater, self).__init__()
        self.attn = nn.Linear(2 * node_embedding_dim + edge_embedding_dim, 1)
        self.update = nn.Linear(2 * node_embedding_dim + edge_embedding_dim, edge_embedding_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, src_node_embeddings, dst_node_embeddings, edge_embeddings):
        """
        Args:
            src_node_embeddings (Tensor): [E, node_embedding_dim]
            dst_node_embeddings (Tensor): [E, node_embedding_dim]
            edge_embeddings (Tensor): [E, edge_embedding_dim]

        Returns:
            Tensor: [E, edge_embedding_dim] - Updated edge embeddings
        """
        # Concatenate source and destination node embeddings with edge embeddings
        combined = torch.cat([src_node_embeddings, dst_node_embeddings, edge_embeddings],
                             dim=1)  # [E, 2*node_dim + edge_dim]

        # Compute attention scores
        attn_scores = self.leaky_relu(self.attn(combined))  # [E, 1]
        attn_scores = self.sigmoid(attn_scores)  # [E, 1]

        # Update edge embeddings
        updated_edges = self.update(combined)  # [E, edge_dim]

        # Apply attention scores
        updated_edges = updated_edges * attn_scores  # [E, edge_dim]

        return updated_edges


class EdgeAttentionLayer(nn.Module):
    def __init__(self, node_embedding_dim, edge_embedding_dim, edge_out_channels, dropout=0.6):
        """
        Args:
            node_embedding_dim (int): Dimension of node embeddings.
            edge_embedding_dim (int): Dimension of edge embeddings.
            edge_out_channels (int): Desired output dimension for edge embeddings after attention.
            dropout (float): Dropout rate.
        """
        super(EdgeAttentionLayer, self).__init__()

        # Initialize the EdgeAttentionUpdater
        self.edge_attention_updater = EdgeAttentionUpdater(
            node_embedding_dim=node_embedding_dim,
            edge_embedding_dim=edge_embedding_dim
        )

        # Additional linear layers for combining embeddings
        self.edge_attn_linear = nn.Linear(edge_out_channels, edge_out_channels)
        self.node_attn_linear = nn.Linear(node_embedding_dim, edge_out_channels)

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, edge_embeddings, edge_index, edge_attr, node_embeddings, num_nodes):
        """
        Args:
            edge_embeddings (Tensor): [E, edge_embedding_dim]
            edge_index (Tensor): [2, E]
            edge_attr (Tensor): [E, edge_attr_dim=256]
            node_embeddings (Tensor): [N, node_embedding_dim]
            num_nodes (int): Total number of nodes in the graph

        Returns:
            Tensor: [E, edge_out_channels] - Refined edge embeddings
        """
        # print("Starting EdgeAttentionLayer.forward()")
        # print(f"edge_embeddings shape: {edge_embeddings.shape}")
        # print(f"edge_index shape: {edge_index.shape}")
        # print(f"edge_attr shape: {edge_attr.shape}")
        # print(f"node_embeddings shape: {node_embeddings.shape}")
        # print(f"num_nodes: {num_nodes}")

        # Ensure edge_index is of type torch.long
        if edge_index.dtype != torch.long:
            print(f"Converting edge_index from {edge_index.dtype} to torch.long")
            edge_index = edge_index.long()

        # Validate edge_index against the correct num_nodes
        if edge_index.numel() > 0:  # Ensure edge_index is not empty
            max_index = edge_index.max().item()
            min_index = edge_index.min().item()
            #print(f"Edge index max: {max_index}, min: {min_index}, num_nodes: {num_nodes}")
            if max_index >= num_nodes or min_index < 0:
                print(f"Invalid edge_index values: {edge_index}")
                raise RuntimeError("edge_index contains invalid node indices!")

        # Check for NaN or Inf in edge_embeddings, edge_attr, and node_embeddings
        if not torch.isfinite(edge_embeddings).all():
            print("edge_embeddings contains NaN or Inf values!")
            raise RuntimeError("edge_embeddings contains NaN or Inf values!")
        if not torch.isfinite(edge_attr).all():
            print("edge_attr contains NaN or Inf values!")
            raise RuntimeError("edge_attr contains NaN or Inf values!")
        if not torch.isfinite(node_embeddings).all():
            print("node_embeddings contains NaN or Inf values!")
            raise RuntimeError("node_embeddings contains NaN or Inf values!")

        # print(f"edge_in_channels: {edge_embeddings.shape[1]}")
        # print(f"edge_out_channels: {self.edge_attn_linear.out_features}")
        # print(f"dropout: {self.dropout.p}")

        # Extract source and destination node embeddings for each edge
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]
        src_node_embeddings = node_embeddings[src_nodes]  # [E, node_embedding_dim]
        dst_node_embeddings = node_embeddings[dst_nodes]  # [E, node_embedding_dim]

        # Update edge embeddings using EdgeAttentionUpdater
        updated_edge_embeddings = self.edge_attention_updater(
            src_node_embeddings=src_node_embeddings,
            dst_node_embeddings=dst_node_embeddings,
            edge_embeddings=edge_embeddings
        )  # [E, edge_embedding_dim]

        # print(f"After EdgeAttentionUpdater: {updated_edge_embeddings.shape}")

        # Further refine edge embeddings with additional linear layers
        edge_feat = self.edge_attn_linear(updated_edge_embeddings)  # [E, edge_out_channels]
        src_feat = self.node_attn_linear(src_node_embeddings)  # [E, edge_out_channels]
        dst_feat = self.node_attn_linear(dst_node_embeddings)  # [E, edge_out_channels]

        attn_scores = self.leaky_relu(edge_feat + src_feat + dst_feat)  # [E, edge_out_channels]
        attn_scores = F.softmax(attn_scores, dim=0)  # Normalize across edges

        attn_scores = self.dropout(attn_scores)  # [E, edge_out_channels]
        refined_edge_embeddings = updated_edge_embeddings * attn_scores  # [E, edge_embedding_dim]

        #print("Completed EdgeAttentionLayer.forward()")
        return refined_edge_embeddings


# --- Define the Main Model ---
# Node_Edge_cls_Module_Attention
class Node_Edge_cls_Module(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 node_num_classes, edge_num_classes=2, subgraph_num_classes=5,
                 subgraph_hidden_size=256):
        super(Node_Edge_cls_Module, self).__init__()

        # Graph Attention Network for Node Embeddings
        self.gat_model = GATConv(in_channels, hidden_channels, heads=1, concat=True, dropout=0.6)
        self.node_embedding_linear = nn.Linear(hidden_channels, out_channels)

        self.edge_emb = nn.Sequential(
            nn.Linear(2 * out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 256)  # Output dimension for edge_features
        )
        # Fully Connected Layers for Classification
        self.node_classifier = nn.Linear(out_channels, node_num_classes)
        self.edge_classifier = nn.Linear(out_channels, edge_num_classes)

        # Attention Layers for Edge Smoothing and Importance
        self.edge_attention_layer = EdgeAttentionLayer(
            node_embedding_dim=out_channels,
            edge_embedding_dim=out_channels,
            edge_out_channels= 256,
            dropout=0.6
        )

        # Subgraph Classification Layers
        # self.subgraph_gru = nn.GRU(input_size=out_channels, hidden_size=subgraph_hidden_size, batch_first=True,num_layers=2)
        self.subgraph_classifier = nn.Linear(subgraph_hidden_size, subgraph_num_classes)

    def forward(self, data):
        device = data.x.device

        # Unpack data
        x = data.x.to(device)  # [N, in_channels]
        edge_index = data.edge_index.to(device)  # [2, E]
        edge_attr = data.edge_attr.to(device)  # [E, in_channels]
        batch_indices = data.batch.to(device)  # [N]
        node_labels = data.y.to(device)  # [N]

        # If edge labels exist
        edge_labels = data.edge_y.to(device) if hasattr(data, 'edge_y') else None
        num_audios = batch_indices.max().item() + 1

        node_predictions_list = []
        all_node_labels = []
        edge_predictions_list = []
        all_edge_labels = [] if edge_labels is not None else None


        # Collect all edge_logits across the batch for attention-based smoothing
        edge_graph_offsets = [0]  # Cumulative sum of edges per graph

        # To store node embeddings for later use in attention layer
        all_node_embeddings = []
        node_counts = []  # Track number of nodes per graph
        edge_indices_per_graph = []
        edge_embeddings_per_graph = []



        # Calculate node offsets for mapping local to global indices
        node_offsets = [0]
        for i in range(num_audios):
            mask = (batch_indices == i)
            num_nodes_i = mask.sum().item()
            node_counts.append(num_nodes_i)
            node_offsets.append(node_offsets[-1] + num_nodes_i)

        for i in range(num_audios):
            mask = (batch_indices == i)
            cur_nodes = x[mask]  # [N_i, in_channels]
            # node_counts.append(cur_nodes.size(0))  # Track node count
            cur_node_labels = node_labels[mask]  # [N_i]

            # Get the global indices of the nodes for the current audio
            cur_nodes_indices = mask.nonzero(as_tuple=False).view(-1)

            # Filter edges that belong to the current audio
            edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
            cur_edge_index0 = edge_index[:, edge_mask]  # [2, E_i]
            cur_edge_attr0 = edge_attr[edge_mask]  # [E_i, in_channels]

            # Map global indices to local indices
            mapping = {g_idx.item(): l_idx for l_idx, g_idx in enumerate(cur_nodes_indices)}
            # Convert global -> local edge indices
            src_local = torch.tensor([mapping[idx.item()] for idx in cur_edge_index0[0]],
                                     dtype=torch.long, device=device)
            dst_local = torch.tensor([mapping[idx.item()] for idx in cur_edge_index0[1]],
                                     dtype=torch.long, device=device)
            cur_edge_index = torch.stack([src_local, dst_local], dim=0)  # [2, E_i]

            # Extract edge labels for the current audio if available
            if edge_labels is not None:
                cur_edge_labels = edge_labels[edge_mask]  # [E_i]
            else:
                cur_edge_labels = None

            #print(f"Graph {i}: cur_edge_index max: {cur_edge_index.max().item()}, num_nodes: {cur_nodes.size(0)}")
            assert cur_edge_index.max() < cur_nodes.size(0), f"Invalid edge index in graph {i}!"

            # Apply GAT to node features
            node_embeddings = self.gat_model(cur_nodes, cur_edge_index,  edge_attr=cur_edge_attr0)  # [N_i, hidden_channels]
            node_embeddings = F.elu(node_embeddings)  # Activation
            node_embeddings = self.node_embedding_linear(node_embeddings)  # [N_i, out_channels]

            all_node_embeddings.append(node_embeddings)

            # Node classification
            node_logits = self.node_classifier(node_embeddings)  # [N_i, node_num_classes]
            node_predictions_list.append(node_logits)
            all_node_labels.append(cur_node_labels)

            # Edge classification
            E = cur_edge_index.size(1)
            if E > 0:
                src, dst = cur_edge_index


                # Alternative approach: concatenating node features and passing through MLP
                src_node_features = node_embeddings[src]  # [E_i, out_channels]
                dst_node_features = node_embeddings[dst]  # [E_i, out_channels]
                combined_node_features = torch.cat([src_node_features, dst_node_features],  dim=1)  # [E_i, 2 * out_channels]


                edge_embeddings = self.edge_emb(combined_node_features)  # [E_i, 256]
                edge_logits = self.edge_classifier(edge_embeddings)  # [E_i, edge_num_classes]
                edge_predictions_list.append(edge_logits)


                # Save GT for edges
                if cur_edge_labels is not None:
                    all_edge_labels.append(cur_edge_labels)

                # Collect edge embeddings for attention-based smoothing

                edge_indices_per_graph.append(cur_edge_index)
                edge_embeddings_per_graph.append(edge_embeddings)
            else:
                # No edges in this graph
                edge_indices_per_graph.append(cur_edge_index)
                edge_embeddings_per_graph.append(torch.empty((0, self.gat_model.out_channels), device=device))

        if edge_indices_per_graph:
            # Adjust edge indices with global offsets using tracked node counts
            node_offset = 0
            for i, cur_edge_index in enumerate(edge_indices_per_graph):
                # cur_edge_index += node_offset  # Add offset to align indices
                cur_edge_index = cur_edge_index + node_offsets[i]  # Out-of-place addition
                edge_indices_per_graph[i] = cur_edge_index
                node_offset += node_counts[i]  # Update node offset for next graph

            # Concatenate all edge indices
            concatenated_edge_index = torch.cat(edge_indices_per_graph, dim=1)  # Combine all local edge indices
        else:
            concatenated_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

        num_nodes = x.size(0)  # Total number of nodes in the batch
        print(f"Total number of nodes in batch: {num_nodes}")
        if concatenated_edge_index.numel() > 0:
            print(
                f"Concatenated edge_index max: {concatenated_edge_index.max().item()}, min: {concatenated_edge_index.min().item()}")
        else:
            print("No edges in the batch.")

        # Concatenate all edge_embeddings
        if edge_embeddings_per_graph:
            concatenated_edge_embeddings = torch.cat(edge_embeddings_per_graph, dim=0)  # [Total_E, out_channels]
        else:
            concatenated_edge_embeddings = torch.empty((0, self.gat_model.out_channels), device=device)

        # Filter invalid edges
        valid_mask = (concatenated_edge_index[0] < num_nodes) & (concatenated_edge_index[1] < num_nodes)
        concatenated_edge_index = concatenated_edge_index[:, valid_mask]
        concatenated_edge_embeddings = concatenated_edge_embeddings[valid_mask]
        concatenated_edge_attr = edge_attr[valid_mask] if edge_attr is not None else None



        print(f"After filtering:")
        print(f"Number of nodes: {num_nodes}")
        if concatenated_edge_index.numel() > 0:
            print(
                f"Edge index max: {concatenated_edge_index.max().item()}, min: {concatenated_edge_index.min().item()}")
        else:
            print("No valid edges after filtering.")

        assert concatenated_edge_index.max().item() < num_nodes, "Invalid edge_index: max index out of range!"
        assert concatenated_edge_index.min().item() >= 0, "Invalid edge_index: negative indices!"


        # Aggregate all node embeddings
        all_node_embeddings_cat = torch.cat(all_node_embeddings, dim=0)  # [Total_N, out_channels]


        # Apply Attention-Based Edge Smoothing
        if concatenated_edge_embeddings.size(0) > 0:
            smoothed_edge_embeddings = self.edge_attention_layer(
                edge_embeddings=concatenated_edge_embeddings,  # [Total_E, out_channels]
                edge_index=concatenated_edge_index,  # [2, E]
                edge_attr=concatenated_edge_attr,  # [E, in_channels]
                node_embeddings=all_node_embeddings_cat,  # [N, out_channels]
                num_nodes=num_nodes
            )  # [Total_E, out_channels]

        # === Refine Edge Predictions Based on Smoothed Embeddings ===
        if smoothed_edge_embeddings.size(0) > 0:
            refined_edge_logits = self.edge_classifier(smoothed_edge_embeddings)  # [Total_E, 2]
            edge_pred_labels_smoothed = refined_edge_logits.argmax(dim=1).cpu().numpy()  # [Total_E]
        else:
            edge_pred_labels_smoothed = np.array([], dtype=int)

        subgraph_node_embeddings = []  # List of tensors [num_nodes_in_subgraph, out_channels]
        subgraph_labels = []  # List of tensors [1]
        subgraph_node_indices_list = []
        subgraph_node_predictions_list = []

        # === Create Subgraph Data Objects and Batch Them ===
        subgraph_data_list = []

        for i in range(num_audios):
            E = edge_indices_per_graph[i].size(1)
            if E > 0:
                # Extract smoothed edge labels for the current graph
                if len(edge_graph_offsets) > i + 1:
                    start = edge_graph_offsets[i]
                    end = edge_graph_offsets[i + 1]
                    edge_pred_labels_smoothed_i = edge_pred_labels_smoothed[start:end]
                else:
                    edge_pred_labels_smoothed_i = np.array([], dtype=int)
            else:
                edge_pred_labels_smoothed_i = np.array([], dtype=int)

            # Identify important edges based on smoothed predictions
            if E > 0 and len(edge_pred_labels_smoothed_i) > 0:
                important_edge_mask = (edge_pred_labels_smoothed_i == 1)
                important_edge_indices = torch.nonzero(torch.tensor(important_edge_mask, device=device),
                                                       as_tuple=False).squeeze()
                if important_edge_indices.numel() > 0:
                    important_edge_index = edge_indices_per_graph[i][:, important_edge_indices]

                    # Validate edge indices
                    print(f"Graph {i}: Important_edge_index shape: {important_edge_index.shape}")
                    if important_edge_index.numel() > 0:
                        max_idx = important_edge_index.max().item()
                        print(f"Graph {i}: Important_edge_index max: {max_idx}")
                        assert max_idx < node_counts[
                            i], f"Graph {i}: Edge index max {max_idx} >= num_nodes {node_counts[i]}"
                        assert important_edge_index.min().item() >= 0, f"Graph {i}: Edge index min {important_edge_index.min().item()} <0"

                    labels = self.custom_connected_components(important_edge_index, num_nodes=node_counts[i])
                    num_subgraphs = labels.max().item() + 1
                    for sub_id in range(num_subgraphs):
                        sub_nodes = (labels == sub_id).nonzero(as_tuple=False).view(-1)
                        sub_nodes_tensor = sub_nodes.to(device)

                        # Map local subgraph node indices to global node indices
                        global_sub_nodes_tensor = sub_nodes_tensor + node_offsets[i]

                        # Get subgraph node embeddings
                        subgraph_node_emb = all_node_embeddings_cat[global_sub_nodes_tensor]  # Correctly indexed
                        subgraph_node_embeddings.append(subgraph_node_emb)

                        # Determine subgraph label via majority vote
                        node_pred_labels = F.softmax(node_predictions_list[i], dim=1).argmax(dim=1).cpu().numpy()
                        node_labels_in_subgraph = node_pred_labels[sub_nodes_tensor.cpu().numpy()]
                        if len(node_labels_in_subgraph) > 0:
                            labels_, counts = np.unique(node_labels_in_subgraph, return_counts=True)
                            subgraph_label = labels_[counts.argmax()]
                        else:
                            subgraph_label = 0  # Default to normal if no nodes

                        subgraph_label_tensor = torch.tensor([subgraph_label], dtype=torch.long, device=device)
                        subgraph_labels.append(subgraph_label_tensor)

                        # Store node indices & their logits for consistency loss
                        subgraph_node_indices_list.append(global_sub_nodes_tensor)
                        subgraph_node_predictions_list.append(
                            node_predictions_list[i][sub_nodes_tensor])  # Use local indices
            else:
                # No important edges => treat the entire sample as one normal subgraph
                n_nodes = node_counts[i]
                if n_nodes > 0:
                    sub_nodes_tensor = torch.arange(n_nodes, device=device)
                    global_sub_nodes_tensor = sub_nodes_tensor + node_offsets[i]
                    subgraph_node_emb = all_node_embeddings_cat[global_sub_nodes_tensor]  # Correctly indexed
                    subgraph_node_embeddings.append(subgraph_node_emb)

                    # Determine subgraph label via majority vote
                    node_pred_labels = F.softmax(node_predictions_list[i], dim=1).argmax(dim=1).cpu().numpy()
                    node_labels_in_subgraph = node_pred_labels
                    if len(node_labels_in_subgraph) > 0:
                        labels_, counts = np.unique(node_labels_in_subgraph, return_counts=True)
                        subgraph_label = labels_[counts.argmax()]
                    else:
                        subgraph_label = 0  # Default to normal if no nodes

                    subgraph_label_tensor = torch.tensor([subgraph_label], dtype=torch.long, device=device)
                    subgraph_labels.append(subgraph_label_tensor)

                    # Store node indices & their logits for consistency loss
                    subgraph_node_indices_list.append(global_sub_nodes_tensor)
                    subgraph_node_predictions_list.append(node_predictions_list[i][sub_nodes_tensor])

        # Create Data objects for subgraphs
        for emb, label in zip(subgraph_node_embeddings, subgraph_labels):
            data = Data(x=emb, y=label)
            subgraph_data_list.append(data)

        # Create a Batch object from the list of subgraph Data objects
        if subgraph_data_list:
            subgraph_batch = Batch.from_data_list(subgraph_data_list)
            # subgraph_batch.x: [Total_Subgraph_Nodes, out_channels]
            # subgraph_batch.batch: [Total_Subgraph_Nodes] indicating subgraph membership
            # subgraph_batch.y: [num_subgraphs, subgraph_num_classes]

            # Option 1: Direct Classification without GRU
            subgraph_embeddings = torch_scatter.scatter_mean(subgraph_batch.x, subgraph_batch.batch,
                                                             dim=0)  # [num_subgraphs, out_channels]

            # Classify subgraphs directly
            subgraph_logits = self.subgraph_classifier(subgraph_embeddings)  # [num_subgraphs, subgraph_num_classes]

            # Option 2: GRU-Based Classification (if desired)
            """
            # Aggregate node embeddings by mean pooling per subgraph
            subgraph_embeddings = torch_scatter.scatter_mean(subgraph_batch.x, subgraph_batch.batch, dim=0)  # [num_subgraphs, out_channels]

            # Reshape for GRU: [num_subgraphs, seq_len=1, out_channels]
            subgraph_embeddings = subgraph_embeddings.unsqueeze(1)  # [num_subgraphs, 1, out_channels]

            # Pack the sequences
            lengths = torch.ones(subgraph_embeddings.size(0), dtype=torch.long, device=device)
            packed_subgraph_embeddings = pack_padded_sequence(
                subgraph_embeddings,
                lengths,
                batch_first=True,
                enforce_sorted=False
            )

            # Pass through GRU
            packed_output, hidden = self.subgraph_gru(packed_subgraph_embeddings)
            # hidden: [num_layers, num_subgraphs, hidden_size]

            # Extract the last hidden state
            final_hidden = hidden[-1]  # [num_subgraphs, hidden_size]

            # Classify subgraphs
            subgraph_logits = self.subgraph_classifier(final_hidden)  # [num_subgraphs, subgraph_num_classes]
            """
        else:
            subgraph_logits = torch.empty((0, self.subgraph_classifier.out_features), device=device)

        # Concatenate node predictions and labels
        node_predictions = torch.cat(node_predictions_list, dim=0)  # [Total_nodes, node_num_classes]
        node_labels = torch.cat(all_node_labels, dim=0)  # [Total_nodes]

        # Concatenate edge predictions and labels if present
        if edge_predictions_list:
            edge_predictions = torch.cat(edge_predictions_list, dim=0)  # [Total_edges, 2]
        else:
            edge_predictions = torch.empty((0, 2), device=device)

        if all_edge_labels is not None and all_edge_labels:
            edge_labels = torch.cat(all_edge_labels, dim=0)  # [Total_edges]
        else:
            edge_labels = None

        # Concatenate subgraph labels
        subgraph_labels_tensor = torch.cat(subgraph_labels, dim=0) if subgraph_labels else torch.empty((0,),
                                                                                                       dtype=torch.long,
                                                                                                       device=device)

        # Compile outputs
        outputs = {
            'node_predictions': node_predictions,  # [Total_nodes, node_num_classes]
            'node_labels': node_labels,  # [Total_nodes]

            'edge_predictions': edge_predictions,  # [Total_edges, 2]
            'edge_labels': edge_labels,  # [Total_edges]

            'subgraph_predictions': subgraph_logits,  # [num_subgraphs, subgraph_num_classes]
            'subgraph_labels': subgraph_labels_tensor,  # [num_subgraphs]

            'subgraph_node_indices': subgraph_node_indices_list,  # List of tensors
            'subgraph_node_predictions': subgraph_node_predictions_list  # List of tensors
        }

        return outputs

    # --- Additional Helper Functions ---

    def custom_connected_components(self, edge_index, num_nodes):
        """
        Identifies connected components in a graph using Union-Find.

        Args:
            edge_index (Tensor): [2, E] tensor containing edge indices.
            num_nodes (int): Number of nodes in the graph.

        Returns:
            Tensor: [num_nodes] tensor containing component labels for each node.
        """
        parent = list(range(num_nodes))  # Initialize parent pointers

        def find(x):
            # Path compression
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            x_root = find(x)
            y_root = find(y)
            if x_root != y_root:
                parent[y_root] = x_root

        # Perform unions for each edge
        for i in range(edge_index.size(1)):
            x = edge_index[0, i].item()
            y = edge_index[1, i].item()

            # Debugging: Print the edge being processed
            print(f"Processing edge {i}: ({x}, {y})")

            # Assert that x and y are within valid range
            assert 0 <= x < num_nodes, f"Node index x={x} out of bounds for num_nodes={num_nodes}"
            assert 0 <= y < num_nodes, f"Node index y={y} out of bounds for num_nodes={num_nodes}"

            union(x, y)

        # Assign component labels
        components = [find(x) for x in range(num_nodes)]
        components = torch.tensor(components, dtype=torch.long, device=edge_index.device)

        # Optional: Map component roots to continuous labels
        unique_roots, labels = torch.unique(components, return_inverse=True)
        return labels


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
                 node_fea_dim=512,
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

        self.include_gender = include_gender
        self.include_location = include_location

        # Determine the number of additional features
        # num_additional_features = 1  # For node_vad_flags_tensor
        # No extra features now
        num_additional_features = 0
        if self.include_gender:
            num_additional_features += 2  # Gender one-hot encoding size
        if self.include_location:
            num_additional_features += 4  # Location one-hot encoding size

        # Adjust in_channels accordingly
        adjusted_in_channels = node_fea_dim + num_additional_features
        # Edge feature encoder: from raw edge_attr (scalar) to learnable embedding

        # if you change output dim , the GAT layer ,edge_dim=2  & EdgeAttention should also be changed ;
        # Define an MLP to generate edge features from node features
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_fea_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # Output dimension for edge_features
        )  # 边的属性这里设置为 3, 代表的含义是， 正常和正常， 正常和异常，  异常和异常；  存在着三种类型的边；

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
            all_chunks = []  #  用于存储节点 以及 节点标签
            all_chunk_labels = []

            batch_indices = []
            num_nodes_per_sample = []

            chunk_times_all = []
            node_vad_flags_all = []

            node_gender_all = []
            node_location_all = []

            # for sample_idx, (spec, labels, vad_intervals, audio_dur) in enumerate(
            #         zip(spectrotograms, frame_labels, vad_timestamps, audio_durations)):

            for sample_idx, (spec, labels, vad_intervals,) in enumerate(
                        zip(spectrograms, frame_labels, vad_timestamps)):

                gender = genders[sample_idx]  # 0 or 1
                location = locations[sample_idx]  # 0 to 3

                n_frames = spec.shape[1]
                num_nodes = 0


                # Define chunk size and stride
                chunk_size = 5  # Adjust as needed
                stride = 5  # For non-overlapping chunks; adjust as needed


                # Get frame times
                frame_times = np.arange(n_frames + 1) * self.frame_hop / self.fs
                # Adjust VAD timestamps
                adjusted_vad_intervals = adjust_vad_timestamps(vad_intervals, frame_times)

                # Process spectrogram into fixed-size chunks
                chunk_times = []
                node_vad_flags = []


                for j in range(0, n_frames - chunk_size + 1, stride):
                    chunk = spec[:, j:j + chunk_size, :]
                    all_chunks.append(chunk)

                    # Compute node labels for the chunk
                    # label_chunk = labels[:, j:j + chunk_size]
                    label_chunk = labels[j:j + chunk_size, :]  # Adjusted indexing for shape (chunk_size, 7)
                    node_label = self.get_node_label(label_chunk)
                    all_chunk_labels.append(node_label)

                    # Start/end times
                    chunk_start_frame = j
                    chunk_end_frame = j + chunk_size - 1
                    chunk_start_time = frame_times[chunk_start_frame]
                    chunk_end_time = frame_times[chunk_end_frame]
                    chunk_times.append((chunk_start_time, chunk_end_time))

                    # Whether chunk is within VAD intervals (for debugging or reference, not for model input)
                    node_within_vad = is_chunk_within_vad(chunk_start_time, chunk_end_time, adjusted_vad_intervals)
                    node_vad_flags.append(node_within_vad)


                    # Append gender and location info
                    node_gender_all.append(gender)
                    node_location_all.append(location)

                    num_nodes += 1
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
            node_features = self.node_fea_generator(all_chunks_tensor)  # Output: (total_nodes, feature_dim)
            node_features = node_features.to(device)

            # Stack node labels
            node_labels = torch.tensor(all_chunk_labels, dtype=torch.long, device=node_features.device)
            # Convert batch_indices to tensor
            batch_indices = torch.tensor(batch_indices, dtype=torch.long, device=node_features.device)


            # *** No more node_vad_flags in the features => removed to avoid time leakage ***

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
            start_idx = 0
            for num_nodes in num_nodes_per_sample:
                end_idx = start_idx + num_nodes
                sample_node_features = node_features[start_idx:end_idx]
                sample_node_labels = node_labels[start_idx:end_idx]
                sample_chunk_times = chunk_times_all[start_idx:end_idx]  # 当前样本 chunk time 所处于时间戳信息中；


                # Create edge index
                num_nodes_sample = sample_node_features.size(0)
                if num_nodes_sample > 1:
                    edge_index = torch.stack([
                        torch.arange(num_nodes_sample - 1, dtype=torch.long, device=node_features.device),
                        torch.arange(1, num_nodes_sample, dtype=torch.long, device=node_features.device)
                    ], dim=0)


                    # Generate edge attributes and labels
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

                    edge_labels = torch.tensor(edge_labels, dtype=torch.long, device=device)



                    # Generate edge_features based on connected node features
                    src_node_features = sample_node_features[edge_index[0]]  # [E, feature_dim]
                    dst_node_features = sample_node_features[edge_index[1]]  # [E, feature_dim]
                    combined_node_features = torch.cat([src_node_features, dst_node_features],
                                                       dim=1)  # [E, 2 * feature_dim]
                    edge_features = self.edge_mlp(combined_node_features)  # [E, 3]



                else:
                    edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
                    edge_features = torch.empty((0, 3), dtype=torch.float32, device=device)
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

            # Batch all graphs together
            batch_graph = Batch.from_data_list(all_graphs).to(all_chunks_tensor.device)

            # Node-Level Classification
            module_output = self.node_edge_cls(batch_graph)
            node_predictions, edge_predictions = module_output["node_predictions"], module_output["edge_predictions"]


            subgraph_predictions= module_output["subgraph_predictions"]
            subgraph_labels =  module_output["subgraph_labels"]
            subgraph_node_indices = module_output["subgraph_node_indices"]
            subgraph_node_predictions = module_output["subgraph_node_predictions"]

            # note, Factor Graph Layer for consistency refinement
            node_predictions_refined, edge_predictions_refined = self.factor_graph_layer(node_predictions, edge_predictions, batch_graph)

            # Prepare outputs for the node predictions
            outputs = {
                'node_predictions': node_predictions,
                'node_labels': node_labels,

                'edge_predictions': edge_predictions,
                'edge_labels': torch.cat([g.edge_y for g in all_graphs], dim=0) if len(all_graphs) > 0 else torch.empty( 0, dtype=torch.long, device=device),

                'node_pred_refine': node_predictions_refined,
                'edge_pred_refine': edge_predictions_refined,


                'subgraph_predictions': subgraph_predictions,
                'subgraph_labels': subgraph_labels,

                'subgraph_node_indices': subgraph_node_indices,
                'subgraph_node_predictions': subgraph_node_predictions,

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

