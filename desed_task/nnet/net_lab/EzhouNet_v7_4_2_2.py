

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
        self.gat = GATConv(in_channels, out_channels, heads=heads, concat=False, edge_dim=2)

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




# --- The CRF function is reused here ---,   使用 crf 用于对边的预测进行平滑操作；

def graph_crf_inference(unary_potentials, adjacency_matrix, num_iterations=5):
    """
    Perform mean-field inference for graph-based CRF smoothing,
    now for edges.

    Args:
        unary_potentials: Tensor of shape [N, num_classes]
        adjacency_matrix: Tensor of shape [N, N], values indicate adjacency among edges
        num_iterations: Number of iterations of mean-field approximation

    Returns:
        smoothed_potentials: Tensor of shape [N, num_classes]
    """
    N, num_classes = unary_potentials.shape

    # Initialize q with unary
    q = F.softmax(unary_potentials, dim=1)  # (N, num_classes)

    # Normalize adjacency
    degree_matrix = adjacency_matrix.sum(dim=1, keepdim=True).clamp(min=1e-9)
    normalized_adjacency = adjacency_matrix / degree_matrix

    for _ in range(num_iterations):
        # Message passing
        q_message = torch.matmul(normalized_adjacency, q)  # (N, num_classes)

        # Combine unary and pairwise
        q_update = unary_potentials + q_message

        # Softmax
        q = F.softmax(q_update, dim=1)

    return q


def build_edge_adjacency_matrix(edge_index, device='cpu'):
    """
    Build an E×E adjacency matrix for edges that share a node.

    Args:
        edge_index: (2, E) tensor: each column is (src, dst)
        device: which torch device

    Returns:
        adjacency_matrix: (E, E) torch.FloatTensor
    """
    src, dst = edge_index
    E = src.size(0)
    adjacency_matrix = torch.zeros((E, E), dtype=torch.float32, device=device)

    for e1 in range(E):
        for e2 in range(e1 + 1, E):
            # If they share any node
            if (src[e1] == src[e2]) or (src[e1] == dst[e2]) \
               or (dst[e1] == src[e2]) or (dst[e1] == dst[e2]):
                adjacency_matrix[e1, e2] = 1.0
                adjacency_matrix[e2, e1] = 1.0

    return adjacency_matrix


from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# from torch_geometric.utils import connected_components

from torch_sparse import coalesce

import torch

def custom_connected_components(edge_index, num_nodes):
    """
    Identify connected components using Union-Find algorithm.

    Args:
        edge_index (Tensor): Shape [2, E], edge list.
        num_nodes (int): Number of nodes in the graph.

    Returns:
        int: Number of connected components.
        Tensor: Component labels for each node, shape [num_nodes].
    """
    parent = torch.arange(num_nodes, device=edge_index.device)

    def find(x):
        # Path compression
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        x_root = find(x)
        y_root = find(y)
        if x_root != y_root:
            parent[y_root] = x_root

    for edge in edge_index.t():
        union(edge[0], edge[1])

    # Final component labels
    for node in range(num_nodes):
        parent[node] = find(node)

    # Number of unique components
    num_subgraphs = len(torch.unique(parent))

    return num_subgraphs, parent


def build_batched_edge_adjacency(edge_indices, batch_size, device='cpu'):
    """
    Build a batched adjacency matrix for all edges in the batch.

    Args:
        edge_indices: List of (2, E_i) tensors for each graph in the batch.
        batch_size: Number of graphs in the batch.
        device: Torch device.

    Returns:
        batched_adjacency: Sparse tensor of shape (Total_E, Total_E)
    """
    all_adjacencies = []
    offset = 0
    total_nodes = 0  # To compute the correct total number of nodes

    for edge_index in edge_indices:
        E = edge_index.size(1)
        # For each edge, find other edges that share a node
        # Create a (E, E) adjacency
        # Efficiently by mapping nodes to edges
        src, dst = edge_index

        max_node = max(src.max().item(), dst.max().item()) + 1
        total_nodes += max_node

        node_to_edges = {}
        for e in range(E):
            node_to_edges.setdefault(src[e].item() + offset, []).append(e + offset)
            node_to_edges.setdefault(dst[e].item() + offset, []).append(e + offset)

        # Now, for each edge, find its neighbors
        rows = []
        cols = []
        for e in range(offset, offset + E):
            neighbors = node_to_edges.get(src[e - offset].item(), []) + node_to_edges.get(dst[e - offset].item(), [])
            rows.extend([e] * len(neighbors))
            cols.extend(neighbors)

        all_adjacencies.append(torch.tensor([rows, cols], dtype=torch.long, device=device))
        offset += max_node  # Increment offset by the number of nodes in this graph

    print(f"Total nodes: {total_nodes}")


    if all_adjacencies:
        all_adjacencies = torch.cat(all_adjacencies, dim=1)  # (2, Total_E * avg_neighbors)


        print(f"Max row index: {all_adjacencies[0].max().item()}")
        print(f"Max col index: {all_adjacencies[1].max().item()}")
        print(f"Expected total nodes: {total_nodes}")

        # Validate indices against total_nodes
        assert int(all_adjacencies[0].max()) < total_nodes, "Row index out of bounds"
        assert int(all_adjacencies[1].max()) < total_nodes, "Column index out of bounds"

        all_adjacencies, _ = coalesce(
            all_adjacencies,
            torch.ones(all_adjacencies.size(1), device=device),
            m=total_nodes,
            n=total_nodes
        )
    else:
        all_adjacencies = torch.empty((2, 0), dtype=torch.long, device=device)

    # Create a Sparse Tensor
    Total_E = offset
    batched_adjacency = torch.sparse_coo_tensor(
        all_adjacencies,
        torch.ones(all_adjacencies.size(1), device=device),
        (Total_E, Total_E)
    )

    return batched_adjacency


def batched_graph_crf_inference(unary_potentials, batched_adjacency, num_iterations=5):
    q = F.softmax(unary_potentials, dim=1)  # [Total_E, num_classes]

    # Compute degree matrix
    degree = torch.sparse.sum(batched_adjacency, dim=1).to_dense().unsqueeze(1)
    degree = degree.clamp(min=1e-9)
    normalized_adjacency = batched_adjacency.clone().coalesce()
    indices = normalized_adjacency.indices()
    values = normalized_adjacency.values()

    # Normalize values and filter invalid indices
    valid_indices = (indices[0] < q.size(0)) & (indices[1] < q.size(0))
    indices = indices[:, valid_indices]
    values = values[valid_indices] / degree.squeeze(1)[indices[0]]

    # Reconstruct normalized adjacency matrix
    normalized_adjacency = torch.sparse_coo_tensor(
        indices,
        values,
        (q.size(0), q.size(0)),
        device=unary_potentials.device
    )

    for _ in range(num_iterations):
        q_message = torch.sparse.mm(normalized_adjacency, q)
        q_update = unary_potentials + q_message
        q = F.softmax(q_update, dim=1)

    return q



class Node_Edge_cls_Module(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 node_num_classes, edge_num_classes=2, subgraph_num_classes=5,
                 subgraph_hidden_size=128):
        super(Node_Edge_cls_Module, self).__init__()

        self.gat_model = GATModel(in_channels, hidden_channels, out_channels)

        # Fully Connected Layers
        self.node_classifier = nn.Linear(out_channels, node_num_classes)
        self.edge_classifier = nn.Linear(out_channels, edge_num_classes)

        # Subgraph Embedding Layers
        self.subgraph_embed_linear = nn.Linear(out_channels, out_channels)
        self.subgraph_gru = nn.GRU(input_size=out_channels, hidden_size=subgraph_hidden_size, batch_first=True,num_layers=2)
        self.subgraph_classifier = nn.Linear(subgraph_hidden_size, subgraph_num_classes)

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

        subgraph_node_embeddings = []  # List of tensors [num_nodes_in_subgraph, out_channels]
        subgraph_labels = []  # List of tensors [1]
        subgraph_node_indices_list = []
        subgraph_node_predictions_list = []

        # Collect all edge_logits across the batch for batched CRF
        all_edge_logits = []
        edge_graph_offsets = [0]  # Cumulative sum of edges per graph
        edge_indices_per_graph = []


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
            mapping = {g_idx.item(): l_idx for l_idx, g_idx in enumerate(cur_nodes_indices)}
            # Convert global -> local edge indices
            src_local = torch.tensor([mapping[idx.item()] for idx in cur_edge_index0[0]],
                                     dtype=torch.long, device=device)
            dst_local = torch.tensor([mapping[idx.item()] for idx in cur_edge_index0[1]],
                                     dtype=torch.long, device=device)
            cur_edge_index = torch.stack([src_local, dst_local], dim=0)

            # Extract edge labels for the current audio if available
            if edge_labels is not None:
                cur_edge_labels = edge_labels[edge_mask]
            else:
                cur_edge_labels = None

            # Create Data object for the current audio subgraph
            cur_graph = Data(
                x=cur_nodes,
                edge_index=cur_edge_index,
                edge_attr=cur_edge_attr0,
                y=cur_node_labels
            )
            if cur_edge_labels is not None:
                cur_graph.edge_y = cur_edge_labels

            # Apply GAT
            node_embeddings, attn_weights = self.gat_model(cur_graph.x, cur_graph.edge_index, cur_graph.edge_attr)

            # Node classification
            node_logits = self.node_classifier(node_embeddings)
            node_predictions_list.append(node_logits)
            all_node_labels.append(cur_node_labels)

            # Edge classification
            E = cur_graph.edge_index.size(1)
            if E > 0:
                src, dst = cur_graph.edge_index
                edge_embeddings = 0.5 * (node_embeddings[src] + node_embeddings[dst])  # (E, out_channels)
                edge_logits = self.edge_classifier(edge_embeddings)  # (E, 2)
                edge_predictions_list.append(edge_logits)

                # Save GT for edges
                if cur_edge_labels is not None:
                    all_edge_labels.append(cur_edge_labels)

                # Collect edge logits for batched CRF
                all_edge_logits.append(edge_logits)
                edge_graph_offsets.append(edge_graph_offsets[-1] + E)
                edge_indices_per_graph.append(cur_graph.edge_index)
            else:
                # No edges in this graph
                all_edge_logits.append(torch.empty((0, 2), device=device))
                edge_graph_offsets.append(edge_graph_offsets[-1])
                edge_indices_per_graph.append(cur_graph.edge_index)

        # Concatenate all edge_logits
        if all_edge_logits:
            concatenated_edge_logits = torch.cat(all_edge_logits, dim=0)  # [Total_E, 2]
        else:
            concatenated_edge_logits = torch.empty((0, 2), device=device)

        # Build batched adjacency matrix
        batched_adjacency = build_batched_edge_adjacency(edge_indices_per_graph, num_audios, device=device)

        # Perform batched CRF inference
        if concatenated_edge_logits.size(0) > 0:
            smoothed_edge_probs = batched_graph_crf_inference(
                unary_potentials=concatenated_edge_logits,  # [Total_E, 2]
                batched_adjacency=batched_adjacency,  # [Total_E, Total_E]
                num_iterations=5
            )  # [Total_E, 2]
            edge_pred_labels_smoothed = smoothed_edge_probs.argmax(dim=1).cpu().numpy()  # [Total_E]
        else:
            edge_pred_labels_smoothed = np.array([], dtype=int)

        print("Node embeddings from GAT model:")
        print(f"Shape: {node_embeddings.shape}")
        print(f"Max value: {node_embeddings.max().item()}")
        print(f"Min value: {node_embeddings.min().item()}")
        print(f"Any NaN values: {torch.isnan(node_embeddings).any().item()}")

        # Now, assign smoothed edge labels back to each graph
        current_offset = 0
        for i in range(num_audios):
            E = all_edge_logits[i].size(0)
            if E > 0:
                edge_pred_labels_smoothed_i = edge_pred_labels_smoothed[current_offset:current_offset + E]
                current_offset += E
            else:
                edge_pred_labels_smoothed_i = np.array([], dtype=int)

            # Build subgraphs from abnormal edges using PyG connected_components
            # If PyG's connected_components does not support batched operations, process per graph efficiently
            # Here, we process per graph but without using NetworkX

            if E > 0 and len(edge_pred_labels_smoothed_i) > 0:
                abnormal_edge_mask = torch.tensor(edge_pred_labels_smoothed_i == 1, device=device)
                abnormal_edge_indices = torch.nonzero(abnormal_edge_mask, as_tuple=False).squeeze()
                if abnormal_edge_indices.numel() > 0:
                    abnormal_edge_index = edge_indices_per_graph[i][:, abnormal_edge_indices]
                    # Find connected components using PyG
                    num_subgraphs, labels =  custom_connected_components(abnormal_edge_index,
                                                                 num_nodes=x[batch_indices == i].size(0))
                    for sub_id in range(num_subgraphs):
                        sub_nodes_mask = (labels == sub_id)
                        sub_nodes = sub_nodes_mask.nonzero(as_tuple=False).view(-1)
                        sub_nodes_tensor = sub_nodes.to(device)

                        # Get subgraph node embeddings
                        subgraph_node_emb = node_embeddings[sub_nodes_tensor]  # [num_nodes_in_subgraph, out_channels]
                        subgraph_node_embeddings.append(subgraph_node_emb)

                        # Determine subgraph label via majority vote
                        # node_pred_labels = F.softmax(node_logits, dim=1).argmax(dim=1).cpu().numpy()
                        try:
                            # Debug node_logits values
                            print("Inspecting node_logits values:")
                            print(f"Max value: {node_logits.max().item()}")
                            print(f"Min value: {node_logits.min().item()}")
                            print(f"Any NaN values: {torch.isnan(node_logits).any().item()}")
                            print(f"Any Inf values: {torch.isinf(node_logits).any().item()}")

                            # Debug label range
                            print(f"Node label range: min={node_labels.min()}, max={node_labels.max()}")
                            assert node_labels.min() >= 0 and node_labels.max() < node_logits.size(1), \
                                "Node labels are out of range!"

                            # Synchronize device
                            torch.cuda.synchronize()

                            # Normalize logits for stability
                            node_logits = node_logits / 2.0

                            # Compute predictions
                            node_pred_labels = F.softmax(node_logits, dim=1).argmax(dim=1).cpu().numpy()

                        except RuntimeError as e:
                            print(f"node_logits shape: {node_logits.shape}")
                            print(f"node_logits: {node_logits}")
                            raise e

                        node_labels_in_subgraph = node_pred_labels[sub_nodes_tensor.cpu().numpy()]
                        if len(node_labels_in_subgraph) > 0:
                            labels_, counts = np.unique(node_labels_in_subgraph, return_counts=True)
                            subgraph_label = labels_[np.argmax(counts)]
                        else:
                            subgraph_label = 0  # Default to normal if no nodes

                        subgraph_label_tensor = torch.tensor([subgraph_label], dtype=torch.long, device=device)
                        subgraph_labels.append(subgraph_label_tensor)

                        # Store node indices & their logits for consistency loss
                        subgraph_node_indices_list.append(sub_nodes_tensor)
                        subgraph_node_predictions_list.append(node_logits[sub_nodes_tensor])
            else:
                # No abnormal edges => treat the entire sample as one normal subgraph
                n_nodes = x[batch_indices == i].size(0)
                if n_nodes > 0:
                    sub_nodes_tensor = torch.arange(n_nodes, device=device)
                    subgraph_node_emb = node_embeddings[sub_nodes_tensor]  # [n_nodes, out_channels]
                    subgraph_node_embeddings.append(subgraph_node_emb)

                    # Determine subgraph label via majority vote
                    node_pred_labels = F.softmax(node_logits, dim=1).argmax(dim=1).cpu().numpy()
                    node_labels_in_subgraph = node_pred_labels.numpy()
                    if len(node_labels_in_subgraph) > 0:
                        labels_, counts = np.unique(node_labels_in_subgraph, return_counts=True)
                        subgraph_label = labels_[np.argmax(counts)]
                    else:
                        subgraph_label = 0  # Default to normal if no nodes

                    subgraph_label_tensor = torch.tensor([subgraph_label], dtype=torch.long, device=device)
                    subgraph_labels.append(subgraph_label_tensor)

                    # Store node indices & their logits for consistency loss
                    subgraph_node_indices_list.append(sub_nodes_tensor)
                    subgraph_node_predictions_list.append(node_logits[sub_nodes_tensor])

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

        # === Batch Processing for Subgraph Classification ===

        if subgraph_node_embeddings:
            # 1. Collect all subgraph node embeddings and their lengths
            lengths = [emb.size(0) for emb in subgraph_node_embeddings]  # List of num_nodes_in_subgraph

            # Filter out subgraphs with zero nodes
            valid_indices = [i for i, length in enumerate(lengths) if length > 0]
            subgraph_node_embeddings = [subgraph_node_embeddings[i] for i in valid_indices]
            subgraph_labels = [subgraph_labels[i] for i in valid_indices]
            subgraph_node_indices_list = [subgraph_node_indices_list[i] for i in valid_indices]
            subgraph_node_predictions_list = [subgraph_node_predictions_list[i] for i in valid_indices]
            lengths = [lengths[i] for i in valid_indices]

            if len(lengths) > 0:  # Ensure there are still valid subgraphs
                max_length = max(lengths)

                # 2. Pad subgraph node embeddings to max_length
                padded_subgraph_embeddings = []
                for emb in subgraph_node_embeddings:
                    if emb.size(0) < max_length:
                        pad_size = max_length - emb.size(0)
                        padding = torch.zeros((pad_size, emb.size(1)), dtype=emb.dtype, device=device)
                        emb_padded = torch.cat([emb, padding], dim=0)
                    else:
                        emb_padded = emb
                    padded_subgraph_embeddings.append(emb_padded)

                # 3. Stack into [num_subgraphs, max_length, out_channels]
                subgraph_embeddings_stacked = torch.stack(padded_subgraph_embeddings, dim=0)
                lengths_tensor = torch.tensor(lengths, dtype=torch.long, device='cpu')

                # 4. Pack the sequences
                packed_subgraph_embeddings = pack_padded_sequence(
                    subgraph_embeddings_stacked,
                    lengths_tensor,
                    batch_first=True,
                    enforce_sorted=False
                )

            # 5. Pass through GRU
            packed_output, hidden = self.subgraph_gru(packed_subgraph_embeddings)
            # hidden: [num_layers, num_subgraphs, hidden_size]

            # 6. Extract the last hidden state for each subgraph
            # Assuming single-layer GRU
            final_hidden = hidden[-1]  # [num_subgraphs, hidden_size]

            # 7. Pass through subgraph classifier
            subgraph_logits = self.subgraph_classifier(final_hidden)  # [num_subgraphs, subgraph_num_classes]

            # 8. Collect subgraph_logits and labels
            subgraph_predictions = subgraph_logits  # [num_subgraphs, subgraph_num_classes]
            subgraph_labels_tensor = torch.cat(subgraph_labels, dim=0)  # [num_subgraphs]
        else:
            subgraph_predictions = torch.empty((0, self.subgraph_classifier.out_features), device=device)
            subgraph_labels_tensor = torch.empty((0,), dtype=torch.long, device=device)

        outputs = {
            'node_predictions': node_predictions,
            'node_labels': node_labels,

            'edge_predictions': edge_predictions,
            'edge_labels': edge_labels,

            'subgraph_predictions': subgraph_predictions,
            'subgraph_labels': subgraph_labels_tensor,

            'subgraph_node_indices': subgraph_node_indices_list,
            'subgraph_node_predictions': subgraph_node_predictions_list
        }

        return outputs


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
        edge_feature_dim = 2 # if you change this , the GAT layer,edge_dim=2 should also be changed ;
        #self.edge_encoder = nn.Linear(1, edge_feature_dim)
        # Instead of encoding a time-based scalar => define a single learnable vector
        self.learnable_edge_emb = nn.Parameter(torch.randn(edge_feature_dim))

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


                    # Instead of using time-based scalar => use the single learnable vector
                    num_edges_sample = edge_index.size(1)
                    edge_features = self.learnable_edge_emb.unsqueeze(0).repeat(num_edges_sample, 1)



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

