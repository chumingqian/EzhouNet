

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


# v6-5-4, 将vad 的时间戳信息编码到 node 的特征中，以及图的边缘属性中；
# v6-5-5, 增加采集时的胸腔位置信息，以及性别信息；；

# v7-1-1,   开始使用边的预测信息
# v7-1-6,   vad  作为边的属性， 但是不加入到 node feature 中；


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
    def __init__(self, in_channels, hidden_channels, out_channels, node_num_classes, edge_num_classes=2):
        super(Node_Edge_cls_Module, self).__init__()
        self.gat_model = GATModel(in_channels, hidden_channels, out_channels)
        # Fully Connected Layer for Node Classification
        self.node_classifier = nn.Linear(out_channels, node_num_classes)
        # Fully Connected Layer for Edge Classification
        self.edge_classifier = nn.Linear(out_channels, edge_num_classes)

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
            node_predictions_list.append(node_pred)
            all_node_labels.append(cur_node_labels)

            # Edge classification
            if cur_graph.edge_index.size(1) > 0:
                src, dst = cur_graph.edge_index
                edge_embeddings = (node_embeddings[src] + node_embeddings[dst]) / 2.0
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

        return node_predictions,  edge_predictions,






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
        num_additional_features = 0  # For node_vad_flags_tensor
        if self.include_gender:
            num_additional_features += 2  # Gender one-hot encoding size
        if self.include_location:
            num_additional_features += 4  # Location one-hot encoding size

        # Adjust in_channels accordingly
        adjusted_in_channels = node_fea_dim + num_additional_features
        # Edge feature encoder: from raw edge_attr (scalar) to learnable embedding
        edge_feature_dim = 2 # if you change this , the GAT layer,edge_dim=2 should also be changed ;
        self.edge_encoder = nn.Linear(1, edge_feature_dim)
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
            node_features = self.node_fea_generator(all_chunks_tensor)  # Output: (total_nodes, feature_dim)

            # Stack node labels
            node_labels = torch.tensor(all_chunk_labels, dtype=torch.long, device=node_features.device)
            # Convert batch_indices to tensor
            batch_indices = torch.tensor(batch_indices, dtype=torch.long, device=node_features.device)

            # Add VAD flag to node features
            node_vad_flags_tensor  = torch.tensor(node_vad_flags_all, dtype=torch.float32, device=node_features.device).unsqueeze(1)
            # node_features = torch.cat([node_features, node_vad_flags_tensor], dim=1)

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
            node_offset = 0
            for num_nodes in num_nodes_per_sample:
                end_idx = start_idx + num_nodes
                sample_node_features = node_features[start_idx:end_idx]
                sample_node_labels = node_labels[start_idx:end_idx]      # 代表当前样本上每个节点的标签；
                sample_chunk_times = chunk_times_all[start_idx:end_idx]  # 每个节点，所对应的5帧，所处于的时间戳区间，
                sample_node_vad_flags = node_vad_flags_all[start_idx:end_idx] # 该节点是否处于真实的 vad 区间中；

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
                        edge_attr_val = float(edge_within_vad)
                        edge_attrs.append([edge_attr_val])  #  这里边的属性不能依靠 时间戳信息生成；

                    edge_attr = torch.tensor(edge_attrs, dtype=torch.float32, device=device)
                    edge_labels = torch.tensor(edge_labels, dtype=torch.long, device=device)

                    # Encode edge attributes to learnable embeddings
                    edge_features = self.edge_encoder(edge_attr)  # 这里边的属性，直接使用一个可学习参数；

                else:
                    edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
                    edge_attr = torch.empty((0, 2), dtype=torch.float32, device=device)
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
            node_predictions, edge_predictions = self.node_edge_cls(batch_graph)
            # note, Factor Graph Layer for consistency refinement
            node_predictions_refined, edge_predictions_refined = self.factor_graph_layer(node_predictions, edge_predictions, batch_graph)

            # Prepare outputs for the node predictions
            outputs = {
                'node_predictions': node_predictions,
                'node_labels': node_labels,

                'edge_predictions': edge_predictions, # note, it's wrong should be edge_predictions ;
                'edge_labels': torch.cat([g.edge_y for g in all_graphs], dim=0) if len(all_graphs) > 0 else torch.empty( 0, dtype=torch.long, device=device),

                'node_pred_refine': node_predictions_refined,
                'edge_pred_refine': edge_predictions_refined,

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

