import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
import  numpy as np

class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=2):
        super(GATLayer, self).__init__()
        self.gat = GATConv(in_channels, out_channels, heads=heads, concat=False)

    def forward(self, x, edge_index):
        return self.gat(x, edge_index)


class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GATModel, self).__init__()
        self.gat1 = GATLayer(in_channels, hidden_channels)
        self.gat2 = GATLayer(hidden_channels, out_channels)

    def forward(self, x, edge_index):

        # 该行检索模型参数的设备。使用next(self.parameters())确保获取第一个参数的设备。
        # 如果模型移动到cuda ，这将检索cuda:0 。
        device = next(self.parameters()).device

        x, edge_index = x.to(device), edge_index.to(device)

        x = self.gat1(x, edge_index)
        x = torch.relu(x)
        x = self.gat2(x, edge_index)
        return x


class TemporalAttentionLayer(nn.Module):
    def __init__(self, node_features):
        super(TemporalAttentionLayer, self).__init__()
        self.attention = nn.Linear(node_features, 1)

    def forward(self, node_seq):
        """
        node_seq: [num_nodes, node_features]
        """
        # Dynamically compute num_nodes from input
        num_nodes = node_seq.size(0)

        # Compute attention weights over the time dimension (num_nodes)
        attn_weights = torch.softmax(self.attention(node_seq), dim=0)  # (num_nodes, 1)

        # Apply attention weights to node sequence
        attended_seq = torch.sum(attn_weights * node_seq, dim=0)  # Weighted sum over num_nodes
        return attended_seq


class GNNWithTemporalAttention(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNWithTemporalAttention, self).__init__()
        self.gat_model = GATModel(in_channels, hidden_channels, out_channels)
        self.temporal_attention = TemporalAttentionLayer(out_channels)
        self.audio_classifier = nn.Linear(out_channels, 2)  # Normal/Abnormal classification

    def forward(self, data):
        device = data.x.device

        x, edge_index, batch_indices = data.x.to(device), data.edge_index.to(device), data.batch.to(device)
        num_audios = batch_indices.max().item() + 1  # Get the number of unique audios in the batch

        # Encode all nodes in the batch collectively
        node_embeddings = self.gat_model(x, edge_index)



        # Sample-Level Classification,
        #   将batch 中所有的 Node  重新拆分成每个音频所对应的node,
        #   构建出每个单独样本audio音频的图数据， 从而完成每个音频在 audio 级别的分类预测

        audio_cls = []
        for i in range(num_audios):
            mask = (batch_indices == i)
            cur_audio_nodes = x[mask]

            # Get the global indices of the nodes for the current audio
            cur_nodes_indices = mask.nonzero(as_tuple=False).view(-1)

            # Filter edges that belong to the current audio
            edge_mask = mask[edge_index[0]] & mask[edge_index[1]]  # Only keep edges between current audio nodes
            cur_audio_edge_index0 = edge_index[:, edge_mask]

            # Map global indices to local indices for the current graph
            mapping = {global_idx.item(): local_idx for local_idx, global_idx in enumerate(cur_nodes_indices)}
            cur_audio_edge_index = torch.stack(
                [torch.tensor([mapping[idx.item()] for idx in edge], dtype=torch.long) for edge in
                 cur_audio_edge_index0],
                dim=1
            ).t()

            # Create Data object for GNN
            cur_audio_graph = Data(x=cur_audio_nodes, edge_index=cur_audio_edge_index)
            cur_audio_graph = cur_audio_graph    #.cuda()
            # Apply GAT for spatial graph attention
            cur_audio_node_embeddings = self.gat_model(cur_audio_graph.x, cur_audio_graph.edge_index)  # (num_nodes, embed_dim)

            # Apply temporal attention for record-level prediction
            attended_node_embeddings = self.temporal_attention(cur_audio_node_embeddings)  # (embed_dim)
            audio_type = self.audio_classifier(attended_node_embeddings)
            audio_cls.append(audio_type)

        audio_result = torch.stack(audio_cls, dim=0)
        return node_embeddings, audio_result





from desed_task.nnet.DCNN_v3 import  DynamicFeatureExtractor
class GNNRespiraModel(nn.Module):
    def __init__(self, in_channels , hidden_channels, out_channels, num_classes,
                 n_input_ch,
                 activation="Relu",
                 conv_dropout=0,
                 kernel=[3, 3, 3],
                 pad=[1, 1, 1],
                 stride=[1, 1, 1],
                 # n_filt=[64, 64, 64],
                 n_filt=[16, 64, 128],  # n channels
                 pooling=[(1, 4), (1, 4), (1, 4)],
                 normalization="batch",
                 n_basis_kernels=4,
                 DY_layers=[0, 1, 1, 1, 1, 1, 1],
                 temperature=31,
                 pool_dim='freq',
                 node_fea_dim=512,

                 ):
        super(GNNRespiraModel, self).__init__()

        # Node Feature Generator (CNN)
        self.node_fea_generator = DynamicFeatureExtractor( n_input_ch,
                 activation= activation,
                 conv_dropout= conv_dropout,
                 kernel= kernel,
                 pad= pad,
                 stride=stride,
                 # n_filt=[64, 64, 64],
                 n_filt=n_filt,  # n channels
                 pooling=pooling,
                 normalization=normalization,
                 n_basis_kernels=n_basis_kernels,
                 DY_layers=DY_layers,
                 temperature=temperature,
                 pool_dim=pool_dim,
                 node_fea_dim=node_fea_dim,)  # Define your CNN here

        # GNN with Temporal Attention for Record-Level Prediction
        self.gnn_with_attention = GNNWithTemporalAttention(in_channels, hidden_channels, out_channels)

        # Fully Connected Layer for Node Classification
        self.node_classifier = nn.Linear(out_channels, num_classes)

    # input: batch Node,  Data(x=node_features, edge_index=edge_index, y=node_labels)
    def forward(self, batch_data):
        # Get node embeddings and record-level prediction
        spectrograms = batch_data['spectrograms']  # Shape: (batch_size, channels, max_frames, n_mels)
        frame_labels = batch_data['frame_labels']  # Shape: (batch_size, num_classes, max_frames)
        c_ex_mixtures = batch_data['c_ex_mixtures']
        batch_size = len(spectrograms)

        all_chunks = []
        all_chunk_labels = []
        batch_indices = []
        num_nodes_per_sample = []

        for sample_idx, (spec, labels) in enumerate(zip(spectrograms, frame_labels)):
            n_frames = spec.shape[1]
            num_nodes = 0

            # Define chunk size
            chunk_size = 5  # Adjust as needed

            # Process spectrogram into fixed-size chunks
            for j in range(0, n_frames - chunk_size + 1, chunk_size):
                chunk = spec[:, j:j + chunk_size, :]  # Shape: (channels, chunk_size, n_mels)
                all_chunks.append(chunk)

                # Compute node labels for the chunk
                label_chunk = labels[:, j:j + chunk_size]
                node_label = self.get_node_label(label_chunk)
                all_chunk_labels.append(node_label)

                num_nodes += 1

                # Append the sample index for this node
                batch_indices.append(sample_idx)

            num_nodes_per_sample.append(num_nodes)

        if not all_chunks:
            raise ValueError("No valid chunks generated in the batch.")

        # Stack all chunks and process them together
        all_chunks = [torch.tensor(chunk, dtype=torch.float32) if not isinstance(chunk, torch.Tensor) else chunk for
                      chunk in all_chunks]  # list: num_chunks *(channels, chunk_size, n_mels)
        all_chunks_tensor = torch.stack(all_chunks)  # Shape: (total_chunks, channels, chunk_size, n_mels)

        #  debug here, to watch out the  batch  dimmention;
        node_features = self.node_fea_generator(all_chunks_tensor)  # Output: (total_nodes, feature_dim)

        # Stack node labels
        # node_labels = torch.stack(all_chunk_labels)  # Shape: (total_nodes, num_classes)
        node_labels = torch.tensor(all_chunk_labels)
        #node_labels = torch.tensor(all_chunk_labels, dtype=torch.long)

        # Convert batch_indices to tensor
        batch_indices = torch.tensor(batch_indices, dtype=torch.long, device=node_features.device)

        # Create graphs for each sample
        all_graphs = []
        start_idx = 0
        for num_nodes in num_nodes_per_sample:
            end_idx = start_idx + num_nodes
            sample_node_features = node_features[start_idx:end_idx]
            sample_node_labels = node_labels[start_idx:end_idx]

            # Create edge index
            num_nodes_sample = sample_node_features.size(0)
            if num_nodes_sample > 1:
                edge_index = torch.stack([
                    torch.arange(num_nodes_sample - 1, dtype=torch.long, device=node_features.device),
                    torch.arange(1, num_nodes_sample, dtype=torch.long, device=node_features.device)
                ], dim=0)
            else:
                # Handle the case with only one node
                edge_index = torch.empty((2, 0), dtype=torch.long, device=node_features.device)

            # Create graph data object
            graph_data = Data(x=sample_node_features, edge_index=edge_index, y=sample_node_labels)
            all_graphs.append(graph_data)
            start_idx = end_idx

        # Batch all graphs together
        batch_graph = Batch.from_data_list(all_graphs).to(all_chunks_tensor.device)

        # batch_indices = torch.tensor(batch_indices, dtype=torch.long, device=spectrograms.device)
        # node_labels = torch.cat(all_node_labels, dim=0).to(spectrograms.device)  # Shape: (total_nodes, num_classes)


        node_embeddings, record_predictions = self.gnn_with_attention(batch_graph)

        # Node-Level Classification
        node_predictions = torch.softmax(self.node_classifier(node_embeddings), dim=-1)


        # Prepare outputs
        outputs = {
            'node_predictions': node_predictions,        # Shape: (total_nodes, num_classes)
            'record_predictions': record_predictions,    # Shape: (batch_size, num_classes)
            'node_labels': node_labels,                  # Shape: (total_nodes, num_classes)
            'batch_indices': batch_indices ,              # Shape: (total_nodes,)
            'batch_audio_names':c_ex_mixtures
        }


        return outputs


    def get_node_label(self, frames, normal_label=0, abnormal_threshold=0.2):

        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()


        #frames_label = frames.argmax(dim=0) # Shape: (n_frames,)# This is correct for PyTorch
        frames_label = frames.argmax(axis=0)  # Use 'axis' for NumPy
        counts = np.bincount(frames_label)
        total_frames = len(frames_label)

        if len(counts) <= normal_label:
            return 0

        majority_label = counts.argmax()
        majority_count = counts[majority_label]

        if majority_label == normal_label:
            significant_abnormals = [(label, count) for label, count in enumerate(counts)
                                     if label != normal_label and count > abnormal_threshold * total_frames]

            if significant_abnormals:
                # return sorted(significant_abnormals, key=lambda x: -x[1])[0][0]
                # Return the abnormal label with the most counts
                return max(significant_abnormals, key=lambda x: x[1])[0]
            return normal_label
        else:
            return majority_label



    def generate_record_label(self, node_labels, batch_indices):
        """
        Automatically generate the record-level label based on node-level labels for all samples in the batch.
        If all node labels are normal for a sample, mark the record as normal.
        Otherwise, mark the record as abnormal.
        """
        device = node_labels.device
        batch_indices = batch_indices.to(device)

        normal_label = 0
        num_audios = batch_indices.max().item() + 1  # Get the number of unique audios in the batch
        record_labels = []
        for i in range(num_audios):
            mask = (batch_indices == i)
            mask = mask.to(device)


            cur_node_labels = node_labels[mask]
            if (cur_node_labels == normal_label).all():
                record_labels.append(0)  # Normal
            else:
                record_labels.append(1)  # Abnormal

        return torch.tensor(record_labels, dtype=torch.long)
# Example instantiation:
# model = GNNRespiraModel(in_channels=64, hidden_channels=128, out_channels=128, num_classes=5)
# This model can then be passed to a PyTorch Lightning Trainer.






from torch_geometric.data import Data, Batch
import random

if __name__ == "__main__":
    # Dummy input generation
    batch_size = 4
    in_channels = 256

    random_graphs = []
    for _ in range(batch_size):
        num_nodes = random.randint(5, 15)  # Random number of nodes per graph
        x = torch.randn((num_nodes, in_channels))
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))  # Random edges
        graph = Data(x=x, edge_index=edge_index)
        random_graphs.append(graph)

    batch = Batch.from_data_list(random_graphs)

    # Forward pass through the model
    model = GNNRespiraModel(in_channels=256, hidden_channels=128, out_channels=128, num_classes=7)
    node_predictions, record_predictions = model(batch)
    print("Node Predictions:", node_predictions.size())
    print("Record Predictions:", record_predictions.size())





