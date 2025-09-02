import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
import  numpy as np


#  v6, 双阶段任务，带预选机制的声音事件检测系统，  使用2分类，
#  并且引入模型预判机制， 只有被模型预测为异常类型的样本才会进入到下一个阶段，进行检测任务。


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

        x = self.gat1(x, edge_index)   #(cur_sample_groups=3 or 5, fea_dim )
        x = torch.relu(x)
        x = self.gat2(x, edge_index)
        return x


class AttentionTemporalLayer(nn.Module):
    def __init__(self, node_features):
        super(AttentionTemporalLayer, self).__init__()
        self.attention = nn.Linear(node_features, 1)

    def forward(self, node_seq):
        """
        node_seq: [num_groups= 5 or 3, node_features=256]
        """
        # Dynamically compute num_nodes from input
        num_nodes = node_seq.size(0)

        # Compute attention weights over the time dimension (num_nodes)
        attn_weights = torch.softmax(self.attention(node_seq), dim=0)  # (num_nodes, 1)

        # Apply attention weights to node sequence
        attended_seq = torch.sum(attn_weights * node_seq, dim=0)  # Weighted sum over num_nodes
        return attended_seq


class Node_cls_Module(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes):
        super(Node_cls_Module, self).__init__()
        self.gat_model = GATModel(in_channels, hidden_channels, out_channels)


        # Fully Connected Layer for Node Classification
        # node 节点类型总共有7 种， 1 normal + 6 abnormal;
        self.node_classifier = nn.Linear(out_channels, num_classes)


    def forward(self, data):
        device = data.x.device

        x, edge_index, batch_indices = data.x.to(device), data.edge_index.to(device), data.batch.to(device)
        num_audios = batch_indices.max().item() + 1  # Get the number of unique audios in the batch

        # Encode all nodes in the batch collectively
        node_embeddings = self.gat_model(x, edge_index)
        node_pred =  self.node_classifier(node_embeddings)


        return node_pred



class BinaryModule(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(BinaryModule, self).__init__()

        self.audio_gat_model = GATModel(in_channels, hidden_channels, out_channels)
        self.temporal_attention = AttentionTemporalLayer(out_channels)
        self.audio_classifier = nn.Linear(out_channels, 2)  # Normal/Abnormal classification
        # self.audio_classifier = nn.Linear(out_channels, 1)


    def forward(self, data):
        device = data.x.device

        x, edge_index, batch_indices = data.x.to(device), data.edge_index.to(device), data.batch.to(device)
        num_audios = batch_indices.max().item() + 1  # Get the number of unique audios in the batch

        #   Sample-Level Classification,
        #   将batch 中所有的Node 重新拆分成每个音频所对应的node,
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
            cur_audio_node_embeddings = self.audio_gat_model(cur_audio_graph.x, cur_audio_graph.edge_index)  # (num_groups_cur_sample, embed_dim= out channels)

            # Apply temporal attention for record-level prediction
            attended_node_embeddings = self.temporal_attention(cur_audio_node_embeddings)  #  (num_groups_cur_sample, embed_dim= out channels) -->  (embed_dim)
            audio_type = self.audio_classifier(attended_node_embeddings)
            audio_cls.append(audio_type)

        audio_result = torch.stack(audio_cls, dim=0)
        return  audio_result




from desed_task.nnet.DCNN_v3 import  DynamicFeatureExtractor
class GraphRespiratory(nn.Module):
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
                 group_fea_dim = 768,
                 node_fea_dim=512,
                 ):

        super(GraphRespiratory, self).__init__()
        # Node Feature Generator (CNN)

        self.group_fea_generator = DynamicFeatureExtractor( n_input_ch,
                 activation= "cg",
                 conv_dropout= conv_dropout,
                 kernel= [11, 7, 5,  3, 3, 3],
                 pad=    [1, 1, 1,  1, 1, 1],
                 stride= [2, 1, 1,  1, 1, 1],
                 # n_filt=[64, 64, 64],
                 n_filt= [ 16, 32, 64,  128, 256, 512] ,  # n channels
                 pooling= [ [ 2, 2 ], [ 2, 1 ], [ 2, 2 ], [ 2, 1 ], [ 2, 2 ], [ 2, 1 ] ],
                 normalization=normalization,
                 n_basis_kernels=n_basis_kernels,
                 DY_layers=DY_layers,
                 temperature=temperature,
                 pool_dim=pool_dim,
                 stage = "class",
                 node_fea_dim = group_fea_dim,
                )  # Define your CNN here


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
                 stage = "detect",
                 node_fea_dim=node_fea_dim,)  # Define your CNN here



        self.binary_cls_module = BinaryModule(group_fea_dim, hidden_channels, out_channels)
        # GNN with Temporal Attention for Record-Level Prediction
        self.node_cls_module = Node_cls_Module(in_channels, hidden_channels, out_channels,num_classes=num_classes)





    # input: batch Node,  Data(x=node_features, edge_index=edge_index, y=node_labels)
    def forward(self, batch_data, stage):
        #  将分类和检测的分段 功能直接在 forward 中实现，
        # 1. 首先通过分类模块， 对当前的 batch 的所有样本进行分类；

        device = next(self.parameters()).device

        if stage == "classification":
            # Perform record-level binary classification using graph data
            spectrograms = batch_data['spectrograms']  # List of spectrogram tensors
            c_ex_mixtures = batch_data['c_ex_mixtures']
            batch_size = len(spectrograms)

            all_chunks = []
            all_chunk_labels = []  # For classification, these can be dummy labels or ignored
            batch_group_indice= []
            num_groups_per_sample = []

            for sample_idx, spec in enumerate(spectrograms):
                n_frames = spec.shape[1]
                num_groups = 0

                # Define chunk size
                chunk_size =  186  # about  close to 3 seconds;
                # so, 560 frames, 937 frames  will  gen three  or five  groups;

                # Process spectrogram into chunks
                for j in range(0, n_frames - chunk_size + 1, chunk_size):
                    chunk = spec[:, j:j + chunk_size, :]  # Shape: (channels, chunk_size, n_mels)
                    all_chunks.append(chunk)
                    num_groups += 1
                    # Append the sample index for this node
                    batch_group_indice.append(sample_idx)

                num_groups_per_sample.append(num_groups)

            if not all_chunks:
                raise ValueError("No valid chunks generated in the batch.")

            # Stack all chunks and process them together
            all_chunks_tensor = torch.stack(all_chunks)  # Shape: (total_batch_groups, channels, chunk_size, n_mels)
            all_chunks_tensor = all_chunks_tensor.to(device)


            # Generate node features
            group_features = self.group_fea_generator(all_chunks_tensor)  # Output: (batch_groups, feature_dim)

            # Convert batch_group_indice  to tensor
            batch_group_indice= torch.tensor(batch_group_indice, dtype=torch.long, device=device)

            # Create graphs for each sample
            all_graphs = []
            start_idx = 0
            for num_groups in num_groups_per_sample:
                end_idx = start_idx + num_groups
                sample_group_features = group_features[start_idx:end_idx]

                # Create edge index
                num_groups_sample = sample_group_features.size(0)
                if num_groups_sample > 1:
                    edge_index = torch.stack([
                        torch.arange(num_groups_sample - 1, dtype=torch.long, device=device),
                        torch.arange(1, num_groups_sample, dtype=torch.long, device=device)
                    ], dim=0)
                else:
                    # Handle the case with only one node
                    edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

                # Create graph data object
                graph_data = Data(x=sample_group_features, edge_index=edge_index)
                all_graphs.append(graph_data)
                start_idx = end_idx

            # Batch all graphs together
            batch_group_graph = Batch.from_data_list(all_graphs).to(device)

            # Pass through GNN
            record_binary_predictions = self.binary_cls_module(batch_group_graph)


            outputs = {
                'record_binary_predictions': record_binary_predictions,  # Shape: (batch_size, num_classes)
            }
            return outputs

        elif stage == "detection":
            # ... (existing detection code)

            # 2.  —->  提取出被模型判断为异常的样本,  将他们重新构成一个新的batch , 输入到网络中的检测模块，
            # Get node embeddings and record-level prediction
            spectrograms = batch_data['spectrograms']  # Shape: (batch_size, channels, variable_frames, n_mels)
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

                # Check if the last chunk includes all frames
                last_chunk_end = (n_frames - chunk_size + 1) + chunk_size - 1
                if last_chunk_end < n_frames - 1:
                    # There are leftover frames that were not included
                    # Include the last chunk_size frames as the final chunk
                    chunk = spec[:, n_frames - chunk_size:, :]
                    all_chunks.append(chunk)

                    # Compute node labels for the chunk
                    label_chunk = labels[:, n_frames - chunk_size:]
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
                          chunk in all_chunks]  # list: num_chunks * (channels, chunk_size, n_mels)
            all_chunks_tensor = torch.stack(all_chunks)  # Shape: (total_chunks, channels, chunk_size, n_mels)

            #  debug here, to watch out the  batch  dimmention;
            node_features = self.node_fea_generator(all_chunks_tensor)  # Output: (total_nodes, feature_dim)

            # Stack node labels
            node_labels = torch.tensor(all_chunk_labels, dtype=torch.long, device=node_features.device)
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



            # Node-Level Classification
            # node_predictions = torch.softmax(self.node_classifier(node_embeddings), dim=-1) # Shape: (total_nodes, num_classes)
            node_predictions = self.node_cls_module(batch_graph)  # Shape: (total_nodes, num_classes)

            # Prepare outputs for the  node  predictions
            outputs = {
                'node_predictions': node_predictions,
                'node_labels': node_labels,
                'batch_indices': batch_indices,
                'batch_audio_names': c_ex_mixtures  # This should correspond to the samples in detection
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



    # def generate_record_label(self, node_labels, batch_indices):
    #     """
    #     Automatically generate the record-level label based on node-level labels for all samples in the batch.
    #     If all node labels are normal for a sample, mark the record as normal.
    #     Otherwise, mark the record as abnormal.
    #     """
    #     device = node_labels.device
    #     batch_indices = batch_indices.to(device)
    #
    #     normal_label = 0
    #     num_audios = batch_indices.max().item() + 1  # Get the number of unique audios in the batch
    #     record_labels = []
    #     for i in range(num_audios):
    #         mask = (batch_indices == i)
    #         mask = mask.to(device)
    #
    #
    #         cur_node_labels = node_labels[mask]
    #         if (cur_node_labels == normal_label).all():
    #             record_labels.append(0)  # Normal
    #         else:
    #             record_labels.append(1)  # Abnormal
    #
    #     return torch.tensor(record_labels, dtype=torch.long)
    #



# Example instantiation:
# model = GNNRespiraModel(in_channels=64, hidden_channels=128, out_channels=128, num_classes=5)
# This model can then be passed to a PyTorch Lightning Trainer.






from torch_geometric.data import Data, Batch
import random

if __name__ == "__main__":
    # Dummy input generation
    # batch_size = 4
    # in_channels = 256
    #
    # random_graphs = []
    # for _ in range(batch_size):
    #     num_nodes = random.randint(5, 15)  # Random number of nodes per graph
    #     x = torch.randn((num_nodes, in_channels))
    #     edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))  # Random edges
    #     graph = Data(x=x, edge_index=edge_index)
    #     random_graphs.append(graph)
    #
    # batch = Batch.from_data_list(random_graphs)
    #
    # # Forward pass through the model
    # model =GraphRespiratory(in_channels=256, hidden_channels=128, out_channels=128, num_classes=7)
    # node_predictions, record_predictions = model(batch)
    # print("Node Predictions:", node_predictions.size())
    # print("Record Predictions:", record_predictions.size())



    # Define the batch size and spectrogram dimensions
    batch_size = 4
    channels = 3 # Update based on your model's expected input
    n_frames = 560  # Number of frames in the spectrogram
    n_mels = 84   # Number of Mel frequency bins

    # Create synthetic spectrograms
    spectrograms = [torch.randn(channels, n_frames, n_mels) for _ in range(batch_size)]

    # If there are other tensors expected in the batch data, create them here
    # Example: c_ex_mixtures (if they're just identifiers or additional metadata)
    c_ex_mixtures = ['mixture1', 'mixture2', 'mixture3', 'mixture4']

    # Assemble the batch data dictionary expected by your model
    batch_data = {
        'spectrograms': spectrograms,
        'c_ex_mixtures': c_ex_mixtures
    }

    # Define the stage for which you are debugging
    stage = "classification"

    # Instantiate your model here (assuming it's already defined and named 'model')
    # model = YourModel()
    model =GraphRespiratory(n_input_ch= channels,  in_channels= 768, hidden_channels=256, out_channels=128, num_classes=2)
    # Now you can call the forward function with the synthetic batch data
    outputs = model.forward(batch_data, stage)

    # Print the outputs to verify correctness
    print(outputs)






