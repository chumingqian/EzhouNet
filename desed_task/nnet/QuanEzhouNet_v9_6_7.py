from torch_geometric.nn import GATConv, global_mean_pool
from torch import Tensor
from desed_task.nnet.DCNN_v9_5   import  DynamicFeatureExtractor

# Adjusted GATConv that supports edge attributes (your implementation)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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



#  v9-3-5,对 node embedding  先进行 Linear 转换，
# 对区间定位的特征中加入 start, end; 而区间置信度和分类中的特征只包含local feat,   abnormal feature;
# 并且此时区间定位 和 区间置信度与分类 使用两个分离的头；


# v9-3-12:  调整为双向 Bigru,  并且此时，使用集成头的方式；



import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from brevitas.nn import QuantLinear, QuantIdentity
import torch_geometric.utils as pyg_utils

# Placeholder for DynamicFeatureExtractor (assumed to be CNN-based)
class DynamicFeatureExtractor(nn.Module):
    def __init__(self, n_input_ch, node_fea_dim=256, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(n_input_ch, 16, kernel_size=3, padding=1)
        self.quant_conv = QuantLinear(16 * 25 * 40, node_fea_dim, weight_bit_width=8, bias=False)  # Simplified
    def forward(self, x):
        x = F.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        x = self.quant_conv(x)
        return x

# Degree-aware normalization module
class DegreeNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.5))  # Learnable exponent for degree normalization

    def forward(self, x, edge_index):
        degree = pyg_utils.degree(edge_index[0], num_nodes=x.size(0))
        n = (degree + 1) ** self.a  # Avoid division by zero
        return x / n.unsqueeze(-1)

# Quantized GATConv layer
class QuantGATConv(GATConv):
    def __init__(self, in_channels, out_channels, heads=1, concat=False, edge_dim=32, bit_width=8):
        super().__init__(in_channels, out_channels, heads=heads, concat=concat, edge_dim=edge_dim)
        self.lin = QuantLinear(in_channels, heads * out_channels, bias=False, weight_bit_width=bit_width)
        self.att_quant = QuantIdentity(bit_width=bit_width)  # Quantize attention outputs

    def forward(self, x, edge_index, edge_attr=None):
        out = self.lin(x)
        out = self.propagate(edge_index, x=out, edge_attr=edge_attr)
        return self.att_quant(out), None  # Simplified; attention weights omitted for brevity

class QuantGATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=2, bit_width=8):
        super().__init__()
        self.gat = QuantGATConv(in_channels, out_channels, heads=heads, concat=False, edge_dim=32, bit_width=bit_width)

    def forward(self, x, edge_index, edge_attr):
        out, _ = self.gat(x, edge_index, edge_attr=edge_attr)
        return out, None  # Placeholder for attention weights

class QuantGATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, bit_width=8):
        super().__init__()
        self.norm1 = DegreeNorm()
        self.gat1 = QuantGATLayer(in_channels, hidden_channels, bit_width=bit_width)
        self.norm2 = DegreeNorm()
        self.gat2 = QuantGATLayer(hidden_channels, out_channels, bit_width=bit_width)

    def forward(self, x, edge_index, edge_attr):
        x_norm = self.norm1(x, edge_index)
        x_conf, _ = self.gat1(x_norm, edge_index, edge_attr)
        x2_in = torch.relu(x_conf)
        x2_norm = self.norm2(x2_in, edge_index)
        x_cls, _ = self.gat2(x2_norm, edge_index, edge_attr)
        return x_cls, (None, None)  # Placeholder for attention weights

class Interval_Refine(nn.Module):
    def __init__(self, node_embedding_dim, num_classes=5, num_refine_layers=1, dist_bins_list=[80, 60, 80]):
        super().__init__()
        self.node_embedding_dim = node_embedding_dim
        self.num_classes = num_classes
        self.num_scales = 3
        self.dist_bins = dist_bins_list

        self.local_feat_rnns = nn.ModuleList([
            nn.GRU(node_embedding_dim, node_embedding_dim, batch_first=True, bidirectional=True)
            for _ in range(self.num_scales)
        ])
        self.unify_heads = nn.ModuleList([
            nn.Sequential(
                QuantLinear(2 * node_embedding_dim + 3 + 4, 256, weight_bit_width=8, bias=False),
                nn.ReLU(),
                nn.Dropout(0.2),
                QuantLinear(256, 256, weight_bit_width=8, bias=False),
                nn.ReLU(),
                nn.Dropout(0.2),
                QuantLinear(256, 2 * dist_bins_list[i] + 1 + (num_classes - 1), weight_bit_width=8, bias=False)
            ) for i in range(self.num_scales)
        ])
        self.start_weight_params = nn.ParameterList([
            nn.Parameter(torch.linspace(-20.0, 20.0, dist_bins_list[i]))
            for i in range(self.num_scales)
        ])
        self.end_weight_params = nn.ParameterList([
            nn.Parameter(torch.linspace(-20.0, 20.0, dist_bins_list[i]))
            for i in range(self.num_scales)
        ])

    def forward(self, node_embeddings, time_positions, node_pred, audio_len, **kwargs):
        device = node_embeddings.device
        anchor_intervals = kwargs.get('cur_anchor_intervals', torch.zeros(70, 2, device=device))
        num_intervals_per_scale = kwargs.get('num_intervals_per_scale', [15, 40, 15])

        split_indices = torch.cumsum(torch.tensor(num_intervals_per_scale, device=device), dim=0)
        anchor_intervals_list = torch.split(anchor_intervals, num_intervals_per_scale, dim=0)

        final_bounds_list = []
        interval_conf_logits_list = []
        interval_cls_logits_list = []

        for scale_idx in range(self.num_scales):
            anchor_intervals = anchor_intervals_list[scale_idx]
            num_intervals_scale = num_intervals_per_scale[scale_idx]
            if num_intervals_scale == 0:
                continue

            local_feat_rnn = self.local_feat_rnns[scale_idx]
            start_weight_params = self.start_weight_params[scale_idx]
            end_weight_params = self.end_weight_params[scale_idx]

            x = node_embeddings  # Simplified; temporal encoding omitted
            smoothed_pred = F.softmax(node_pred, dim=-1)
            abnormal_scores = smoothed_pred[:, 0]

            time_pos_1d = time_positions.squeeze(-1) * audio_len.item()
            starts = anchor_intervals[:, 0].clone()
            ends = anchor_intervals[:, 1].clone()
            in_interval_mask = (time_pos_1d.unsqueeze(0) >= starts.unsqueeze(1)) & \
                               (time_pos_1d.unsqueeze(0) <= ends.unsqueeze(1))

            local_x_list = [x[mask] if mask.sum() > 0 else torch.zeros((1, self.node_embedding_dim), device=device)
                            for mask in in_interval_mask]
            sequence_lengths = [lx.size(0) for lx in local_x_list]

            sorted_indices = sorted(range(num_intervals_scale), key=lambda i: sequence_lengths[i], reverse=True)
            sorted_local_x = [local_x_list[i] for i in sorted_indices]
            sorted_lengths = [sequence_lengths[i] for i in sorted_indices]

            max_len = max(sorted_lengths)
            padded_local_x = torch.stack([F.pad(seq, (0, 0, 0, max_len - seq.size(0))) for seq in sorted_local_x])
            packed_local_x = torch.nn.utils.rnn.pack_padded_sequence(padded_local_x, lengths=sorted_lengths, batch_first=True)
            _, h_local = local_feat_rnn(packed_local_x)

            unsort_indices = torch.argsort(torch.tensor(sorted_indices, device=device))
            h_local = h_local[:, unsort_indices, :]
            local_feat = torch.cat([h_local[0], h_local[1]], dim=1)

            shared_feature = torch.cat([local_feat, abnormal_scores[:num_intervals_scale].unsqueeze(-1),
                                        (starts + ends) / 2 / audio_len.item(), (ends - starts) / audio_len.item()], dim=1)
            unify_output = self.unify_heads[scale_idx](shared_feature)

            start_logits = unify_output[:, :self.dist_bins[scale_idx]]
            end_logits = unify_output[:, self.dist_bins[scale_idx]:2 * self.dist_bins[scale_idx]]
            interval_conf_logits_scale = unify_output[:, 2 * self.dist_bins[scale_idx]]
            interval_cls_logits_scale = unify_output[:, 2 * self.dist_bins[scale_idx] + 1:]

            start_offsets = (F.softmax(start_logits, dim=1) * start_weight_params).sum(dim=1)
            end_offsets = (F.softmax(end_logits, dim=1) * end_weight_params).sum(dim=1)

            starts = torch.clamp(starts + start_offsets, 0, audio_len.item())
            ends = torch.clamp(ends + end_offsets, 0, audio_len.item())

            final_bounds_list.append(torch.stack([starts, ends], dim=1))
            interval_conf_logits_list.append(interval_conf_logits_scale)
            interval_cls_logits_list.append(interval_cls_logits_scale)

        return {
            "final_bounds": torch.cat(final_bounds_list, dim=0),
            "interval_conf_logits": torch.cat(interval_conf_logits_list, dim=0),
            "interval_cls_logits": torch.cat(interval_cls_logits_list, dim=0),
            "distill_loss": torch.tensor(0.0, device=device)
        }

class Node_Interval_cls_Module(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_feature_dim, node_num_classes):
        super().__init__()
        self.gat_model = QuantGATModel(in_channels, hidden_channels, out_channels)
        self.node_class_heads = nn.Sequential(
            QuantLinear(out_channels + 4, 64, weight_bit_width=8, bias=False),
            nn.ReLU(),
            nn.Dropout(0.2),
            QuantLinear(64, node_num_classes, weight_bit_width=8, bias=False)
        )
        self.interval_refine = Interval_Refine(out_channels)

    def forward(self, data, audio_dur, anchor_intervals, fs, frame_hop, bt_resp_loc):
        x, edge_index, edge_attr, batch_indices = data.x, data.edge_index, data.edge_attr, data.batch
        node_embeddings, _ = self.gat_model(x, edge_index, edge_attr)
        node_emb_w_loc = torch.cat([node_embeddings, bt_resp_loc], dim=1)
        node_preds = self.node_class_heads(node_emb_w_loc)

        num_audios = batch_indices.max().item() + 1
        intervals_list, interval_conf_list, interval_cls_list = [], [], []

        for i in range(num_audios):
            mask = (batch_indices == i)
            cur_node_embeddings = node_embeddings[mask]
            cur_node_preds = node_preds[mask]
            cur_duration = torch.tensor(audio_dur[i], dtype=torch.float)
            num_frames = int(cur_duration * fs / frame_hop)
            num_nodes = mask.sum().item()
            time_positions = (torch.arange(num_nodes, device=x.device) * 5 + 12.5) / num_frames
            time_positions = time_positions.unsqueeze(-1)

            event_output = self.interval_refine(
                cur_node_embeddings, time_positions, cur_node_preds, cur_duration,
                cur_anchor_intervals=anchor_intervals[i], num_intervals_per_scale=[15, 40, 15]
            )
            intervals_list.append(event_output["final_bounds"])
            interval_conf_list.append(event_output["interval_conf_logits"])
            interval_cls_list.append(event_output["interval_cls_logits"])

        return [node_preds, intervals_list, interval_conf_list, interval_cls_list, torch.tensor(0.0, device=x.device), None, None]

class GraphRespiratory(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes, n_input_ch, node_fea_dim=256, fs=8000, frame_hop=128):
        super().__init__()
        self.fs, self.frame_hop = fs, frame_hop
        self.node_fea_generator = DynamicFeatureExtractor(n_input_ch, node_fea_dim=node_fea_dim)
        self.node_interval_pred = Node_Interval_cls_Module(node_fea_dim, hidden_channels, out_channels, 32, num_classes)

    def forward(self, batch__, level):
        if level != "node":
            return {}
        batch_data = batch__  # Assuming batch__ is a dict
        spectrograms, audio_durations, anchor_intervals = batch_data['spectrograms'], batch_data['audio_dur'], batch_data["anchor_intervals"]
        all_chunks = [torch.tensor(spec[:, :25, :], dtype=torch.float32) for spec in spectrograms for _ in range(1)]  # Simplified
        all_chunks_tensor = torch.stack(all_chunks)
        node_features = self.node_fea_generator(all_chunks_tensor)
        batch_indices = torch.zeros(len(all_chunks), dtype=torch.long, device=node_features.device)
        graph_data = Data(x=node_features, edge_index=torch.empty((2, 0), dtype=torch.long), edge_attr=torch.empty((0, 32)))
        graph_data.batch = batch_indices
        bt_resp_loc = torch.zeros(len(all_chunks), 4, device=node_features.device)  # Dummy location
        pred_output = self.node_interval_pred(graph_data, audio_durations, anchor_intervals, self.fs, self.frame_hop, bt_resp_loc)
        return {
            "node_predictions": pred_output[0],
            "pred_intervals": pred_output[1],
            "pred_intervals_conf_logits": pred_output[2],
            "pred_intervals_cls_logits": pred_output[3],
            "distill_loss": pred_output[4]
        }

if __name__ == "__main__":
    model = GraphRespiratory(1, 128, 128, 5, 1)
    batch_data = {
        "spectrograms": [torch.randn(1, 100, 40)] * 2,
        "audio_dur": [10.0, 10.0],
        "anchor_intervals": [torch.zeros(70, 2)] * 2
    }
    outputs = model(batch_data, "node")
    print(outputs["node_predictions"].shape)


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
                 include_location=  True,  # note,  put on  the triger
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

        self.node_edge_proj =  nn.Sequential( nn.Linear(node_fea_dim, 128),
                                            # Example hidden layer
                                             nn.ReLU(),
                                             nn.Dropout(0.5),
                                             nn.Linear(128, 16)
                                                )


        # Edge feature encoder: from raw edge_attr (scalar) to learnable embedding
        edge_feature_dim = 32 # if you change this , the GAT layer,edge_dim=2 should also be changed ;
        self.edge_encoder = nn.Linear(2*16, edge_feature_dim) # *2, due to  combine two edge_proj
        self.node_interval_pred = Node_Interval_cls_Module(node_fea_dim, hidden_channels, out_channels, edge_feature_dim, node_num_classes=num_classes)


        # abnormal_ratio = 0.25  # , this will  be change  according to the  frams stride
        # pos_weight_node = (1 - abnormal_ratio) / abnormal_ratio
        # pos_weight_node_tensor = torch.tensor(pos_weight_node, dtype=torch.float32)
        # self.node_conf_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_node_tensor)
        #


        # interval_abnormal_ratio = 0.3  # , this will  be change  according to the  frams stride
        # pos_weight_it = (1 - interval_abnormal_ratio) / interval_abnormal_ratio
        # pos_weight_it_tensor = torch.tensor(pos_weight_it, dtype=torch.float32)
        # self.interval_conf_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_it_tensor)




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

                # note,  change frames  in  each  group;
                chunk_size = 25
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

            if self.include_gender:
                node_gender_tensor = torch.tensor(node_gender_all, dtype=torch.long, device=node_features.device)
                node_gender_one_hot = F.one_hot(node_gender_tensor, num_classes=2).float()
                # node_features = torch.cat([node_features, node_gender_one_hot], dim=1)

            if self.include_location:
                node_location_all = [t.item() for t in node_location_all]
                node_location_tensor = torch.tensor(node_location_all, dtype=torch.long, device=node_features.device)
                node_location_one_hot = F.one_hot(node_location_tensor, num_classes=4).float()
                # node_features = torch.cat([node_features, node_location_one_hot], dim=1)

            node_edge_fea = self.node_edge_proj(node_features)


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
                sample_node_edge_fea = node_edge_fea[start_idx:end_idx]

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

            batch_graph = Batch.from_data_list(all_graphs).to(all_chunks_tensor.device)
            pred_output = self.node_interval_pred(batch_graph, audio_durations, anchor_intervals, self.fs, self.frame_hop, node_location_one_hot)

            node_predictions = pred_output[0]
            pred_intervals = pred_output[1]
            pred_interval_conf = pred_output[2]
            pred_interval_cls = pred_output[3]
            distill_loss = pred_output[4]
            edge_preds = pred_output[5]
            bt_edge_labels =pred_output[6]


            # print(f"node_predictions shape: {node_predictions.shape}")  # Should output [sum(N_i), num_classes]

            # Prepare outputs for the node predictions
            outputs = {
                'edge_labels': bt_edge_labels.float(),
                'edge_preds':  edge_preds,

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

