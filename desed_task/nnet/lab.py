def forward(self, batch_data, level):
    device = next(self.parameters()).device

    if level == "node":
        spectrograms = batch_data['spectrograms']
        frame_labels = batch_data['frame_labels']
        c_ex_mixtures = batch_data['c_ex_mixtures']
        vad_timestamps = batch_data['vad_timestamps']
        genders = batch_data['genders']
        locations = batch_data['chest_loc']
        audio_durations = batch_data['audio_dur']
        anchor_intervals = batch_data["anchor_intervals"]

        all_chunks = []
        all_chunk_type_labels = []
        all_chunk_confidences = []
        batch_indices = []
        num_nodes_per_sample = []
        chunk_times_all = []

        # Process chunks
        for sample_idx, (spec, labels, vad_intervals, audio_dur) in enumerate(
                zip(spectrograms, frame_labels, vad_timestamps, audio_durations)):
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
            stride = 5
            chunk_times = []
            for j in range(0, n_frames - chunk_size + 1, stride):
                chunk = spec[:, j:j + chunk_size, :]
                all_chunks.append(chunk)
                label_chunk = labels[j:j + chunk_size, :]
                confidence, type_label = self.get_node_labels(label_chunk, chunk_size)
                all_chunk_confidences.append(confidence)
                all_chunk_type_labels.append(type_label)

                chunk_start_time = frame_times[j]
                chunk_end_time = frame_times[j + chunk_size - 1]
                chunk_times.append((chunk_start_time, chunk_end_time))
                num_nodes += 1
                batch_indices.append(sample_idx)

            # Handle last chunk
            last_chunk_end = (n_frames - chunk_size + 1 - stride)
            if n_frames - chunk_size > last_chunk_end and n_frames % stride != 0:
                chunk = spec[:, n_frames - chunk_size:, :]
                all_chunks.append(chunk)
                label_chunk = labels[n_frames - chunk_size:, :]
                confidence, type_label = self.get_node_labels(label_chunk, chunk_size)
                all_chunk_confidences.append(confidence)
                all_chunk_type_labels.append(type_label)

                chunk_start_time = frame_times[n_frames - chunk_size]
                chunk_end_time = frame_times[n_frames - 1]
                chunk_times.append((chunk_start_time, chunk_end_time))
                num_nodes += 1
                batch_indices.append(sample_idx)

            num_nodes_per_sample.append(num_nodes)
            chunk_times_all.extend(chunk_times)

        if not all_chunks:
            raise ValueError("No valid chunks generated in the batch.")

        all_chunks_tensor = torch.stack(
            [torch.tensor(chunk, dtype=torch.float32) if not isinstance(chunk, torch.Tensor) else chunk for chunk in
             all_chunks])
        node_features = self.node_fea_generator(all_chunks_tensor)  # [total_nodes, node_fea_dim]

        # Construct graphs with virtual nodes
        all_graphs = []
        start_idx = 0
        for sample_idx, num_nodes in enumerate(num_nodes_per_sample):
            end_idx = start_idx + num_nodes
            sample_node_features = node_features[start_idx:end_idx]
            sample_node_type_labels = torch.tensor(all_chunk_type_labels[start_idx:end_idx], dtype=torch.long,
                                                   device=device)
            sample_chunk_times = chunk_times_all[start_idx:end_idx]

            # Global features for virtual node
            global_fea_parts = []
            if self.include_gender:
                gender = genders[sample_idx]
                gender_one_hot = F.one_hot(torch.tensor(gender, device=device), num_classes=2).float()
                global_fea_parts.append(gender_one_hot)
            if self.include_location:
                location = locations[sample_idx]
                location_one_hot = F.one_hot(torch.tensor(location, device=device), num_classes=4).float()
                global_fea_parts.append(location_one_hot)
            if global_fea_parts:
                global_fea = torch.cat(global_fea_parts, dim=0)
                virtual_node_fea = self.global_fea_proj(global_fea).unsqueeze(0)  # [1, node_fea_dim]
                sample_x = torch.cat([sample_node_features, virtual_node_fea], dim=0)  # [num_nodes + 1, node_fea_dim]
                virtual_node_idx = num_nodes
                sample_y = torch.cat([sample_node_type_labels, torch.tensor([-1], dtype=torch.long, device=device)])
            else:
                sample_x = sample_node_features
                virtual_node_idx = None
                sample_y = sample_node_type_labels

            # Edge construction
            if num_nodes > 1:
                original_edge_index = torch.stack([
                    torch.arange(num_nodes - 1, dtype=torch.long, device=device),
                    torch.arange(1, num_nodes, dtype=torch.long, device=device)
                ], dim=0)
                edge_labels = [(sample_node_type_labels[src] != -1 or sample_node_type_labels[dst] != -1)
                               for src, dst in zip(original_edge_index[0], original_edge_index[1])]
                edge_labels = torch.tensor(edge_labels, dtype=torch.long, device=device).long()
            else:
                original_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
                edge_labels = torch.empty((0,), dtype=torch.long, device=device)

            if virtual_node_idx is not None:
                edges_from_virtual = torch.stack([
                    torch.full((num_nodes,), virtual_node_idx, dtype=torch.long, device=device),
                    torch.arange(num_nodes, dtype=torch.long, device=device)
                ], dim=0)
                edges_to_virtual = torch.stack([
                    torch.arange(num_nodes, dtype=torch.long, device=device),
                    torch.full((num_nodes,), virtual_node_idx, dtype=torch.long, device=device)
                ], dim=0)
                edge_index = torch.cat([original_edge_index, edges_from_virtual, edges_to_virtual], dim=1)
            else:
                edge_index = original_edge_index

            # Edge features
            sample_node_edge_fea = self.node_edge_proj(sample_x)
            src_edge_fea = sample_node_edge_fea[edge_index[0]]
            dst_edge_fea = sample_node_edge_fea[edge_index[1]]
            combined_edge_features = torch.cat([src_edge_fea, dst_edge_fea], dim=1)
            edge_features = self.edge_encoder(combined_edge_features)

            graph_data = Data(x=sample_x, edge_index=edge_index, edge_attr=edge_features, y=sample_y)
            graph_data.edge_y = edge_labels  # Only for original edges
            all_graphs.append(graph_data)
            start_idx = end_idx

        batch_graph = Batch.from_data_list(all_graphs).to(device)
        pred_output = self.node_interval_pred(batch_graph, audio_durations, anchor_intervals, self.fs, self.frame_hop)

        # Outputs remain largely the same
        node_predictions, pred_intervals, pred_interval_conf, pred_interval_cls, distill_loss = pred_output
        batch_indices = torch.tensor(batch_indices, dtype=torch.long, device=device)

        outputs = {
            'node_predictions': node_predictions,
            'node_type_labels': torch.tensor(all_chunk_type_labels, dtype=torch.long, device=device),
            'node_confidences': torch.tensor(all_chunk_confidences, dtype=torch.float, device=device),
            'pred_intervals': pred_intervals,
            'pred_intervals_conf_logits': pred_interval_conf,
            'pred_intervals_cls_logits': pred_interval_cls,
            'distill_loss': distill_loss,
            'batch_edge_index': batch_graph.edge_index,
            'batch_graph': batch_graph,
            'batch_indices': batch_indices,
            'batch_audio_names': c_ex_mixtures
        }

    return outputs