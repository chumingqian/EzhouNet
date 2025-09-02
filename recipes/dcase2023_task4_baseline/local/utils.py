import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import scipy.ndimage
import soundfile
import torch
from desed_task.evaluation.evaluation_measures import compute_sed_eval_metrics
from sed_scores_eval.base_modules.scores import create_score_dataframe
from thop import clever_format, profile



def batched_decode_preds(
    strong_preds,
    filenames,
    encoder,
    thresholds=[0.5],
    median_filter=7,
    pad_indx=None,
):
    """Decode a batch of predictions to dataframes. Each threshold gives a different dataframe and stored in a
    dictionary

    Args:
        strong_preds: torch.Tensor, batch of strong predictions.
        filenames: list, the list of filenames of the current batch.
        encoder: ManyHotEncoder object, object used to decode predictions.
        thresholds: list, the list of thresholds to be used for predictions.
        median_filter: int, the number of frames for which to apply median window (smoothing).
        pad_indx: list, the list of indexes which have been used for padding.

    Returns:
        dict of predictions, each keys is a threshold and the value is the DataFrame of predictions.
    """
    # Init a dataframe per threshold
    scores_raw = {}
    scores_postprocessed = {} # 为原始和处理后的分数准备两个字典(scores_raw, scores_postprocessed)以保存中间结果
    prediction_dfs = {}       # 函数为每个指定的阈值初始化空的预测数据框（储存在prediction_dfs字典中）
    for threshold in thresholds:
        prediction_dfs[threshold] = pd.DataFrame()

    for j in range(strong_preds.shape[0]):  # over batches 对输入批次中的每个预测（通过strong_preds表示）以及相应的文件名进行循环处理。
        audio_id = Path(filenames[j]).stem  # 从当前文件名中提取音频ID，并构造完整的.wav文件名，用于后续构建预测DataFrame。
        filename = audio_id + ".wav"
        c_scores = strong_preds[j]  # (cls=10, frames=156)

        if pad_indx is not None: # 如果提供了pad_indx，说明输入数据可能存在填充，此时会根据pad_indx去除预测得分中的填充部分
            true_len = int(c_scores.shape[-1] * pad_indx[j].item())
            c_scores = c_scores[:true_len]


        #  1. 原始分数
        c_scores = c_scores.transpose(0, 1).detach().cpu().numpy() # (156,10)将预测得分从PyTorch张量转换为NumPy数组，并进行维度转换，便于后续处理。
        scores_raw[audio_id] = create_score_dataframe(
            scores=c_scores, # (156, 10)  调用create_score_dataframe函数，基于时间戳和事件类别标签，将未处理的预测分数转化为数据框，并存入scores_raw字典。
            timestamps=encoder._frame_to_time(np.arange(len(c_scores) + 1)), # ( array[1-156])
            event_classes=encoder.labels, # 10 个标签类
        )

        # 2.  中值滤波处理：对预测得分应用中值滤波，用于平滑分数序列，减少噪声影响。
        #  处理后的分数同样被转换为数据框并存入scores_postprocessed字典。
        c_scores = scipy.ndimage.filters.median_filter(c_scores, (median_filter, 1))
        scores_postprocessed[audio_id] = create_score_dataframe(
            scores=c_scores,
            timestamps=encoder._frame_to_time(np.arange(len(c_scores) + 1)),
            event_classes=encoder.labels,
        )


        # 3. 将处理后的分数通过阈值进一步处理。
        for c_th in thresholds: #根据阈值生成预测：针对每一个预设的阈值：
            pred = c_scores > c_th #应用阈值过滤预测得分，得到布尔矩阵表示预测事件是否发生。
            pred = encoder.decode_strong(pred) #使用encoder.decode_strong方法将布尔矩阵解码为具体的事件标签、起始时间和结束时间，形成预测数据
            pred = pd.DataFrame(pred, columns=["event_label", "onset", "offset"])
            pred["filename"] = filename # 构建包含"event_label", "onset", "offset"和"filename"列的DataFrame，其中"filename"列填充当前处理的音频文件名
            prediction_dfs[c_th] = pd.concat(
                [prediction_dfs[c_th], pred], ignore_index=True
            ) # 将每个阈值对应的预测DataFrame追加到prediction_dfs字典中相应阈值的DataFrame

    return scores_raw, scores_postprocessed, prediction_dfs
    #scores_raw: 原始预测得分转换的数据框集合,
    # scores_postprocessed: 经过中值滤波处理后的预测得分数据框集合,
    # prediction_dfs: 每个阈值对应的一个预测数据框的字典，汇总了所有音频片段的预测结果。
    # 整体上，这个函数实现了预测结果的细致后处理与组织，为后续的分析或评估提供了结构化的数据准备。



def adjust_timestamps(time_intervals):
    # Flatten the array and ensure the last end time is included
    flat_timestamps = time_intervals.ravel()
    if flat_timestamps[-2] != flat_timestamps[-1]:  # Check if last end is different from the second last (start of last frame)
        # Append the last end time
        flat_timestamps = np.append(flat_timestamps, time_intervals[-1, 1])
    return flat_timestamps


def format_timestamps_for_dataframe(time_intervals):
    if time_intervals.ndim != 2 or time_intervals.shape[1] != 2:
        raise ValueError("time_intervals must be a 2D array with shape (N, 2)")

    # Initialize the corrected timestamps list with the first start time
    corrected_timestamps = [time_intervals[0, 0]]

    # Loop through the intervals appending the start of the next interval
    for interval in time_intervals[1:]:
        corrected_timestamps.append(interval[0])

    # Append the end time of the last interval
    corrected_timestamps.append(time_intervals[-1, 1])

    return np.array(corrected_timestamps)



def get_vad_mask(time_intervals, vad_intervals):
    """
    Determines which time intervals overlap with VAD intervals.
    Args:
        time_intervals: numpy array of shape (num_nodes, 2), where each row is [start_time, end_time].
        vad_intervals: List of dicts with 'start' and 'end' times.
    Returns:
        vad_mask: Boolean array indicating whether each time interval overlaps with any VAD interval.
    """
    vad_mask = np.zeros(len(time_intervals), dtype=bool)
    for idx, (start, end) in enumerate(time_intervals):
        for vad_interval in vad_intervals:
            vad_start = vad_interval['start']
            vad_end = vad_interval['end']
            if end > vad_start and start < vad_end:
                vad_mask[idx] = True
                break
    return vad_mask


# def batched_vad_node_decode_preds(
#     strong_preds,
#     filenames,
#     encoder,
#     batch_indices,
#     vad_timestamps,
#     thresholds=[0.5],
#     median_filter=7,
#     frames_per_node=5,
#     batch_dur=None,
#     pad_indx=None,
# ):
#     # Initialize dataframes
#     scores_raw = {}
#     scores_postprocessed = {}
#     prediction_dfs = {}
#     pred_abnormal_events= {}
#     for threshold in thresholds:
#         prediction_dfs[threshold] = pd.DataFrame()
#         pred_abnormal_events[threshold] = pd.DataFrame()
#
#     num_audios = batch_indices.max().item() + 1
#     for i in range(num_audios):
#         mask = (batch_indices == i)
#         cur_node_labels = strong_preds[mask]
#         cur_filenames = filenames[i]
#         vad_intervals = vad_timestamps[i]
#
#         # Get the time intervals for nodes in the current sample
#         time_intervals = encoder.node_to_time_interval(cur_node_labels, frames_per_node, batch_dur[i])
#
#         audio_id = Path(cur_filenames).stem
#         filename = audio_id + ".wav"
#         c_scores = strong_preds[mask].detach().cpu().numpy()  # (num_nodes, num_classes)
#
#         # 1. Apply median filter
#         c_scores = scipy.ndimage.filters.median_filter(c_scores, (median_filter, 1))
#
#         # 2. Filter c_scores and time_intervals to only include intervals within VAD timestamps
#         # Assuming time_intervals is a list of [start_time, end_time] pairs
#         time_intervals_np = np.array(time_intervals)  # Shape: (num_nodes, 2)
#         vad_mask = get_vad_mask(time_intervals_np, vad_intervals)
#
#         # Apply mask
#         c_scores_filtered = c_scores[vad_mask]
#         time_intervals_filtered = time_intervals_np[vad_mask]
#
#         # 3. Create scores_raw and scores_postprocessed using filtered data
#         scores_raw[audio_id] = create_score_dataframe(
#             scores=c_scores_filtered,
#             timestamps=time_intervals_filtered,
#             event_classes=encoder.labels,
#         )
#
#         scores_postprocessed[audio_id] = create_score_dataframe(
#             scores=c_scores_filtered,
#             timestamps=time_intervals_filtered,
#             event_classes=encoder.labels,
#         )
#
#         # 4. Apply threshold to generate predictions
#         for c_th in thresholds:
#             pred = c_scores_filtered > c_th
#             pred = encoder.decode_strong(pred, time_intervals_filtered)
#             pred = pd.DataFrame(pred, columns=["event_label", "onset", "offset"])
#             pred["filename"] = filename
#             prediction_dfs[c_th] = pd.concat(
#                 [prediction_dfs[c_th], pred], ignore_index=True
#             )
#
#             # Filter out events with "Normal" label
#             pred_abnormal = pred[pred["event_label"] != "Normal"]
#             pred_abnormal_events[c_th] = pd.concat(
#                 [pred_abnormal_events[c_th], pred_abnormal], ignore_index=True
#             )
#
#     return scores_raw, scores_postprocessed, prediction_dfs, pred_abnormal_events


def batched_node_decode_preds_v2(
    strong_preds,
    filenames,
    encoder,
    batch_indices,
    thresholds=[0.5],
    median_filter=7,
    frames_per_node =5,
    batch_dur = None,
    pad_indx=None,
):
    # Init a dataframe per threshold
    scores_raw = {}
    scores_postprocessed = {}

    prediction_dfs = {}
    pred_abnormal_events= {}
    for threshold in thresholds:
        prediction_dfs[threshold] = pd.DataFrame()
        pred_abnormal_events[threshold] = pd.DataFrame()


    if batch_indices.numel() == 0:
        # Handle empty batch_indices by returning empty structures
        return scores_raw, scores_postprocessed, prediction_dfs, pred_abnormal_events

    num_audios = batch_indices.max().item() + 1
    for i in range(num_audios):
        mask = (batch_indices == i)
        cur_node_labels = strong_preds[mask]
        cur_filenames = filenames[i]

        # note,  decode  node labes,  Get the time intervals for nodes in the current sample
        time_intervals = encoder.node_to_time_interval(cur_node_labels,frames_per_node, batch_dur[i])

        audio_id = Path(cur_filenames).stem
        filename = audio_id + ".wav"
        c_scores = strong_preds[mask].detach().cpu().numpy()  # (cur_audio_num_nodes, num_classes=7)

        #  1. Original Scores
        scores_raw[audio_id] = create_score_dataframe(
            scores= c_scores,
            timestamps= time_intervals,
            event_classes= encoder.labels,   # check  这里的标签是 真实标签还是 预测标签
        )

        # 2. Median filter processing
        c_scores = scipy.ndimage.filters.median_filter(c_scores, (median_filter, 1))
        scores_postprocessed[audio_id] = create_score_dataframe(
            scores=c_scores,
            timestamps= time_intervals,
            event_classes= encoder.labels,
        )

        # 3. Apply threshold to generate predictions
        for c_th in thresholds:
            pred = c_scores > c_th
            pred = encoder.decode_strong(pred)
            pred = pd.DataFrame(pred, columns=["event_label", "onset", "offset"])
            pred["filename"] = filename
            prediction_dfs[c_th] = pd.concat(
                [prediction_dfs[c_th], pred], ignore_index=True
            )

             # Filter out events with "Normal" label
            pred_abnormal = pred[pred["event_label"]!= "Normal"]
            pred_abnormal_events[c_th] = pd.concat(
                [pred_abnormal_events[c_th], pred_abnormal], ignore_index=True
            )

    return scores_raw, scores_postprocessed, prediction_dfs, pred_abnormal_events



# v2 该版本，引入的是真实标签的 vad timestamps 来进行掩码，
# 用于验证，当引入真实值的时间戳信息后，误报率理论上会降低很多。
def batched_vad_node_decode_preds_v2(
    strong_preds,
    filenames,
    encoder,
    batch_indices,
    vad_timestamps,
    thresholds=[0.5],
    median_filter=7,
    frames_per_node=5,
    batch_dur=None,
    pad_indx=None,
):
    scores_raw = {}
    scores_postprocessed = {}
    prediction_dfs = {}
    pred_abnormal_events= {}
    for threshold in thresholds:
        prediction_dfs[threshold] = pd.DataFrame()
        pred_abnormal_events[threshold] = pd.DataFrame()

    num_audios = batch_indices.max().item() + 1
    for i in range(num_audios):
        mask = (batch_indices == i)
        cur_node_labels = strong_preds[mask]
        cur_filenames = filenames[i]
        vad_intervals = vad_timestamps[i]

        if not vad_intervals:  # Handle empty VAD intervals
            continue  # Skip processing for this audio file

        # Get the time intervals for nodes in the current sample
        time_intervals = encoder.node_to_time_interval_v2(cur_node_labels, frames_per_node, batch_dur[i])

        audio_id = Path(cur_filenames).stem
        filename = audio_id + ".wav"
        c_scores = strong_preds[mask].detach().cpu().numpy()  # (num_nodes, num_classes)

        # 1. Apply median filter
        c_scores = scipy.ndimage.filters.median_filter(c_scores, (median_filter, 1))

        # 2. Filter c_scores and time_intervals to only include intervals within VAD timestamps
        # time_intervals_np = np.array(time_intervals)  # Shape: (num_nodes, 2)

        time_intervals_np = np.array(time_intervals)  # Convert to numpy array if not already
        print("Shape of time_intervals_np:", time_intervals_np.shape)  # Check the shape

        # Ensure it is two-dimensional with 2 columns
        if time_intervals_np.ndim != 2 or time_intervals_np.shape[1] != 2:
            raise ValueError("time_intervals_np must be of shape (num_nodes, 2)")

        vad_mask = get_vad_mask(time_intervals_np, vad_intervals)

        # Apply mask
        c_scores_filtered = c_scores[vad_mask]
        time_intervals_filtered = time_intervals_np[vad_mask]

        # Use the adjusted timestamps
        time_intervals_filtered = np.array(time_intervals_filtered)  # Assuming this is a list of lists or a 2D array
        adjusted_timestamps = format_timestamps_for_dataframe(time_intervals_filtered)

        # 3. Create scores_raw and scores_postprocessed using filtered data
        scores_raw[audio_id] = create_score_dataframe(
            scores=c_scores_filtered,
            timestamps=adjusted_timestamps,
            event_classes=encoder.labels,
        )

        scores_postprocessed[audio_id] = create_score_dataframe(
            scores=c_scores_filtered,
            timestamps=adjusted_timestamps,
            event_classes=encoder.labels,
        )

        # 4. Apply threshold to generate predictions
        for c_th in thresholds:
            pred_th = c_scores_filtered > c_th
            pred_timestamps = encoder.decode_strong(pred_th, batch_dur[i] )
            pred = pd.DataFrame(pred_timestamps, columns=["event_label", "onset", "offset"])
            pred["filename"] = filename
            prediction_dfs[c_th] = pd.concat(
                [prediction_dfs[c_th], pred], ignore_index=True
            )

            # Filter out events with "Normal" label
            pred_abnormal = pred[pred["event_label"] != "Normal"]
            pred_abnormal_events[c_th] = pd.concat(
                [pred_abnormal_events[c_th], pred_abnormal], ignore_index=True
            )

    return scores_raw, scores_postprocessed, prediction_dfs, pred_abnormal_events


def batched_node_edge_decode_preds(
    node_preds,
    edge_preds,
    filenames,
    encoder,
    batch_indices,
    bt_edge_index,
    thresholds=[0.5],
    median_filter=7,
    frames_per_node =5,
    batch_dur = None,
    pad_indx=None,
):
    # Init a dataframe per threshold
    scores_raw = {}
    scores_postprocessed = {}

    prediction_dfs = {}
    pred_abnormal_events= {}
    for threshold in thresholds:
        prediction_dfs[threshold] = pd.DataFrame()
        pred_abnormal_events[threshold] = pd.DataFrame()


    if batch_indices.numel() == 0:
        # Handle empty batch_indices by returning empty structures
        return scores_raw, scores_postprocessed, prediction_dfs, pred_abnormal_events

    num_audios = batch_indices.max().item() + 1
    for i in range(num_audios):

        node_mask = (batch_indices == i)
        cur_node_labels = node_preds[node_mask]
        cur_filenames = os.path.basename( filenames[i])

        # Edge mask: both source and target nodes belong to the current graph
        edge_mask = node_mask[bt_edge_index[0]] & node_mask[bt_edge_index[1]]  # Shape: [num_edges]
        cur_edge_pred = edge_preds[edge_mask]


        # note,  decode  node labes,  Get the time intervals for nodes in the current sample
        time_intervals = encoder.node_to_time_interval(cur_node_labels,frames_per_node, batch_dur[i])
        edge_timestamp_df = generate_edge_timestamp_info( time_intervals,edge_preds=cur_edge_pred ,
                                                          filename= cur_filenames)

        audio_id = Path(cur_filenames).stem
        filename = audio_id + ".wav"
        c_scores = node_preds[node_mask].detach().cpu().numpy()  # (cur_audio_num_nodes, num_classes=7)

        #  1. Original Scores
        scores_raw[audio_id] = create_score_dataframe(
            scores= c_scores,
            timestamps= time_intervals,
            event_classes= encoder.labels,   # check  这里的标签是 真实标签还是 预测标签
        )

        # 2. Median filter processing
        c_scores = scipy.ndimage.filters.median_filter(c_scores, (median_filter, 1))
        scores_postprocessed[audio_id] = create_score_dataframe(
            scores=c_scores,
            timestamps= time_intervals,
            event_classes= encoder.labels,
        )

        # 3. Apply threshold to generate predictions
        for c_th in thresholds:
            pred_th = c_scores > c_th
            pred_timestamps = encoder.decode_strong(pred_th, batch_dur[i])
            pred = pd.DataFrame(pred_timestamps, columns=["event_label", "onset", "offset"])
            pred["filename"] = filename
            prediction_dfs[c_th] = pd.concat(
                [prediction_dfs[c_th], pred], ignore_index=True
            )

             # Filter out events with "Normal" label
            pred_abnormal = pred[pred["event_label"]!= "Normal"]
            pred_abnormal_events[c_th] = pd.concat(
                [pred_abnormal_events[c_th], pred_abnormal], ignore_index=True
            )

        # 4. Refine the edge intervals based on the abnormal node predictions
        refined_edge_intervals = assign_labels_to_edge_intervals_v2(
            pred_abnormal_events=pred_abnormal_events,
            edge_timestamp_df=edge_timestamp_df,
            # thresholds=[0.5,0.3, 0.2, 0.1]
            )

    return  scores_postprocessed, prediction_dfs, pred_abnormal_events, refined_edge_intervals


def filter_pred_abnormal(pred_abnormal, pred_vad_normal):
    """
    Filters pred_abnormal DataFrame by removing or adjusting intervals based on pred_vad_normal.

    Parameters:
        pred_abnormal (pd.DataFrame): DataFrame containing abnormal event predictions.
        pred_vad_normal (pd.DataFrame): DataFrame containing normal VAD predictions.

    Returns:
        pd.DataFrame: Filtered pred_abnormal DataFrame.
    """
    filtered_abnormal = []

    # Ensure that 'onset' and 'offset' are floats
    pred_abnormal = pred_abnormal.copy()
    pred_vad_normal = pred_vad_normal.copy()
    pred_abnormal['onset'] = pred_abnormal['onset'].astype(float)
    pred_abnormal['offset'] = pred_abnormal['offset'].astype(float)
    pred_vad_normal['onset'] = pred_vad_normal['onset'].astype(float)
    pred_vad_normal['offset'] = pred_vad_normal['offset'].astype(float)

    # Process each filename separately
    filenames = pred_abnormal['filename'].unique()

    for filename in filenames:
        # Get abnormal and normal VAD events for the current filename
        ab_events = pred_abnormal[pred_abnormal['filename'] == filename].copy()
        vad_events = pred_vad_normal[pred_vad_normal['filename'] == filename].copy()

        # Sort VAD events by onset
        vad_events = vad_events.sort_values(by='onset').reset_index(drop=True)

        for _, ab_event in ab_events.iterrows():
            ab_onset = ab_event['onset']
            ab_offset = ab_event['offset']
            ab_label = ab_event['event_label']

            # Initialize list to hold remaining intervals after filtering
            remaining_intervals = [(ab_onset, ab_offset)]

            # Iterate over all VAD normal events to apply filtering
            for _, vad_event in vad_events.iterrows():
                vad_onset = vad_event['onset']
                vad_offset = vad_event['offset']

                temp_intervals = []
                for interval in remaining_intervals:
                    current_onset, current_offset = interval

                    # No overlap
                    if vad_offset <= current_onset or vad_onset >= current_offset:
                        temp_intervals.append(interval)
                        continue

                    # Complete overlap: current interval is completely within vad interval
                    if vad_onset <= current_onset and vad_offset >= current_offset:
                        # Entire interval is removed
                        continue

                    # Partial overlaps
                    # Overlap at the beginning
                    if vad_onset <= current_onset < vad_offset < current_offset:
                        new_onset = vad_offset
                        new_offset = current_offset
                        temp_intervals.append((new_onset, new_offset))
                        continue

                    # Overlap at the end
                    if current_onset < vad_onset < current_offset <= vad_offset:
                        new_onset = current_onset
                        new_offset = vad_onset
                        temp_intervals.append((new_onset, new_offset))
                        continue

                    # Overlap in the middle: split into two intervals
                    if current_onset < vad_onset and vad_offset < current_offset:
                        temp_intervals.append((current_onset, vad_onset))
                        temp_intervals.append((vad_offset, current_offset))
                        continue

                remaining_intervals = temp_intervals  # Update the list for next VAD event

                # If no intervals remain, no need to check further
                if not remaining_intervals:
                    break

            # After processing all VAD events, add the remaining intervals to the filtered list
            for interval in remaining_intervals:
                new_event = {
                    'event_label': ab_label,
                    'onset': interval[0],
                    'offset': interval[1],
                    'filename': filename
                }
                filtered_abnormal.append(new_event)

    # Create a new DataFrame from the filtered events
    filtered_abnormal_df = pd.DataFrame(filtered_abnormal)

    # Optionally, reset the index
    filtered_abnormal_df = filtered_abnormal_df.reset_index(drop=True)

    return filtered_abnormal_df

def batched_node_vad_decode_preds(
    node_preds,
    node_vad_preds,
    filenames,
    encoder,
    batch_indices,
    bt_edge_index,
    thresholds=[0.5],
    median_filter=7,
    frames_per_node =5,
    batch_dur = None,
    pad_indx=None,
):
    # Init a dataframe per threshold
    scores_raw = {}
    scores_postprocessed = {}

    prediction_dfs = {}
    pred_abnormal_events= {}


    prediction_vad_dfs = {}
    pred_vad_abnormal_events = {}
    pred_vad_normal_events = {}

    for threshold in thresholds:
        prediction_dfs[threshold] = pd.DataFrame()
        pred_abnormal_events[threshold] = pd.DataFrame()

        prediction_vad_dfs[threshold] = pd.DataFrame()
        pred_vad_abnormal_events[threshold] = pd.DataFrame()
        pred_vad_normal_events[threshold] = pd.DataFrame()


    if batch_indices.numel() == 0:
        # Handle empty batch_indices by returning empty structures
        return scores_raw, scores_postprocessed, prediction_dfs, pred_abnormal_events

    num_audios = batch_indices.max().item() + 1
    for i in range(num_audios):

        node_mask = (batch_indices == i)
        cur_node_labels = node_preds[node_mask]
        cur_filenames = os.path.basename( filenames[i])


        audio_id = Path(cur_filenames).stem
        filename = audio_id + ".wav"
        c_scores = node_preds[node_mask].detach().cpu().numpy()  # (cur_audio_num_nodes, num_classes=7)


        cur_node_vad_pred = node_vad_preds[node_mask]
        cur_node_vad_pred = cur_node_vad_pred.detach().cpu().numpy()  # Shape: (num_nodes,)


        # 2. Identify nodes with predicted label 0 in node_vad_pred
        non_vad_indices = np.where( cur_node_vad_pred == 0)[0]

        c_scores_vad = c_scores.copy()
        val_thread = np.max(c_scores)
        print(f"val_thread (maximum score in c_scores_original): {val_thread}")
        # 3. Modify the first category score to val_thread for these nodes
        c_scores_vad[non_vad_indices, 0] =  val_thread + 0.01


        # note,  decode  node labes,  Get the time intervals for nodes in the current sample
        time_intervals = encoder.node_to_time_interval(cur_node_labels,frames_per_node, batch_dur[i])



        #  1. Original Scores
        scores_raw[audio_id] = create_score_dataframe(
            scores= c_scores,
            timestamps= time_intervals,
            event_classes= encoder.labels,   # check  这里的标签是 真实标签还是 预测标签
        )

        # 2. Median filter processing
        c_scores = scipy.ndimage.filters.median_filter(c_scores, (median_filter, 1))
        c_scores_vad = scipy.ndimage.filters.median_filter(c_scores_vad, (median_filter, 1))


        scores_postprocessed[audio_id] = create_score_dataframe(
            scores=c_scores,
            timestamps= time_intervals,
            event_classes= encoder.labels,
        )

        # 3. Apply threshold to generate predictions
        for c_th in thresholds:
            pred_th = c_scores > c_th
            pred_timestamps = encoder.decode_strong(pred_th, batch_dur[i])
            pred = pd.DataFrame(pred_timestamps, columns=["event_label", "onset", "offset"])
            pred["filename"] = filename
            prediction_dfs[c_th] = pd.concat(
                [prediction_dfs[c_th], pred], ignore_index=True
            )

             # Filter out events with "Normal" label
            pred_abnormal = pred[pred["event_label"]!= "Normal"]
            pred_abnormal_events[c_th] = pd.concat(
                [pred_abnormal_events[c_th], pred_abnormal], ignore_index=True
            )



            pred_vad = c_scores_vad > c_th
            pred_vad_tsp = encoder.decode_strong(pred_vad, batch_dur[i])
            vad_pd = pd.DataFrame(pred_vad_tsp, columns=["event_label", "onset", "offset"])
            vad_pd["filename"] = filename
            prediction_vad_dfs[c_th] = pd.concat(
                [prediction_vad_dfs[c_th], pred], ignore_index=True
            )

             # Find events with "Normal" label
            pred_vad_normal = vad_pd[vad_pd["event_label"] == "Normal"]
            pred_vad_normal_events[c_th] = pd.concat(
                [pred_vad_normal_events[c_th], pred_vad_normal], ignore_index=True
            )

            filtered_vad_pred_abnormal = filter_pred_abnormal(pred_abnormal, pred_vad_normal)
            pred_vad_abnormal_events[c_th] = pd.concat(
                [pred_vad_abnormal_events[c_th], filtered_vad_pred_abnormal], ignore_index=True
            )



    return  scores_postprocessed, prediction_dfs, pred_abnormal_events, pred_vad_abnormal_events






import scipy.ndimage
def generate_edge_intervals(node_intervals):
    """
    Given node time intervals of shape (num_nodes, 2), generate edge intervals.
    Assumes a linear chain of nodes, where edge i connects node i and node i+1.

    Args:
        node_intervals (np.ndarray): shape (num_nodes, 2), each row is [start_time, end_time]

    Returns:
        np.ndarray: shape (num_edges, 2), edge intervals derived from node intervals.
                    edge i interval = [node_intervals[i, 0], node_intervals[i+1, 1]]
    """
    num_nodes = node_intervals.shape[0]
    if num_nodes < 2:
        return np.empty((0, 2))  # No edges if fewer than 2 nodes

    # Create edges by combining consecutive node intervals
    edge_intervals = np.zeros((num_nodes - 1, 2), dtype=node_intervals.dtype)
    for i in range(num_nodes - 1):
        edge_intervals[i, 0] = node_intervals[i, 0]
        edge_intervals[i, 1] = node_intervals[i + 1, 1]
    return edge_intervals


def decode_edge_predictions(edge_preds, edge_intervals, filename, median_filter=3, event_label="Abnormal_Edge"):
    """
    Decode binary edge predictions into continuous timestamp intervals.

    Args:
        edge_preds (np.ndarray or torch.Tensor): Binary predictions for edges (0 or 1). Shape: (num_edges,).
        edge_intervals (np.ndarray): Time intervals for each edge. Shape: (num_edges, 2).
        filename (str): Name of the audio file associated with these edges.
        median_filter (int): Window size for median filtering. If <=1, no median filter is applied.
        event_label (str): Label name for the abnormal edge event.

    Returns:
        pd.DataFrame: A DataFrame with columns ["filename", "event_label", "onset", "offset"] representing continuous abnormal intervals.
    """
    # Convert to numpy if tensor
    if hasattr(edge_preds, 'cpu'):
        edge_preds = edge_preds.cpu().numpy()
    edge_preds = np.asarray(edge_preds).astype(np.int32)

    # Apply median filtering if required
    if median_filter > 1:
        edge_preds = scipy.ndimage.median_filter(edge_preds, size=median_filter)

    # Find contiguous regions of abnormal edges (edge_preds == 1)
    padded = np.concatenate([[0], edge_preds, [0]])
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1

    results = []
    for start, end in zip(starts, ends):
        if start <= end:
            onset = edge_intervals[start, 0]
            offset = edge_intervals[end, 1]
            results.append([filename, event_label, onset, offset])

    df = pd.DataFrame(results, columns=["filename", "event_label", "onset", "offset"])
    return df


def generate_edge_timestamp_info(node_intervals, edge_preds, filename):
    """
    High-level function that:
      1. Generates node intervals from node labels.
      2. Uses node intervals to create edge intervals.
      3. Decodes binary edge predictions to timestamp information.

    Args:
        node_labels (torch.Tensor or np.ndarray): Node-level labels or predictions. Shape: (num_nodes,).
        node_size_in_frames (int): Number of frames represented by each node.
        cur_audio_dur (float): Duration of the current audio in seconds.
        frame_hop (int): Frame hop in samples.
        fs (int): Sampling frequency in Hz.
        edge_preds (torch.Tensor or np.ndarray): Binary edge predictions. Shape: (num_edges,).
        filename (str): Audio filename.

    Returns:
        pd.DataFrame: A DataFrame containing the timestamp information derived from edge predictions.
    """
    # Convert node labels to numpy if tensor
    if hasattr(node_intervals, 'cpu'):
        node_intervals = node_intervals.cpu().numpy()
    node_intervals = np.asarray(node_intervals)

    # Convert to 2D format (intervals)
    intervals_2d = np.column_stack((node_intervals[:-1], node_intervals[1:]))

    # Generate edge intervals
    edge_intervals = generate_edge_intervals(intervals_2d)

    # Decode edge predictions
    edge_timestamp_df = decode_edge_predictions(edge_preds, edge_intervals, filename)

    return edge_timestamp_df


import pandas as pd
import numpy as np


def overlap_intervals(interval1, interval2):
    """
    Compute the overlapping interval between two intervals [start1, end1] and [start2, end2].
    Returns (overlap_start, overlap_end) if they overlap, else None.
    """
    start1, end1 = interval1
    start2, end2 = interval2
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    if overlap_start < overlap_end:
        return (overlap_start, overlap_end)
    return None


def find_closest_node_label(node_intervals, segment):
    """
    Given a set of node intervals with event_labels and a segment [start, end],
    find the node interval whose midpoint is closest in time to the midpoint of 'segment',
    and return that interval's event_label.
    """
    seg_mid = (segment[0] + segment[1]) / 2.0
    # Compute midpoints for node intervals
    node_midpoints = [((r["onset"] + r["offset"]) / 2.0, r["event_label"]) for _, r in node_intervals.iterrows()]
    # Find closest node interval by midpoint distance
    closest_label = None
    closest_dist = float('inf')
    for mid, label in node_midpoints:
        dist = abs(mid - seg_mid)
        if dist < closest_dist:
            closest_dist = dist
            closest_label = label
    return closest_label


import pandas as pd


def assign_labels_to_edge_intervals(pred_abnormal_events, edge_timestamp_df, thresholds=[0.85, 0.70, 0.55, 0.40]):
    """
    Assign event labels to edge intervals based on predicted abnormal events at multiple thresholds.

    Parameters
    ----------
    pred_abnormal_events : dict
        Dictionary where keys are thresholds (floats) and values are DataFrames of predicted abnormal events
        with columns ["event_label", "onset", "offset", "filename"].
    edge_timestamp_df : pd.DataFrame
        DataFrame with columns ["filename", "event_label", "onset", "offset"] representing edge intervals.
    thresholds : list
        List of thresholds in descending order. If not provided, defaults to [0.85, 0.70, 0.55, 0.40].

    Returns
    -------
    refined_edge_intervals : pd.DataFrame
        A DataFrame with columns ["filename", "event_label", "onset", "offset"] where each row represents
        an edge interval assigned one or more abnormal event labels based on the logic described.
        If multiple event labels are assigned to the same edge interval (in the highest threshold scenario),
        multiple rows will appear for the same interval, each with a distinct event_label.
    """

    # Ensure thresholds are sorted in descending order
    thresholds = sorted(thresholds, reverse=True)

    refined_records = []

    for idx, edge_row in edge_timestamp_df.iterrows():
        edge_onset = edge_row['onset']
        edge_offset = edge_row['offset']
        edge_filename = edge_row['filename']

        # We'll store results of event matching here
        matched_events = []

        # Check each threshold in descending order
        for c_th in thresholds:
            if c_th not in pred_abnormal_events:
                continue
            pred_df = pred_abnormal_events[c_th]

            # Find events within the edge interval: fully contained events
            # pred_event_onset >= edge_onset AND pred_event_offset <= edge_offset
            within_mask = (
                    (pred_df['filename'] == edge_filename) &
                    (pred_df['onset'] >= edge_onset) &
                    (pred_df['offset'] <= edge_offset)
            )
            matched = pred_df[within_mask]

            if len(matched) == 1:
                # Found exactly one event - assign it
                matched_events = matched.to_dict('records')
                # Since we found a suitable match at this threshold, we won't check lower thresholds
                break
            elif len(matched) > 1:
                # Found multiple events at this threshold
                # According to the logic:
                # If this threshold is the highest threshold at which we find matches, we keep them all.
                # We do not go to lower thresholds after this.
                matched_events = matched.to_dict('records')
                break
            # If no matches at this threshold, continue to lower threshold.

        # After checking all thresholds:
        if len(matched_events) == 0:
            # No events matched inside this edge interval - discard this interval
            continue
        else:
            # Add the matched events to the refined output
            # Each matched event label from the chosen threshold interval is assigned the same edge interval onset/offset
            for evt in matched_events:
                refined_records.append({
                    "filename": edge_filename,
                    "event_label": evt['event_label'],
                    "onset": edge_onset,
                    "offset": edge_offset
                })

    refined_edge_intervals = pd.DataFrame(refined_records)
    return refined_edge_intervals



def assign_labels_to_edge_intervals_v2(pred_abnormal_events, edge_timestamp_df, thresholds=[0.85, 0.70, 0.55, 0.40]):
    """
    Assign event labels to edge intervals based on predicted abnormal events at multiple thresholds.

    Parameters
    ----------
    pred_abnormal_events : dict
        Dictionary where keys are thresholds (floats) and values are DataFrames of predicted abnormal events
        with columns ["event_label", "onset", "offset", "filename"].
    edge_timestamp_df : pd.DataFrame
        DataFrame with columns ["filename", "event_label", "onset", "offset"] representing edge intervals.
    thresholds : list
        List of thresholds in descending order. If not provided, defaults to [0.85, 0.70, 0.55, 0.40].

    Returns
    -------
    refined_edge_intervals : pd.DataFrame
        A DataFrame with columns ["filename", "event_label", "onset", "offset"] where each row represents
        an edge interval assigned one or more abnormal event labels based on the logic described.
        If multiple event labels are assigned to the same edge interval (in the highest threshold scenario),
        multiple rows will appear for the same interval, each with a distinct event_label.
    """

    # Ensure thresholds are sorted in descending order
    thresholds = sorted(thresholds, reverse=True)

    refined_records = []

    for idx, edge_row in edge_timestamp_df.iterrows():
        edge_onset = edge_row['onset']
        edge_offset = edge_row['offset']
        edge_filename = edge_row['filename']

        # We'll store results of event matching here
        matched_events = []

        # Check each threshold in descending order
        for c_th in thresholds:
            if c_th not in pred_abnormal_events:
                continue
            pred_df = pred_abnormal_events[c_th]

            # Find events within the edge interval: fully contained events
            # pred_event_onset >= edge_onset AND pred_event_offset <= edge_offset
            within_mask = (
                    (pred_df['filename'] == edge_filename) &
                    (pred_df['onset'] >= edge_onset) &
                    (pred_df['offset'] <= edge_offset)
            )
            within_events = pred_df[within_mask]

            # Events with sufficient overlap
            overlap_mask = (
                (pred_df['filename'] == edge_filename) &
                (pred_df['onset'] < edge_offset) &  # Event starts before edge ends
                (pred_df['offset'] > edge_onset)   # Event ends after edge starts
            )
            overlapping_events = pred_df[overlap_mask].copy()

            # Calculate overlap duration and minimum duration
            # Calculate overlap duration and minimum duration if overlapping_events is not empty
            if not overlapping_events.empty:
                overlapping_events['overlap_time'] = overlapping_events.apply(
                    lambda row: max(0, min(row['offset'], edge_offset) - max(row['onset'], edge_onset)),
                    axis=1
                )
                overlapping_events['min_dur'] = overlapping_events.apply(
                    lambda row: min(row['offset'] - row['onset'], edge_offset - edge_onset),
                    axis=1
                )

                # Filter overlapping events by the 30% criterion
                sufficient_overlap = overlapping_events[
                    overlapping_events['overlap_time'] >= 0.3 * overlapping_events['min_dur']
                    ]
            else:
                sufficient_overlap = pd.DataFrame(columns=overlapping_events.columns)

            # Ensure at least one DataFrame is non-empty before concatenation
            if not within_events.empty or not sufficient_overlap.empty:
                matched = pd.concat([within_events, sufficient_overlap]).drop_duplicates()
            else:
                matched = pd.DataFrame(columns=within_events.columns)





            if len(matched) == 1:
                # Found exactly one event - assign it
                matched_events = matched.to_dict('records')
                # Since we found a suitable match at this threshold, we won't check lower thresholds
                break
            elif len(matched) > 1:
                # Found multiple events at this threshold
                # According to the logic:
                # If this threshold is the highest threshold at which we find matches, we keep them all.
                # We do not go to lower thresholds after this.
                matched_events = matched.to_dict('records')
                break
            # If no matches at this threshold, continue to lower threshold.

        # After checking all thresholds:
        if len(matched_events) == 0:
            # No events matched inside this edge interval - discard this interval
            continue
        else:
            # Add the matched events to the refined output
            # Each matched event label from the chosen threshold interval is assigned the same edge interval onset/offset
            for evt in matched_events:
                refined_records.append({
                    "filename": edge_filename,
                    "event_label": evt['event_label'],
                    "onset": edge_onset,
                    "offset": edge_offset
                })

    refined_edge_intervals = pd.DataFrame(refined_records)
    return refined_edge_intervals






# Example usage (within your code flow):
#
# refined_edge_intervals = assign_labels_to_edge_intervals(
#     pred_abnormal_events=pred_abnormal_events,
#     edge_timestamp_df=edge_timestamp_df,
#     thresholds=[0.85, 0.70, 0.55, 0.40]
# )

r"""
scores_raw, scores_postprocessed, 和 prediction_dfs 这三个输出在功能和用途上有所不同，
它们之间的区别与联系如下：

1. scores_raw
定义：原始预测分数数据框。
内容：包含未经任何处理的预测分数，直接从模型输出得到。
作用：保留原始预测结果，便于后续分析或调试。
可用于对比处理前后的差异。


2. scores_postprocessed
定义：经过中值滤波处理后的预测分数数据框。
内容：包含经过平滑处理的预测分数，减少了噪声的影响。
作用： 提高预测结果的稳定性。
适用于更准确的事件检测。

3. prediction_dfs
定义：不同阈值下的最终预测结果数据框。
内容：包含根据不同阈值生成的具体事件预测结果。
作用： 提供多个阈值下的预测结果，便于选择最佳阈值。
方便后续分析和评估。


区别与联系
区别：
原始 vs. 处理后：scores_raw 是原始预测分数，而 scores_postprocessed 是经过中值滤波处理后的预测分数。
预测 vs. 分数：scores_raw 和 scores_postprocessed 存储的是分数，
而 prediction_dfs 存储的是根据这些分数生成的事件预测结果。


联系：
scores_raw 和 scores_postprocessed 都是从模型输出的预测分数生成的，只是后者经过了中值滤波处理。
prediction_dfs 的生成依赖于 scores_postprocessed，即使用处理后的预测分数生成最终的事件预测结果。


为什么需要这三种输出
保留原始数据：scores_raw 保留了原始预测分数，方便后续调试和分析。
提高准确性：scores_postprocessed 通过中值滤波提高了预测分数的稳定性，减少了噪声的影响。
多阈值评估：prediction_dfs 提供了不同阈值下的预测结果，便于选择最佳阈值，提高预测性能。
综上所述，这三种输出各有侧重，共同构成了一个完整的预测流程，确保了从原始数据到最终预测结果的全面覆盖。



"""






# def convert_to_event_based(weak_dataframe):
#     """Convert a weakly labeled DataFrame ('filename', 'event_labels') to a DataFrame strongly labeled
#     ('filename', 'onset', 'offset', 'event_label').
#
#     Args:
#         weak_dataframe: pd.DataFrame, the dataframe to be converted.
#
#     Returns:
#         pd.DataFrame, the dataframe strongly labeled.
#     """
#
#     new = []
#     for i, r in weak_dataframe.iterrows():
#         events = r["event_labels"].split(",")
#         for e in events:
#             new.append(
#                 {"filename": r["filename"], "event_label": e, "onset": 0, "offset": 1}
#             )
#     return pd.DataFrame(new)


import math

def format_metric(val, is_percent=False, width=6):
    if math.isnan(val):
        return 'nan'.ljust(width) if not is_percent else 'nan%'
    if is_percent:
        return f'{val * 100:.1f}%'.ljust(width)
    return f'{val:.2f}'.ljust(width)

# def log_sedeval_metrics(predictions, ground_truth, save_dir=None):
#     """Return the set of metrics from sed_eval
#     Args:
#         predictions: pd.DataFrame, the dataframe of predictions.
#         ground_truth: pd.DataFrame, the dataframe of groundtruth.
#         save_dir: str, path to the folder where to save the event and segment based metrics outputs.
#
#     Returns:
#         tuple, event-based macro-F1 and micro-F1, segment-based macro-F1 and micro-F1
#     """
#
#
#     print( " \ncall the log_sedeval_metrics  ")
#
#     if predictions.empty:
#         return 0.0, 0.0, 0.0, 0.0
#
#     gt = pd.read_csv(ground_truth, sep="\t")
#
#     event_res, segment_res = compute_sed_eval_metrics(predictions, gt)
#
#     if save_dir is not None:
#         os.makedirs(save_dir, exist_ok=True)
#         with open(os.path.join(save_dir, "event_f1.txt"), "w") as f:
#             f.write(str(event_res))
#
#         with open(os.path.join(save_dir, "segment_f1.txt"), "w") as f:
#             f.write(str(segment_res))
#
#     overall_f_score, overall_er = (event_res.results()["overall"]["f_measure"]["f_measure"],
#                          event_res.results()["overall"]["error_rate"]["error_rate"])
#     print(f"\nthe Event based overall   f score: {overall_f_score}")
#     print(f"\nthe Event based overall   error rate : {overall_er}")
#
#
#     cls_score, cls_er = (event_res.results()["class_wise_average"]["f_measure"]["f_measure"],
#                          event_res.results()["class_wise_average"]["error_rate"]["error_rate"])
#     print(f"\nthe Sound Event Detection class wise f score: {cls_score}")
#     print(f"\nthe Sound Event Detection class wise error rate : {cls_er}")
#
#     class_wise_data  =  event_res.results()["class_wise"],
#     class_wise_data = class_wise_data[0]
#     print(class_wise_data)
#     # Print header
#     print("  Class-wise metrics")
#     print("  ======================================")
#     print("    Event label  | Nref    Nsys  | F        Pre      Rec    | ER       Del      Ins    |")
#     print("    ------------ | -----   ----- | ------   ------   ------ | ------   ------   ------ |")
#
#     # Iterate over each class and print the results
#     for event_label, metrics in class_wise_data.items():
#         Nref = int(metrics['count'].get('Nref', 0))
#         Nsys = int(metrics['count'].get('Nsys', 0))
#
#         f_measure = metrics['f_measure'].get('f_measure', float('nan'))
#         precision = metrics['f_measure'].get('precision', float('nan'))
#         recall = metrics['f_measure'].get('recall', float('nan'))
#
#         error_rate = metrics['error_rate'].get('error_rate', float('nan'))
#         deletion_rate = metrics['error_rate'].get('deletion_rate', float('nan'))
#         insertion_rate = metrics['error_rate'].get('insertion_rate', float('nan'))
#
#         # Shorten the event label if it's too long
#         short_label = event_label[:10] + '..' if len(event_label) > 10 else event_label.ljust(10)
#
#         # Print each row with fixed-width columns
#         print(f"    {short_label:<12} | {Nref:<5}   {Nsys:<5} | "
#               f"{format_metric(f_measure, True, 7)}"
#               f"{format_metric(precision, True, 7)}"
#               f"{format_metric(recall, True, 7)} | "
#               f"{format_metric(error_rate, False, 7)}"
#               f"{format_metric(deletion_rate, False, 7)}"
#               f"{format_metric(insertion_rate, False, 7)} |")
#
#
#
#     return (
#         event_res.results()["class_wise_average"]["f_measure"]["f_measure"],
#         event_res.results()["class_wise_average"]["error_rate"]["error_rate"],
#         event_res.results()["class_wise"],
#         event_res.results()["overall"]["f_measure"]["f_measure"],
#
#         segment_res.results()["class_wise_average"]["f_measure"]["f_measure"],
#         segment_res.results()["overall"]["f_measure"]["f_measure"],
#     )  # return also segment measures
#


# def compute_event_based_metrics_ori(predictions, ground_truth, save_dir=None):
#     """Return the set of metrics from sed_eval
#     Args:
#         predictions: pd.DataFrame, the dataframe of predictions.
#         ground_truth: pd.DataFrame, the dataframe of groundtruth.
#         save_dir: str, path to the folder where to save the event and segment based metrics outputs.
#
#     Returns:
#         tuple, event-based macro-F1 and micro-F1, segment-based macro-F1 and micro-F1
#     """
#
#
#     print( " \n call the compute event based metrics  ")
#     if predictions.empty:
#         print( f" \nWarning !!! "
#                f"\n On this epoch under thread, There is No abnormal predictions node!!! ")
#
#         return 0.0, 0.0, 0.0, 0.0
#
#     gt = pd.read_csv(ground_truth, sep="\t")
#     # gt_only_abnormal = gt[gt['event_label'] != 'Normal']
#     gt_only_abnormal = gt[gt['record_bin_label'] != 'Normal']
#
#     event_res, segment_res = compute_sed_eval_metrics(predictions, gt_only_abnormal)
#
#     if save_dir is not None:
#         os.makedirs(save_dir, exist_ok=True)
#         with open(os.path.join(save_dir, "event_f1.txt"), "w") as f:
#             f.write(str(event_res))
#
#         with open(os.path.join(save_dir, "segment_f1.txt"), "w") as f:
#             f.write(str(segment_res))
#
#     overall_f_score, overall_er = (event_res.results()["overall"]["f_measure"]["f_measure"],
#                          event_res.results()["overall"]["error_rate"]["error_rate"])
#     print(f"\nthe Event based overall   f score: {overall_f_score}, \t error rate : {overall_er}")
#
#
#     cls_score, cls_er = (event_res.results()["class_wise_average"]["f_measure"]["f_measure"],
#                          event_res.results()["class_wise_average"]["error_rate"]["error_rate"])
#     print(f"\nthe Event based class wise average f score: {cls_score},\t error rate : {cls_er}\n")
#
#
#     class_wise_data  =  event_res.results()["class_wise"],
#     class_wise_data = class_wise_data[0]
#     # print(class_wise_data)
#     # Print header
#     print("  Class-wise metrics")
#     print("  ======================================")
#     print("    Event label  | Nref    Nsys  | F        Pre      Rec    | ER       Del      Ins    |")
#     print("    ------------ | -----   ----- | ------   ------   ------ | ------   ------   ------ |")
#
#     # Iterate over each class and print the results
#     for event_label, metrics in class_wise_data.items():
#         Nref = int(metrics['count'].get('Nref', 0))
#         Nsys = int(metrics['count'].get('Nsys', 0))
#
#         f_measure = metrics['f_measure'].get('f_measure', float('nan'))
#         precision = metrics['f_measure'].get('precision', float('nan'))
#         recall = metrics['f_measure'].get('recall', float('nan'))
#
#         error_rate = metrics['error_rate'].get('error_rate', float('nan'))
#         deletion_rate = metrics['error_rate'].get('deletion_rate', float('nan'))
#         insertion_rate = metrics['error_rate'].get('insertion_rate', float('nan'))
#
#         # Shorten the event label if it's too long
#         short_label = event_label[:10] + '..' if len(event_label) > 10 else event_label.ljust(10)
#
#         # Shorten the event label if it's too long
#         short_label = event_label[:10] + '..' if len(event_label) > 10 else event_label.ljust(10)
#
#         # Print each row with fixed-width columns
#         print(f"    {short_label:<12} | {Nref:<5}   {Nsys:<5} | "
#               f"{format_metric(f_measure, True, 7)}"
#               f"{format_metric(precision, True, 7)}"
#               f"{format_metric(recall, True, 7)} | "
#               f"{format_metric(error_rate, False, 7)}"
#               f"{format_metric(deletion_rate, False, 7)}"
#               f"{format_metric(insertion_rate, False, 7)} |")
#
#
#
#     return (
#         event_res.results()["class_wise_average"]["f_measure"]["f_measure"],
#         event_res.results()["class_wise_average"]["error_rate"]["error_rate"],
#         event_res.results()["class_wise"],
#         event_res.results()["overall"]["f_measure"]["f_measure"],
#
#         segment_res.results()["class_wise_average"]["f_measure"]["f_measure"],
#         segment_res.results()["overall"]["f_measure"]["f_measure"],
#     )  # return also segment measures



def compute_event_based_metrics(predictions, ground_truth, save_dir=None):
    """Return the set of metrics from sed_eval
    Args:
        predictions: pd.DataFrame, the dataframe of predictions.
        ground_truth: pd.DataFrame, the dataframe of groundtruth.
        save_dir: str, path to the folder where to save the event and segment based metrics outputs.

    Returns:
        tuple, event-based macro-F1 and micro-F1, segment-based macro-F1 and micro-F1
    """


    print( "\n call the compute event based metrics  ")
    if predictions.empty:
        print( f" \nWarning !!! "
               f"\n On this epoch under thread, There is No abnormal predictions node!!! ")

        return 0.0, 0.0, 0.0, 0.0

    gt = pd.read_csv(ground_truth, sep="\t");
    gt = gt[gt['event_label'] != 'Wheeze+Crackle'] #  这种复合音频不纳入统计；
    # gt_only_abnormal = gt[gt['event_label'] != 'Normal'],  # 只保留下真实值 是异常类型的 事件。
    gt_only_abnormal = gt[gt['record_bin_label'] != 'Normal']

    # Check for NaNs or Infs in predictions
    if predictions.isnull().values.any():
        print("Predictions contain NaNs")
        # Handle accordingly

    if np.isinf(predictions.select_dtypes(include=[np.number])).any().any():
        print("Predictions contain Infs")
        # Handle accordingly

    # Similarly for gt_only_abnormal
    if gt_only_abnormal.isnull().values.any():
        print("Ground truth contains NaNs")
        # Handle accordingly

    if np.isinf(gt_only_abnormal.select_dtypes(include=[np.number])).any().any():
        print("Ground truth contains Infs")
        # Handle accordingly

    # event_res, segment_res = compute_sed_eval_metrics(predictions, gt_only_abnormal)
    try:
        event_res, segment_res = compute_sed_eval_metrics(predictions, gt_only_abnormal)
    except Exception as e:
        print(f"Error in compute_sed_eval_metrics function: {e}")
        # Handle the error, possibly return default values
        return 0.0, 0.0, 0.0, 0.0



    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "event_f1.txt"), "w") as f:
            f.write(str(event_res))

        with open(os.path.join(save_dir, "segment_f1.txt"), "w") as f:
            f.write(str(segment_res))

    overall_f_score, overall_er = (event_res.results()["overall"]["f_measure"]["f_measure"],
                         event_res.results()["overall"]["error_rate"]["error_rate"])
    print(f"\nthe Event based overall   f score: {overall_f_score}, \t error rate : {overall_er}")


    cls_score, cls_er = (event_res.results()["class_wise_average"]["f_measure"]["f_measure"],
                         event_res.results()["class_wise_average"]["error_rate"]["error_rate"])
    print(f"\nthe Event based class wise average f score: {cls_score},\t error rate : {cls_er}\n")


    # class_wise_data  =  event_res.results()["class_wise"],
    # class_wise_data = class_wise_data[0]
    class_wise_data = event_res.results().get("class_wise", {})
    if not class_wise_data:
        print("No class-wise data available")
        # Handle accordingly


    # print(class_wise_data)
    # Print header
    print("  Class-wise metrics")
    print("  ======================================")
    print("    Event label  | Nref    Nsys  | F        Pre      Rec    | ER       Del      Ins    |")
    print("    ------------ | -----   ----- | ------   ------   ------ | ------   ------   ------ |")

    # Iterate over each class and print the results
    for event_label, metrics in class_wise_data.items():
        Nref = int(metrics['count'].get('Nref', 0))
        Nsys = int(metrics['count'].get('Nsys', 0))

        f_measure = metrics['f_measure'].get('f_measure', float('nan'))
        precision = metrics['f_measure'].get('precision', float('nan'))
        recall = metrics['f_measure'].get('recall', float('nan'))

        error_rate = metrics['error_rate'].get('error_rate', float('nan'))
        deletion_rate = metrics['error_rate'].get('deletion_rate', float('nan'))
        insertion_rate = metrics['error_rate'].get('insertion_rate', float('nan'))

        # Shorten the event label if it's too long
        short_label = event_label[:10] + '..' if len(event_label) > 10 else event_label.ljust(10)

        # Print each row with fixed-width columns
        print(f"    {short_label:<12} | {Nref:<5}   {Nsys:<5} | "
              f"{format_metric(f_measure, True, 7)}"
              f"{format_metric(precision, True, 7)}"
              f"{format_metric(recall, True, 7)} | "
              f"{format_metric(error_rate, False, 7)}"
              f"{format_metric(deletion_rate, False, 7)}"
              f"{format_metric(insertion_rate, False, 7)} |")



    return (
        event_res.results()["class_wise_average"]["f_measure"]["f_measure"],
        event_res.results()["class_wise_average"]["error_rate"]["error_rate"],
        event_res.results()["class_wise"],
        event_res.results()["overall"]["f_measure"]["f_measure"],

        segment_res.results()["class_wise_average"]["f_measure"]["f_measure"],
        segment_res.results()["overall"]["f_measure"]["f_measure"],
    )  # return also segment measures





def parse_jams(jams_list, encoder, out_json):
    if len(jams_list) == 0:
        raise IndexError("jams list is empty ! Wrong path ?")

    backgrounds = []
    sources = []
    for jamfile in jams_list:
        with open(jamfile, "r") as f:
            jdata = json.load(f)

        # check if we have annotations for each source in scaper
        assert len(jdata["annotations"][0]["data"]) == len(
            jdata["annotations"][-1]["sandbox"]["scaper"]["isolated_events_audio_path"]
        )

        for indx, sound in enumerate(jdata["annotations"][0]["data"]):
            source_name = Path(
                jdata["annotations"][-1]["sandbox"]["scaper"][
                    "isolated_events_audio_path"
                ][indx]
            ).stem
            source_file = os.path.join(
                Path(jamfile).parent,
                Path(jamfile).stem + "_events",
                source_name + ".wav",
            )

            if sound["value"]["role"] == "background":
                backgrounds.append(source_file)
            else:  # it is an event
                if (
                    sound["value"]["label"] not in encoder.labels
                ):  # correct different labels
                    if sound["value"]["label"].startswith("Frying"):
                        sound["value"]["label"] = "Frying"
                    elif sound["value"]["label"].startswith("Vacuum_cleaner"):
                        sound["value"]["label"] = "Vacuum_cleaner"
                    else:
                        raise NotImplementedError

                sources.append(
                    {
                        "filename": source_file,
                        "onset": sound["value"]["event_time"],
                        "offset": sound["value"]["event_time"]
                        + sound["value"]["event_duration"],
                        "event_label": sound["value"]["label"],
                    }
                )

    os.makedirs(Path(out_json).parent, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({"backgrounds": backgrounds, "sources": sources}, f, indent=4)


def generate_tsv_wav_durations(audio_dir, out_tsv):
    """
        Generate a dataframe with filename and duration of the file

    Args:
        audio_dir: str, the path of the folder where audio files are (used by glob.glob)
        out_tsv: str, the path of the output tsv file

    Returns:
        pd.DataFrame: the dataframe containing filenames and durations
    """
    meta_list = []
    for file in glob.glob(os.path.join(audio_dir, "*.wav")):
        d = soundfile.info(file).duration
        meta_list.append([os.path.basename(file), d])
    meta_df = pd.DataFrame(meta_list, columns=["filename", "duration"])
    if out_tsv is not None:
        meta_df.to_csv(out_tsv, sep="\t", index=False, float_format="%.1f")

    return meta_df


def calculate_macs(model, config, dataset=None):
    """
    The function calculate the multiply–accumulate operation (MACs) of the model given as input.

    Args:
        model: deep learning model to calculate the macs for
        config: config used to train the model
        dataset: dataset used to train the model

    Returns:

    """
    n_frames = int(
        (
            (config["feats"]["sample_rate"] * config["data"]["audio_max_len"])
            / config["feats"]["hop_length"]
        )
        + 1
    )
    input_size = [1, config["feats"]["n_mels"], n_frames] #(1,mels=128, 626 )
    input = torch.randn(input_size)

    if "use_embeddings" in config["net"] and config["net"]["use_embeddings"]:
        audio, label, padded_indxs, path, embeddings = dataset[0]
        embeddings = embeddings.repeat(1, 1, 1)  # 如果使用，则从数据集中读取第一个样本，并复制其嵌入层输出以匹配输入形状
        macs, params = profile(model, inputs=(input, None, embeddings))
    else: # 调用 profile 函数计算模型的 MACs 和参数数量。
        macs, params = profile(model, inputs=(input,))
    # 将计算结果格式化为字符串，并返回格式化后的 MACs 和参数数量。
    macs, params = clever_format([macs, params], "%.3f")
    return macs, params



"""
c_scores = scipy.ndimage.filters.median_filter(c_scores, (median_filter, 1))

First Dimension（第一维度的中值筛选）：中值筛选条件沿第一维度（156 个样本）应用。
过滤器垂直向下滑动 10 列中的每一列，
将每个位置的值替换为过滤器窗口大小median_filter内垂直相邻位置的中值。

第二个维度不进行筛选：第二个维度（10 个类别）不受筛选条件的影响，因为沿此轴的大小设置为 1。



帧 （156）：中值筛选器单独处理每个类别的 156 帧中的分数。
如果相邻帧之间存在杂色或分数突然变化，则此筛选将通过将每个帧的分数替换为周围帧的中值分数
（在指定的窗口大小 median_filter 内）来减少此类波动。


类别 （10）：帧的每个类别的分数都独立于其他类别进行过滤，
确保仅减少时间噪声（帧之间），而不会在类别之间混合分数。
类别之间不会进行筛选，因为第二个维度（10 个类别）的筛选条件大小为 1，
这意味着每个帧的值仅根据其相邻帧进行筛选，而不是跨类别进行筛选。



median_score:
大小为 7 的窗口意味着对于每个值（或帧，在本例中），
过滤器会考虑当前帧及其前后的 3 帧（即每侧 3 个相邻帧）。


过滤器对这 7 个值进行排序，并选择中间的值作为中位数，将当前帧的值替换为该中位数。

假设现在有10 个类别，能否根据不同的类别， 设置不同的median score 大小，
比方说对于 class1 median score设置为 5，   calss2  对应的median score设置为 7. 等等， 如何实现；





"""