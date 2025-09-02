import glob
import os
import random
from pathlib import Path

import h5py

import torchaudio
from torch.utils.data import Dataset

from desed_task.utils.data_spec_aug import gen_augmented, standardize_rows, GroupSpecAugment_v2

# prepare for the  data aug
from pytorch_wavelets import DWT1DForward, DWT1DInverse
# prep for gen  spectrogram and  spec aug
from torchaudio.transforms import  AmplitudeToDB, MelSpectrogram
from  nnAudio.features import  CQT2010v2, Gammatonegram

# v5,  gen the  record multi label, prepare for the  balanced sampler


# v1_1,  对每个样本， 生成先验 anchor interval,
# 并通过iou  计算每个 anchor interval 对应的真实区间以及区间类型






def to_mono(mixture, random_ch=False):
    if mixture.ndim > 1:  # multi channel
        if not random_ch:
            mixture = torch.mean(mixture, 0)
        else:  # randomly select one channel
            indx = np.random.randint(0, mixture.shape[0] - 1)
            mixture = mixture[indx]
    return mixture


def pad_audio(audio, target_len, fs, test=False): # note, 该函数用于对 不足或超过10s的 音频，进行对齐，并重新设置起始和终止时间；
    if audio.shape[-1] < target_len:
        audio = torch.nn.functional.pad(
            audio, (0, target_len - audio.shape[-1]), mode="constant"
        )
        # 若音频长度小于目标长度，则在末尾添加零填充，并计算填充比例
        padded_indx = [target_len / len(audio)]
        onset_s = 0.000

    elif len(audio) > target_len:
        if test:
            clip_onset = 0
        else: #若音频过长，根据测试模式随机或从起始截取目标长度片段，并计算起始时间；
            clip_onset = random.randint(0, len(audio) - target_len)
        audio = audio[clip_onset : clip_onset + target_len]
        onset_s = round(clip_onset / fs, 3)

        padded_indx = [target_len / len(audio)]
    else:
        onset_s = 0.000
        padded_indx = [1.0]

    offset_s = round(onset_s + (target_len / fs), 3) # 主要用于音频或信号处理中的时间偏移计算。
    return audio, onset_s, offset_s, padded_indx



import numpy as np

def process_labels_NA(df, onset, offset):
    # Only process rows where "onset" and "offset" are not NA
    df_valid = df.dropna(subset=["onset", "offset"])

    # Convert "onset" and "offset" to numeric values (this ensures any stray "NA" strings are removed)
    df_valid["onset"] = pd.to_numeric(df_valid["onset"], errors='coerce')
    df_valid["offset"] = pd.to_numeric(df_valid["offset"], errors='coerce')

    # Drop rows where conversion to numeric failed (i.e., still has NA)
    df_valid = df_valid.dropna(subset=["onset", "offset"])

    # Subtract the given onset value
    df_valid["onset"] = df_valid["onset"] - onset
    df_valid["offset"] = df_valid["offset"] - onset

    # Ensure "onset" is not less than 0 and "offset" is not greater than 16
    df_valid["onset"] = df_valid["onset"].apply(lambda x: max(0, x))
    df_valid["offset"] = df_valid["offset"].apply(lambda x: min(16, x))

    # Keep only rows where "onset" is less than "offset" and drop duplicates
    df_new = df_valid[df_valid["onset"] < df_valid["offset"]]

    # Append rows with NA values for "onset" or "offset" back to the final DataFrame
    df_na = df[df["onset"].isna() | df["offset"].isna()]
    df_final = pd.concat([df_new, df_na], ignore_index=True)

    return df_final.drop_duplicates()


interval_4labels = {
    "Normal": -1,
    "Crackle": 0,
    "Wheeze": 1,
    "Stridor": 2,
    "Rhonchi": 3
}


from torch.utils.data import Dataset
from tqdm import  tqdm
import  datetime

class RespiraGnnSet(Dataset):
    def __init__(
        self,
        audio_folder,
        tsv_entries,
        encoder,

        fs=  8000,
        n_fft =  1024,
        win_len =  1024,
        hop_len = 128,
        f_max = 8000,
        n_mels = 82,

        interval_iou_threshold = 0.5, # use for assign the GT for each anchor interval;

        train_flag = False,
        audio_aug=False,
        spec_aug=False,
        return_filename=False,
        random_channel=False,
        multisrc=False,
        node_fea_generator = None,
        embeddings_hdf5_file=None,
        embedding_type=None,
        mask_events_other_than=None,
        test=False,
    ):
        super(RespiraGnnSet, self).__init__()
        self.encoder = encoder
        self.fs = fs
        self.train_flag = train_flag
        self.return_filename = return_filename
        self.random_channel = random_channel
        self.multisrc = multisrc


        self.audio_aug = audio_aug
        self.spec_aug = GroupSpecAugment_v2(specaug_policy='group_sr22k')

        self.node_fea_generator = node_fea_generator
        self.embeddings_hdf5_file = embeddings_hdf5_file
        self.embedding_type = embedding_type
        self.test = test

        if mask_events_other_than is not None:
            # Fetch indexes to mask
            self.mask_events_other_than = torch.ones(len(encoder.labels))
            for indx, cls in enumerate(encoder.labels):
                if cls not in mask_events_other_than:
                    self.mask_events_other_than[indx] = 0
        else:
            self.mask_events_other_than = torch.ones(len(encoder.labels))
        self.mask_events_other_than = self.mask_events_other_than.bool()

        # # 该函数调用将删除tsv_entries DataFrame中所有含有空值（NaN）的行
        # tsv_entries = tsv_entries.dropna() #  train set: (4192, 6) dropna-->  (2355,6)
        examples = {}
        position_mapping_trunc = {
            "L1": 0,  # 2nd ICS, right MCL
            "L2": 1,  # 5th ICS, right MCL
            "L3": 2,  # 4th ICS, right MAL
            "L4": 3,  # 10th ICS, right MAL
            "L5": 4,  # 2nd ICS, left MCL
            "L6": 5,  # 5th ICS, left MCL
            "L7": 6,  # 4th ICS, left MAL
            "L8": 7,  # 10th ICS, left MCL
        }

        for i, r in tsv_entries.iterrows():
            if "Wheeze+Crackle" in str(r["event_label"]):
                continue

            raw_fname = r["filename"]
            fname = raw_fname.replace("_label", "")  # Clean filename for consistency

            if fname not in examples:
                # Determine chest position based on filename format
                if fname.startswith("trunc"):
                    try:
                        parts = fname.split("-")[-1]  # 'LX_N.wav'
                        loc_part = parts.split("_")[0]  # 'LX'
                        chest_position = position_mapping_trunc.get(loc_part, -1)
                    except Exception as e:
                        raise ValueError(f"Failed to parse trunc filename '{fname}': {e}")
                elif fname.startswith("steth"):
                    chest_position = -1  # Placeholder for unknown auscultation position
                else:
                    # Original format: "65097128_5.6_1_p1_2242.wav"
                    filename_parts = fname.split('_')
                    if len(filename_parts) != 5:
                        raise ValueError(
                            f"Filename '{fname}' does not follow the expected format with 4 underscores.")
                    position_mapping_orig = {"p1": 0, "p2": 1, "p3": 2, "p4": 3}
                    position_key = filename_parts[3]
                    if position_key not in position_mapping_orig:
                        raise ValueError(f"Invalid chest position key '{position_key}' in filename '{fname}'.")
                    chest_position = position_mapping_orig[position_key]

                confidence = 1.0 if "confidence" not in r else r["confidence"]

                examples[fname] = {
                    "audio_path": os.path.join(audio_folder, fname),
                    "events": [],
                    "confidence": confidence,
                    "record_bin_label": r["record_bin_label"],
                    "record_abnormal_label": r["record_abnormal_label"],
                    "chest_position": chest_position,
                    "vad_timestamps": []
                }

            # Always add an event entry, even if "onset" or "offset" is NA
            if not pd.isna(r["onset"]) and not pd.isna(r["offset"]):
                event = {
                    "event_label": r["event_label"],
                    "onset": r["onset"],
                    "offset": r["offset"],
                    "confidence": confidence,
                }
                examples[fname]["events"].append(event)

                vad_event = {
                    "start": r["onset"],
                    "end": r["offset"],
                    "event_label": r["event_label"]
                }
                examples[fname]["vad_timestamps"].append(vad_event)
            else:
                event = {
                    "event_label": r.get("event_label", "NA"),
                    "onset": np.nan,
                    "offset": np.nan,
                    "confidence": confidence,
                }
                examples[fname]["events"].append(event)

        self.examples = examples
        self.examples_list = list(examples.keys())




        # --- Fixed Anchor Intervals ---
        # Define fixed anchor intervals in normalized [0,1] space.
        # For example, use three scales (e.g., 0.05, 0.1, 0.2) and assign them cyclically.
        scales = torch.tensor([0.05, 0.1, 0.2])
        # Determine the number of centers for each scale
        # Use a base number and scale inversely with the width
        base_num = 2  # Adjustable base number to control density
        num_centers = [int(base_num / scale.item()) for scale in scales]

        # Generate intervals for each scale
        anchor_intervals_list = []
        for k, scale in enumerate(scales): # (start, end ) format
            n_k = num_centers[k]
            centers_k = (torch.arange(n_k) + 0.5) / n_k  # Centers from near 0 to near 1
            starts_k = torch.clamp(centers_k - scale / 2, 0, 1)  # Start = center - scale/2
            ends_k = torch.clamp(centers_k + scale / 2, 0, 1)  # End = center + scale/2
            intervals_k = torch.stack([starts_k, ends_k], dim=1)  # Shape (n_k, 2)
            anchor_intervals_list.append(intervals_k)

        # Combine all intervals into a single tensor
        self.anchor_intervals = torch.cat(anchor_intervals_list, dim=0)
        self.iou_threshold = interval_iou_threshold
        self.num_anchor_intervals = sum(num_centers)

        #  重新遍历 self.examples,
        #  将 audio,  labels_df,
        #  anchor_interval, GT for the anchor_interval
        #  这属性加入到每个样本的字典中；
        start_time = datetime.datetime.now()
        # Dictionary to store counts for each individual event type
        event_type_counts = {}
        self.assignments = []
        for id, audio_name in enumerate(tqdm(self.examples,desc='reorganize audio info,' )):
            cur_dict = self.examples[audio_name]
            cur_audio_path =  cur_dict["audio_path"]

            audio, onset_s, offset_s, padded_indx = self.read_audio(
                cur_audio_path, self.multisrc, self.random_channel, None, self.test
            )

            audio_dur = offset_s
            # dict["events"]: 代表当前音频所包含的事件个数，以及详细标注信息；
            labels = cur_dict.get("events", [])  # 提取出其中声音事件的信息；
            # print("Labels Structure:", labels)
            labels_df = pd.DataFrame(labels)
            # 根据onset 重新标定每个音频的事件的起始，终止时间；
            labels_df = process_labels_NA(labels_df, onset_s, offset_s)
            new_items = {"audio_data": audio, "labels_df": labels_df,"audio_dur":offset_s}
            cur_dict.update(new_items)

            # Extract event labels, skipping NaN values
            event_labels = [event["event_label"] for event in labels if pd.notna(event["event_label"])]
            for label in event_labels:    # Count each event type individually
                if label not in event_type_counts:
                    event_type_counts[label] = 0
                event_type_counts[label] += 1


            vad_timestamps = cur_dict.get("vad_timestamps", labels)
            # GT intervals from vad_timestamps
            gt_intervals = [(anno["start"], anno["end"]) for anno in vad_timestamps]
            gt_labels = [anno["event_label"] for anno in vad_timestamps]  # Assuming event_label exists
            gt_boxes = torch.tensor(gt_intervals)  # (num_gt, 2)

            # Compute scaled anchor intervals
            # Scale anchor intervals to actual time
            scaled_anchor_intervals = self.anchor_intervals * audio_dur  # (num_anchors, 2), broadcasts scalar
            anchor_boxes = scaled_anchor_intervals  # (num_anchors, 2)

            if len(gt_boxes) == 0 or gt_boxes.dim() < 2:
                assignments = [{
                    "conf": 0.0,
                    "cls": -1,
                    "box": [0.0, 0.0]
                } for _ in range(len(anchor_boxes))]
            else:
                if gt_boxes.dim() == 1:
                    gt_boxes = gt_boxes.view(-1, 2)

                iou_matrix = self.compute_iou_matrix(anchor_boxes, gt_boxes)  # (num_anchors, num_gt)
                # Assign anchors to GT
                assignments = []
                max_ious, max_gt_indices = iou_matrix.max(dim=1)
                for a_idx in range(len(anchor_boxes)):
                    if max_ious[a_idx] >= self.iou_threshold:
                        gt_idx = max_gt_indices[a_idx].item()
                        gt_label = gt_labels[gt_idx]
                        assignments.append({
                            "conf": 1.0 if gt_label != "Normal" else 0.0,
                            "cls": interval_4labels[gt_label],  # Map label to index, -1 for normal
                            "box": gt_intervals[gt_idx]
                        })
                    else:
                        assignments.append({
                            "conf": 0.0,
                            "cls": -1,
                            "box": [0.0, 0.0]
                        })

            self.assignments.append(assignments)
            # Add to sample’s dictionary
            cur_dict["anchor_intervals"] = scaled_anchor_intervals  # Scaled to actual length
            cur_dict["assignments"] = assignments


        # Display the final count for each individual event type
        print("\nThe count of each event type:")
        for event_type, count in event_type_counts.items():
            print(f"{event_type}: {count}")

        end_time = datetime.datetime.now()
        duration = end_time - start_time
        print(f"\n To get the audio data cost time: {duration}, \t Current Dataset init Done !! ")



        if self.embeddings_hdf5_file is not None:
            assert (
                self.embedding_type is not None
            ), "If you use embeddings you need to specify also the type (global or frame)"
            self.ex2emb_idx = {}
            f = h5py.File(self.embeddings_hdf5_file, "r")
            for i, fname in enumerate(f["filenames"]):
                self.ex2emb_idx[fname.decode("UTF-8")] = i
        self._opened_hdf5 = None

        self.powerToDB = AmplitudeToDB(stype='power')
        self.amplitudeToDB = AmplitudeToDB(stype='magnitude')

        # 该方法是调用 torchaudio 中实现的；
        self.mel_spec = MelSpectrogram(
            sample_rate=fs,
            n_fft=n_fft,
            win_length=win_len,
            hop_length=hop_len,
            f_min=25.0,
            f_max=f_max,
            n_mels= n_mels,
        )  #.to("cuda")

        # self.cqt_spectrogram =
        # 这里设置的　cqt　内核使用默认的，　trainable=False;
        self.cqt_spectrogram = CQT2010v2(
            sr=fs,
            hop_length=hop_len,  # hop_len,
            fmin= 25.0,  # 32.7,
            fmax=f_max,  # f_max,
            n_bins= n_mels,  # 这里设置成84 为了和 Mel 滤波器个数对应上；
            trainable=False,
            bins_per_octave= 12,  # 12 bin for 72 n mels;  14 for 84 n mels;
        )# .to("cuda")

        self.gamma_spec = Gammatonegram(
            sr=fs,
            n_fft=n_fft,
            hop_length=hop_len,
            n_bins=  n_mels,
            fmin=25.0,
            fmax=f_max,
            trainable_bins=False,
        )# .to("cuda")



    def __len__(self):
        return len(self.examples_list)

    def __getitem__(self, idx):

        # print(f"\n Fetching item with index: {idx}")

        cur_audio_dict  = self.examples[self.examples_list[idx]]
        cur_audio_path  = cur_audio_dict["audio_path"]
        audio           = cur_audio_dict["audio_data"]

        labels_df       = cur_audio_dict["labels_df"]
        binary_label    = cur_audio_dict["record_bin_label"]

        audio_dur       =  cur_audio_dict["audio_dur"]
        vad_timestamps = cur_audio_dict['vad_timestamps']

        chest_pos       = cur_audio_dict["chest_position"]
       # gender_info     = cur_audio_dict["gender"]


        anchor_intervals = cur_audio_dict["anchor_intervals"]
        assignments = cur_audio_dict["assignments"]

        sample_assignments = self.assignments[idx]
        conf_targets = torch.tensor([a['conf'] for a in sample_assignments], dtype=torch.float)
        cls_targets = torch.tensor([a['cls'] for a in sample_assignments], dtype=torch.long)
        box_targets = torch.tensor([a['box'] for a in sample_assignments], dtype=torch.float)
        # frame_labels  = cur_audio_dict["frame_labels"]


        #  对原始的音频进行数据增强,
        #  只在训练阶段， 并且只对异常样本进行数据增强，
        # if  self.train_flag and binary_label == 'Abnormal':    #  only use data aug  at the training stage;
        #
        # 用于实现对原始的音频数据，在音频层面进行数据增强；
        # 则意味着，一个batch中的每个样本都会使用不同的数据增强，从而提高模型的泛化能力；
        # 需要注意的，数据增强只对训练集使用， 验证集，　测试集不可以使用；
        # 包含了2大部分，
        # 1.分解重构，　2.音频的加噪， 响度， 声带扰动, 音高；
        # 3.　roll, time shift， 加减速， 改变音频的持续时间， 以及音频的顺序这类数据数据增强不能使用， 因为这会是检测任务 方式；


        if self.train_flag and binary_label == 'Abnormal':
            # Convert to numpy array if needed for augmentation
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio, dtype=np.float32)

            # DWT reconstruction augmentation
            if self.audio_aug and random.random() > 0.85:
                dwt = DWT1DForward(wave='db6', J=3)
                idwt = DWT1DInverse(wave='db6')

                audio_tensor = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)
                yl, yh = dwt(audio_tensor)
                audio_tensor = idwt((yl, yh))
                audio = audio_tensor.squeeze(0).squeeze(0).numpy()

            # General audio augmentation
            if self.audio_aug and random.random() > 0.85:
                audio = gen_augmented(audio, self.fs)
                audio = np.array(audio, dtype=np.float32)

        # print(f"\n audio length {len(audio)}")
       # check_array(audio)
        #audio = torch.tensor(audio)
        # Final conversion to PyTorch tensor
        if isinstance(audio, torch.Tensor):
            audio = audio.clone().detach().to(dtype=torch.float32)
        else:
            audio = torch.from_numpy(audio).to(torch.float32)


        #=========================  #  语谱图特征生成以及帧级别的标签生成之后================================
        # use this function to extract features in the dataloader and apply possibly some data augm
        # The output format is (channel=3, fea_dim=84, frames=variable, )
        #audio = audio.to("cuda")
        spectrogram = self.spec_extract_features(audio, binary_label) #  此时，生成的单个样本是 （3, 84,656,）# (chan =3, n_mels=84,  frames= variable)
        spectrogram = spectrogram.permute(0,2,1 ) #(channel=3, fea_dim=84, frames=variable, ) --> (chann, frames, fea_dim)

        # # 这里需要注意， 将frame labels 帧数的标签， 和 spectorgram 语谱图中的帧数进行对应起来；
        cur_audio_frames = spectrogram.size(1)

        # check if labels exists:  将时间标签，转化为对应的帧标签；
        if not len(labels_df):
            max_len_targets = self.encoder.n_frames
            frame_labels = torch.zeros(max_len_targets, len(self.encoder.labels)).float()
        else:
            #  Generate frame-level labels without downsampling
            # note， 该函数揭示了， 将时间级别的标签转化为 帧标签级别的精髓；  (onset, offset)  ---> 转化到对应的帧数；
            strong = self.encoder.encode_variable_df(labels_df,cur_audio_frames) # ndarray（156， 10） 获取帧级别的标签，
            frame_labels = torch.from_numpy(strong).float() # (nframes, n_cls)
            # frame_labels = frame_labels.transpose(0,1)  # ndarray（ n_cls=7, frames=577 or 961，）


        # Map string label to integer
        if binary_label == "Normal":
            binary_label = 0
        elif binary_label == "Abnormal":
            binary_label = 1
        else:
            raise ValueError(f"Unexpected binary label: {binary_label}")



        binary_label = torch.tensor(binary_label, dtype=torch.long)
        chest_pos = torch.tensor(chest_pos, dtype=torch.float)
       # gender_info = torch.tensor(gender_info, dtype=torch.float)
        audio_dur =  torch.tensor(audio_dur, dtype=torch.float)

        out_args  = {
            'spectrogram': spectrogram,
            'frame_labels': frame_labels,

            'c_ex_mixture': cur_audio_path,
            'record_binary_label': binary_label,
            "vad_timestamps": vad_timestamps,


            "chest_pos":chest_pos,
            # "gender_info":gender_info,
            "audio_duration": audio_dur,

            "anchor_intervals":anchor_intervals,
            "assignments": assignments,

            'conf_targets': conf_targets,
            'cls_targets': cls_targets,
            'box_targets': box_targets,
        }


        if self.embeddings_hdf5_file is not None:
            name = Path(c_ex["mixture"]).stem
            index = self.ex2emb_idx[name]

            if self.embedding_type == "global":
                embeddings = torch.from_numpy(
                    self.hdf5_file["global_embeddings"][index]
                ).float()
            elif self.embedding_type == "frame":
                embeddings = torch.from_numpy(
                    np.stack(self.hdf5_file["frame_embeddings"][index])
                ).float()
            else:
                raise NotImplementedError

            out_args.append(embeddings)
        # if self.mask_events_other_than is not None:
        #     out_args.append(self.mask_events_other_than)

        # list:{ spectrogram, label=(cls=7, variable frames), audio_name, embedings,  mask_events,  }
        return out_args



    def read_audio(self, file, multisrc, random_channel, pad_to=None, test=False):
        mixture, fs = torchaudio.load(file)

        if not multisrc:  #
            mixture = to_mono(mixture, random_channel)

        if pad_to is not None:  # 如果指定了pad_to长度，则对音频进行填充
            mixture, onset_s, offset_s, padded_indx = pad_audio(mixture, pad_to, fs, test=test)
        else:
            padded_indx = [1.0]  # 否则设置填充索引为[1.0]，起始和结束时间为None。
            # onset_s = None
            # offset_s = None

            onset_s = 0.000
            offset_s = round(onset_s + ( len(mixture) / fs), 3)  # 主要用于音频或信号处理中的时间偏移计算。

        mixture = mixture.float()
        return mixture, onset_s, offset_s, padded_indx



    def to_mono(self, mixture, random_ch=False):
        if mixture.ndim > 1:
            if not random_ch:
                mixture = torch.mean(mixture, 0)
            else:
                indx = np.random.randint(0, mixture.shape[0] - 1)
                mixture = mixture[indx]
        return mixture

    def compute_iou_matrix(self,pred_boxes, gt_boxes):
        """
        Compute IoU between each pair of anchor interval  and GT intervals.

        Args:
            pred_boxes (torch.Tensor): Shape (N, 2), [[start1, end1], ...]
            gt_boxes (torch.Tensor): Shape (M, 2), [[start1, end1], ...]

        Returns:
            torch.Tensor: IoU matrix of shape (N, M)
        """
        N = pred_boxes.size(0)
        M = gt_boxes.size(0)
        pred_starts = pred_boxes[:, 0].unsqueeze(1)  # (N, 1)
        pred_ends = pred_boxes[:, 1].unsqueeze(1)  # (N, 1)
        gt_starts = gt_boxes[:, 0].unsqueeze(0)  # (1, M)
        gt_ends = gt_boxes[:, 1].unsqueeze(0)  # (1, M)

        inter_start = torch.max(pred_starts, gt_starts)  # (N, M)
        inter_end = torch.min(pred_ends, gt_ends)  # (N, M)
        inter_len = (inter_end - inter_start).clamp(min=0)  # Intersection length

        pred_len = pred_ends - pred_starts
        gt_len = gt_ends - gt_starts
        union_len = pred_len + gt_len - inter_len  # Union length

        iou = inter_len / union_len.clamp(min=1e-6)  # Avoid division by zero
        return iou  # Shape (N, M)




    def spec_extract_features(self, audio, binary_label):
        # Placeholder for feature extraction logic
        # Convert raw audio to spectrogram format (channel=3, frames=variable, fea_dim=128)
        # spectrogram = torchaudio.transforms.MelSpectrogram(fs=self.fs, n_mels=128)(audio)

        # ----------- 生成语谱图类别的特征 ----------------------------
        mel_spec = self.mel_spec(audio)  # (n_mels=84, frames = 626)
        mel_spec = self.powerToDB(mel_spec)
        # note, 对语谱图分别在行维度上进行标准化；
        mel_norm = standardize_rows(mel_spec)
        mel_norm = mel_norm.unsqueeze(0)  # (1, 84, 626)

        cur_cqt = self.cqt_spectrogram(audio)  # (1, bins, frames =626 )
        cur_cqt = self.amplitudeToDB(cur_cqt)
        cur_cqt = cur_cqt.squeeze(0)
        cqt_norm = standardize_rows(cur_cqt)
        cqt_norm = cqt_norm.unsqueeze(0)

        cur_gamma = self.gamma_spec(audio)  # (1, n_mels, frames= 626)
        cur_gamma = self.powerToDB(cur_gamma)
        cur_gamma = cur_gamma.squeeze(0)
        gamma_norm = standardize_rows(cur_gamma)
        gamma_norm = gamma_norm.unsqueeze(0)


        spec_Aug = random.random()
        # note， 在训练阶段， 随机的对语谱图特征，在频率与时间维度上，进行随机的掩码；
        if self.train_flag and binary_label == "Abnormal"  and spec_Aug > 0.5  :
            mel_norm   = self.spec_aug(mel_norm)
            cqt_norm   = self.spec_aug(cqt_norm)
            gamma_norm = self.spec_aug(gamma_norm)

        # (chan =3, n_mels=84,  frames= variable)
        hybrid_spectrogram  = torch.cat((cqt_norm, gamma_norm, mel_norm,), dim=0)
        # 对语谱图特征在通道维度上进行标准化；

        return hybrid_spectrogram





    def get_node_label(self, frames, normal_label=0, abnormal_threshold=0.2):

        frames_label = frames.argmax(dim=0) # Shape: (n_frames,)
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



# 设置一个均衡采样器， 用于保证一个batch 中包含不同类型的样本；





import torch
import pandas as pd
from torch.utils.data import DataLoader

# Mock Encoder class to provide labels and other metadata needed
