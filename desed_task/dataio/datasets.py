import glob
import os
import random
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset


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


def process_labels(df, onset, offset): # 该函数是针对重对齐的音频（即不足或超过10s的），进行起始，终止时间标签的重新计算的
    df["onset"] = df["onset"] - onset   #  将DataFrame df 中的 "onset" 和 "offset" 列减去 onset 值
    df["offset"] = df["offset"] - onset #  确保 "onset" 列值不小于0，"offset" 列值不大于10。
    df["onset"] = df.apply(lambda x: max(0, x["onset"]), axis=1)
    df["offset"] = df.apply(lambda x: min(10, x["offset"]), axis=1) # 保留 "onset" 小于 "offset" 的行，并去除重复行后返回

    df_new = df[(df.onset < df.offset)]
    return df_new.drop_duplicates()


def read_audio(file, multisrc, random_channel, pad_to, test=False):
    mixture, fs = torchaudio.load(file)

    if not multisrc: #
        mixture = to_mono(mixture, random_channel)

    if pad_to is not None:#如果指定了pad_to长度，则对音频进行填充
        mixture, onset_s, offset_s, padded_indx = pad_audio(mixture, pad_to, fs, test=test)
    else:
        padded_indx = [1.0] # 否则设置填充索引为[1.0]，起始和结束时间为None。
        onset_s = None
        offset_s = None

    mixture = mixture.float()
    return mixture, onset_s, offset_s, padded_indx


class StronglyAnnotatedSet(Dataset):
    def __init__(
        self,
        audio_folder,
        tsv_entries,
        encoder,
        pad_to=10,
        fs=16000,
        return_filename=False,
        random_channel=False,
        multisrc=False,
        feats_pipeline=None,
        embeddings_hdf5_file=None,
        embedding_type=None,
        mask_events_other_than=None,
        test=False,
    ):
        self.encoder = encoder
        self.fs = fs
        self.pad_to = pad_to * fs
        self.return_filename = return_filename
        self.random_channel = random_channel
        self.multisrc = multisrc
        self.feats_pipeline = feats_pipeline
        self.embeddings_hdf5_file = embeddings_hdf5_file
        self.embedding_type = embedding_type
        self.test = test
        # 如果指定了mask_events_other_than，则创建一个掩码向量，将不在指定列表中的事件类别标记为无效（值为0）。
        # we mask events that are incompatible with the current setting
        if mask_events_other_than is not None:
            # fetch indexes to mask
            self.mask_events_other_than = torch.ones(len(encoder.labels))
            for indx, cls in enumerate(encoder.labels):
                if cls not in mask_events_other_than:
                    # set to zero corresponding entry, invalid class for this dataset
                    # we will skip loss computation
                    self.mask_events_other_than[indx] = 0
        else:
            # keep all, no mask， 如果未指定，则所有事件类别都有效（值为1）
            self.mask_events_other_than = torch.ones(len(encoder.labels))
        self.mask_events_other_than = self.mask_events_other_than.bool()
        assert embedding_type in [
            "global",
            "frame",
            None,
        ], "embedding type are either frame or global or None, got {}".format(
            embedding_type
        )

        tsv_entries = tsv_entries.dropna() # 该函数调用将删除tsv_entries DataFrame中所有含有空值（NaN）的行

        examples = {}
        for i, r in tsv_entries.iterrows(): # 将数据按文件名分组，构造包含混合音频路径和事件信息的字典examples
            if r["filename"] not in examples.keys():  # 若r["filename"]不在examples字典中，则创建一个新的字典条目，并设置音频文件路径、事件列表和置信度
                confidence = 1.0 if "confidence" not in r.keys() else r["confidence"]
                examples[r["filename"]] = {
                    "mixture": os.path.join(audio_folder, r["filename"]),
                    "events": [],
                    "confidence": confidence,
                }
                if not np.isnan(r["onset"]): # 若r["onset"]不是NaN值，则根据r中的信息添加一个事件到对应文件的事件列表中，并设置事件的标签、起始时间、结束时间和置信度
                    confidence = (
                        1.0 if "confidence" not in r.keys() else r["confidence"]
                    )
                    examples[r["filename"]]["events"].append(
                        {
                            "event_label": r["event_label"],
                            "onset": r["onset"],
                            "offset": r["offset"],
                            "confidence": confidence,
                        }
                    )
            else:  # 若r["filename"]已在examples中，则直接在对应的事件列表中添加事件，条件同样是r["onset"]不是NaN值。
                if not np.isnan(r["onset"]):
                    confidence = (
                        1.0 if "confidence" not in r.keys() else r["confidence"]
                    )
                    examples[r["filename"]]["events"].append(
                        {
                            "event_label": r["event_label"],
                            "onset": r["onset"],
                            "offset": r["offset"],
                            "confidence": confidence,
                        }
                    )

        # we construct a dictionary for each example，
        #  对应10000 份音频， 3万多个事件； 使用字典形式保存每份音频， 每份音频中的事件使用列表存储；
        ## dict{2500 *  dict{mixture: str"audio_path", events: list: 当前音频中的事件个数 * dict{conf:1, event_label:str,  onset: float, offset:float } ， conf: }  }
        self.examples = examples
        self.examples_list = list(examples.keys()) # 提取出每份音频的文件名，存储在列表中；

        if self.embeddings_hdf5_file is not None:
            assert ( # 如果提供了embeddings_hdf5_file，则打开HDF5文件，并构建每个音频文件名到其在嵌入文件中位置的映射
                self.embedding_type is not None
            ), "If you use embeddings you need to specify also the type (global or frame)"
            # fetch dict of positions for each example
            self.ex2emb_idx = {}
            f = h5py.File(self.embeddings_hdf5_file, "r")
            for i, fname in enumerate(f["filenames"]):
                self.ex2emb_idx[fname.decode("UTF-8")] = i
        self._opened_hdf5 = None

    def __len__(self):
        return len(self.examples_list)

    @property
    def hdf5_file(self):
        if self._opened_hdf5 is None:
            self._opened_hdf5 = h5py.File(self.embeddings_hdf5_file, "r")
        return self._opened_hdf5

    def __getitem__(self, item):
        c_ex = self.examples[self.examples_list[item]] # dict{ mixture: str"audio_path", events: list: 当前音频中的事件个数 * dict{conf:1, event_label:str,  onset: float, offset:float } ， conf: 1}  }
        mixture, onset_s, offset_s, padded_indx = read_audio(
            c_ex["mixture"], self.multisrc, self.random_channel, self.pad_to, self.test,
        )

        # labels
        labels = c_ex["events"] #  提取出其中声音事件的信息；

        # to steps，    处理标签信息并转换为适合模型输入的格式
        labels_df = pd.DataFrame(labels)
        labels_df = process_labels(labels_df, onset_s, offset_s) # 根据onset 重新标定每个音频的事件的起始，终止时间；

        # check if labels exists:
        if not len(labels_df):
            max_len_targets = self.encoder.n_frames
            strong = torch.zeros(max_len_targets, len(self.encoder.labels)).float()
        else:
            strong = self.encoder.encode_strong_df(labels_df) # ndarray（156， 10） 获取帧级别的标签， # note， 该函数揭示了， 将时间级别的标签转化为 帧标签级别的精髓；
            strong = torch.from_numpy(strong).float()

        out_args = [mixture, strong.transpose(0, 1), padded_indx]

        if self.feats_pipeline is not None:
            # use this function to extract features in the dataloader and apply possibly some data augm
            feats = self.feats_pipeline(mixture)
            out_args.append(feats)
        if self.return_filename:
            out_args.append(c_ex["mixture"])

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

        if self.mask_events_other_than is not None:
            out_args.append(self.mask_events_other_than)

        return out_args  # list:{ audio,  (10,156), padded_indx, mask_events,  }


class WeakSet(Dataset):
    def __init__(
        self,
        audio_folder,
        tsv_entries,
        encoder,
        pad_to=10,
        fs=16000,
        return_filename=False,
        random_channel=False,
        multisrc=False,
        feats_pipeline=None,
        embeddings_hdf5_file=None,
        embedding_type=None,
        mask_events_other_than=None,
        test=False,
    ):
        self.encoder = encoder
        self.fs = fs
        self.pad_to = pad_to * fs
        self.return_filename = return_filename
        self.random_channel = random_channel
        self.multisrc = multisrc
        self.feats_pipeline = feats_pipeline
        self.embeddings_hdf5_file = embeddings_hdf5_file
        self.embedding_type = embedding_type
        self.mask_events_other_than = mask_events_other_than
        self.test = test

        if mask_events_other_than is not None:
            # fetch indexes to mask
            self.mask_events_other_than = torch.ones(len(encoder.labels))
            for indx, cls in enumerate(encoder.labels):
                if cls not in mask_events_other_than:
                    # set to zero corresponding entry, invalid class for this dataset
                    # we will skip loss computation
                    self.mask_events_other_than[indx] = 0
        else:
            # keep all, no mask
            self.mask_events_other_than = torch.ones(len(encoder.labels))

        self.mask_events_other_than = self.mask_events_other_than.bool()
        assert embedding_type in [
            "global",
            "frame",
            None,
        ], "embedding type are either frame or global or None, got {}".format(
            embedding_type
        )
        # 弱标签， 则存储的是文件名，以及该音频对应的声音事件；
        examples = {}
        for i, r in tsv_entries.iterrows(): # 遍历tsv_entries中的每一行数据
            if r["filename"] not in examples.keys():
                examples[r["filename"]] = {
                    "mixture": os.path.join(audio_folder, r["filename"]),
                    "events": r["event_labels"].split(","),
                } # 如果当前行的filename不在examples字典的键中，则将该filename作为新键添加到examples字典中，并将其值设置为一个新的字典。这个新的字典包含两个键值对："mixture"对应音频文件路径，"events"对应事件标签列表。

        self.examples = examples # 字典，训练集 1420 份，弱标签训练样本； 验证集： 158份；
        self.examples_list = list(examples.keys()) # list, 用于存储对应音频的文件名称；

        if self.embeddings_hdf5_file is not None:
            assert (
                self.embedding_type is not None
            ), "If you use embeddings you need to specify also the type (global or frame)"
            # fetch dict of positions for each example
            self.ex2emb_idx = {}
            f = h5py.File(self.embeddings_hdf5_file, "r")
            for i, fname in enumerate(f["filenames"]):
                self.ex2emb_idx[fname.decode("UTF-8")] = i
        self._opened_hdf5 = None

    def __len__(self):
        return len(self.examples_list)

    @property
    def hdf5_file(self):
        if self._opened_hdf5 is None:
            self._opened_hdf5 = h5py.File(self.embeddings_hdf5_file, "r")
        return self._opened_hdf5

    def __getitem__(self, item):
        file = self.examples_list[item]
        c_ex = self.examples[file]

        mixture, _, _, padded_indx = read_audio(
            c_ex["mixture"], self.multisrc, self.random_channel, self.pad_to, self.test
        )

        # labels
        labels = c_ex["events"]
        # check if labels exists:
        max_len_targets = self.encoder.n_frames
        weak = torch.zeros(max_len_targets, len(self.encoder.labels))
        if len(labels):
            weak_labels = self.encoder.encode_weak(labels)
            weak[0, :] = torch.from_numpy(weak_labels).float()

        out_args = [mixture, weak.transpose(0, 1), padded_indx]

        if self.feats_pipeline is not None:
            feats = self.feats_pipeline(mixture)
            out_args.append(feats)

        if self.return_filename:
            out_args.append(c_ex["mixture"])

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

        if self.mask_events_other_than is not None:
            out_args.append(self.mask_events_other_than)

        return out_args


class UnlabeledSet(Dataset):
    def __init__(
        self,
        unlabeled_folder,
        encoder,
        pad_to=10,
        fs=16000,
        return_filename=False,
        random_channel=False,
        multisrc=False,
        feats_pipeline=None,
        embeddings_hdf5_file=None,
        embedding_type=None,
        mask_events_other_than=None,
        test=False,
    ):
        self.encoder = encoder
        self.fs = fs
        self.pad_to = pad_to * fs if pad_to is not None else None
        self.examples = glob.glob(os.path.join(unlabeled_folder, "*.wav"))
        self.return_filename = return_filename
        self.random_channel = random_channel
        self.multisrc = multisrc
        self.feats_pipeline = feats_pipeline
        self.embeddings_hdf5_file = embeddings_hdf5_file
        self.embedding_type = embedding_type
        self.test = test
        assert embedding_type in [
            "global",
            "frame",
            None,
        ], "embedding type are either frame or global or None, got {}".format(
            embedding_type
        )

        self.mask_events_other_than = mask_events_other_than
        #  # 如果指定了mask_events_other_than，则创建一个掩码向量，将不在指定列表中的事件类别标记为无效（值为0）。
        if mask_events_other_than is not None:
            # fetch indexes to mask
            self.mask_events_other_than = torch.ones(len(encoder.labels))
            for indx, cls in enumerate(encoder.labels):
                if cls not in mask_events_other_than:
                    # set to zero corresponding entry, invalid class for this dataset
                    # we will skip loss computation
                    self.mask_events_other_than[indx] = 0
        else:
            # keep all, no mask
            self.mask_events_other_than = torch.ones(len(encoder.labels))

        self.mask_events_other_than = self.mask_events_other_than.bool()

        if self.embeddings_hdf5_file is not None:
            assert ( # 如果指定了嵌入文件，则读取嵌入文件并建立文件名到索引的映射
                self.embedding_type is not None
            ), "If you use embeddings you need to specify also the type (global or frame)"
            # fetch dict of positions for each example
            self.ex2emb_idx = {}
            f = h5py.File(self.embeddings_hdf5_file, "r")
            for i, fname in enumerate(f["filenames"]):
                self.ex2emb_idx[fname.decode("UTF-8")] = i
        self._opened_hdf5 = None

    def __len__(self):
        return len(self.examples)

    @property
    def hdf5_file(self):
        if self._opened_hdf5 is None:
            self._opened_hdf5 = h5py.File(self.embeddings_hdf5_file, "r")
        return self._opened_hdf5

    def __getitem__(self, item):
        c_ex = self.examples[item]

        mixture, _, _, padded_indx = read_audio(
            c_ex, self.multisrc, self.random_channel, self.pad_to, self.test
        )

        max_len_targets = self.encoder.n_frames
        strong = torch.zeros(max_len_targets, len(self.encoder.labels)).float()
        out_args = [mixture, strong.transpose(0, 1), padded_indx]
        if self.feats_pipeline is not None:
            feats = self.feats_pipeline(mixture)
            out_args.append(feats)

        if self.return_filename:
            out_args.append(c_ex)

        if self.embeddings_hdf5_file is not None:
            name = Path(c_ex).stem
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

        if self.mask_events_other_than is not None:
            out_args.append(self.mask_events_other_than)

        return out_args
