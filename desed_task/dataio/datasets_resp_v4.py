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
# from torch_geometric.data import Dataset, Data

from desed_task.utils.data_spec_aug import gen_augmented, standardize_rows, GroupSpecAugment_v2



# prepare for the  data aug
from pytorch_wavelets import DWT1DForward, DWT1DInverse

# prep for gen  spectrogram and  spec aug
from torchaudio.transforms import  AmplitudeToDB, MelSpectrogram
from  nnAudio.features import  CQT2010v2, Gammatonegram

def check_array(array):
    # Ensure the input is a numpy array
    assert isinstance(array, np.ndarray), f"Expected np.ndarray, got {type(array)}"

    # Check if the array is not empty
    assert array.size > 0, f"Array is empty. Size: {array.size}"

    # Additional condition: check a specific shape or condition
    # For example, ensure it's a 1D array
    assert array.ndim == 1, f"Expected 1D array, got {array.ndim}D array"



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
    df["offset"] = df.apply(lambda x: min(16, x["offset"]), axis=1) # 保留 "onset" 小于 "offset" 的行，并去除重复行后返回

    df_new = df[(df.onset < df.offset)]
    return df_new.drop_duplicates()


# def read_audio(file, multisrc, random_channel, pad_to, test=False):
#     mixture, fs = torchaudio.load(file)
#
#     if not multisrc: #
#         mixture = to_mono(mixture, random_channel)
#
#     if pad_to is not None:#如果指定了pad_to长度，则对音频进行填充
#         mixture, onset_s, offset_s, padded_indx = pad_audio(mixture, pad_to, fs, test=test)
#     else:
#         padded_indx = [1.0] # 否则设置填充索引为[1.0]，起始和结束时间为None。
#         onset_s = None
#         offset_s = None
#
#     mixture = mixture.float()
#     return mixture, onset_s, offset_s, padded_indx




from torch.utils.data import Dataset
from tqdm import  tqdm
import  datetime

class RespiraGnnSet(Dataset):
    def __init__(
        self,
        audio_folder,
        tsv_entries,
        encoder,

        fs=8000,
        n_fft =1024,
        win_len = 1024,
        hop_len = 128,
        f_max = 8000,
        n_filters = 84,

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
        tsv_entries = tsv_entries.dropna()

        examples = {}
        for i, r in tsv_entries.iterrows():
            if r["filename"] not in examples.keys():
                confidence = 1.0 if "confidence" not in r.keys() else r["confidence"]
                examples[r["filename"]] = {
                    "audio_path": os.path.join(audio_folder, r["filename"]),
                    "events": [],
                    "confidence": confidence,
                }
                if not pd.isna(r["onset"]):
                    confidence = 1.0 if "confidence" not in r.keys() else r["confidence"]
                    examples[r["filename"]]["events"].append(
                        {
                            "event_label": r["event_label"],
                            "onset": r["onset"],
                            "offset": r["offset"],
                            "confidence": confidence,
                        }
                    )
            else:
                if not pd.isna(r["onset"]):
                    confidence = 1.0 if "confidence" not in r.keys() else r["confidence"]
                    examples[r["filename"]]["events"].append(
                        {
                            "event_label": r["event_label"],
                            "onset": r["onset"],
                            "offset": r["offset"],
                            "confidence": confidence,
                        }
                    )

        self.examples = examples
        self.examples_list = list(examples.keys())



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
            f_min=32.7,
            f_max=f_max,
            n_mels=n_filters,
        )  #.to("cuda")

        # self.cqt_spectrogram =
        # 这里设置的　cqt　内核使用默认的，　trainable=False;
        self.cqt_spectrogram = CQT2010v2(
            sr=fs,
            hop_length=hop_len,  # hop_len,
            fmin=32.7,  # 32.7,
            fmax=f_max,  # f_max,
            n_bins=84,  # 这里设置成84 为了和 Mel 滤波器个数对应上；
            trainable=False,
        )# .to("cuda")

        self.gamma_spec = Gammatonegram(
            sr=fs,
            n_fft=n_fft,
            hop_length=hop_len,
            n_bins=84,
            fmin=32.7,
            fmax=f_max,
            trainable_bins=False,
        )# .to("cuda")





        #  重新遍历 self.examples, 将 audio,  labels_df 这两个属性加入到每个样本的字典中；
        start_time = datetime.datetime.now()
        for id, audio_name in enumerate(tqdm(self.examples,desc='generate the frame labels,' )):

            # cur_audio_name = os.path.basename(cur_audio_path)
            cur_audio_name = audio_name
            cur_dict = self.examples[audio_name]
            cur_audio_path =  cur_dict["audio_path"]


            audio, onset_s, offset_s, padded_indx = self.read_audio(
                cur_audio_path, self.multisrc, self.random_channel, None, self.test
            )



            #
            # dict["events"]: 代表当前音频所包含的事件个数，以及详细标注信息；
            # labels,
            labels = cur_dict["events"]  # 提取出其中声音事件的信息；
            labels_df = pd.DataFrame(labels)
            labels_df = process_labels(labels_df, onset_s, offset_s)  # 根据onset 重新标定每个音频的事件的起始，终止时间；

            # # 这里先使用一个语谱图， 生成每个audio 样本， 对应的语谱图上的 每帧的标签；
            # # 提前生成帧级别的标签的目的，  减少 self.getitem 中花费的时间，加快训练速度；
            #
            # temp_spec = self.mel_spec(audio)
            # temp_spec = temp_spec.unsqueeze(0)  # (1, 84,  cur_audio_frames)
            # temp_spec = temp_spec.permute(0,2,1) # (channel,  frames,  fea_dim )
            #
            # # 这里需要注意， 将frame labels 帧数的标签， 和 spectorgram 语谱图中的帧数进行对应起来；
            # cur_audio_frames =  temp_spec.size(1)
            #
            # # check if labels exists:  将时间标签，转化为对应的帧标签；
            # if not len(labels_df):
            #     max_len_targets = self.encoder.n_frames
            #     frame_labels = torch.zeros(max_len_targets, len(self.encoder.labels)).float()
            # else:
            #     #  Generate frame-level labels without downsampling
            #     # note， 该函数揭示了， 将时间级别的标签转化为 帧标签级别的精髓；
            #     strong = self.encoder.encode_variable_df(labels_df, cur_audio_frames)  # ndarray（156， 10） 获取帧级别的标签，
            #     frame_labels = torch.from_numpy(strong).float()
            #     frame_labels = frame_labels.transpose(0, 1)  # ndarray（ n_cls=10, 156，）
            #
            #
            # # 即然帧标签可以在这里生成， 那么对应的节点标签能否consider  也在这里生成。
            #
            # new_items = { "label_df": labels_df,  "audio_data": audio,
            #               "frame_labels": frame_labels}

            new_items = {  "audio_data": audio, "label_df": labels_df,  }


            self.examples[cur_audio_name].update(new_items)


        end_time = datetime.datetime.now()
        duration = end_time - start_time
        print(f"\n To get the audio data cost time: {duration}")



        print(" \n The  stage Dataset init Done !! ")

    def __len__(self):
        return len(self.examples_list)

    def __getitem__(self, idx):
        cur_audio_dict  = self.examples[self.examples_list[idx]]

        cur_audio_path = cur_audio_dict["audio_path"]
        audio          =  cur_audio_dict["audio_data"]
        labels_df      = cur_audio_dict["label_df"]
        # frame_labels  = cur_audio_dict["frame_labels"]


        # 用于对原始的音频进行数据增强，或者提取特征；
        #
        if  self.train_flag: #  only use  data aug  at the training stage;

            # 用于实现对原始的音频数据，在音频层面进行数据增强；
            # 则意味着，一个batch中的每个样本都会使用不同的数据增强，从而提高模型的泛化能力；
            # 需要注意的，数据增强只对训练集使用， 验证集，　测试集不可以使用；
            # 包含了2大部分，
            # 1.分解重构，　2.音频的加噪， 响度， 声带扰动，　音高；
            # 3.　roll, time shift， 加减速， 改变音频的持续时间， 以及音频的顺序这类数据数据增强不能使用， 因为这会是检测任务 方式；

            reconstruct_prob = random.random()  # 对每个训练样本使用DWT进行分解重构的概率；
            if self.audio_aug  and reconstruct_prob > 0.5:
                dwt = DWT1DForward(wave='db6', J=3)
                idwt = DWT1DInverse(wave='db6')
                # print(m[0].shape)
                # audio = torch.tensor(audio)
                # If `audio` is already a tensor and you want a separate copy
                if isinstance(audio, torch.Tensor):
                    audio = audio.clone().detach()
                else:
                    # If `audio` is not a tensor, convert it
                    audio = torch.tensor(audio, dtype=torch.float32)

                audio = audio.unsqueeze(0).unsqueeze(0)
                yl, yh = dwt(audio)
                audio = idwt((yl, yh))
                audio = audio.squeeze(0).squeeze(0)
                audio = audio.numpy()
                #print(f"\n after using DWT {len(audio)} ")

            aug_prob = random.random()
            if self.audio_aug  and aug_prob > 0.5:
                #audio = audio.numpy()
                # print(f"\n before gen aug {len(audio)}")
                # print(f"\n before gen aug ,audio's type { audio.type()}")
                if not isinstance(audio, np.ndarray):
                    audio = np.array(audio, dtype=np.float32)

                audio = gen_augmented(audio, self.fs)
                #print(f"\n after using gen audio aug {len(audio)} ")
                audio = np.array(audio, dtype=np.float32)
                # 将会改变音频的长度的数据增强去除，故需要重新对齐到统一长度；

            reconstruct_prob = random.random()  # 对每个训练样本使用DWT进行分解重构的概率；
            if self.audio_aug  and reconstruct_prob > 0.5:
                dwt = DWT1DForward(wave='db6', J=3)
                idwt = DWT1DInverse(wave='db6')
                # print(m[0].shape)
                # audio = torch.tensor(audio)
                # If `audio` is already a tensor and you want a separate copy
                if isinstance(audio, torch.Tensor):
                    audio = audio.clone().detach()
                else:
                    # If `audio` is not a tensor, convert it
                    audio = torch.tensor(audio, dtype=torch.float32)

                audio = audio.unsqueeze(0).unsqueeze(0)
                yl, yh = dwt(audio)
                audio = idwt((yl, yh))
                audio = audio.squeeze(0).squeeze(0)
                audio = audio.numpy()
                #print(f"\n after using DWT {len(audio)} ")

            aug_prob = random.random()
            if self.audio_aug  and aug_prob > 0.5:
                #audio = audio.numpy()
                # print(f"\n before gen aug {len(audio)}")
                # print(f"\n before gen aug ,audio's type { audio.type()}")
                if not isinstance(audio, np.ndarray):
                    audio = np.array(audio, dtype=np.float32)

                audio = gen_augmented(audio, self.fs)
                #print(f"\n after using gen audio aug {len(audio)} ")
                audio = np.array(audio, dtype=np.float32)
                # 将会改变音频的长度的数据增强去除，故需要重新对齐到统一长度；



        # print(f"\n audio length {len(audio)}")
       # check_array(audio)
        #audio = torch.tensor(audio)
        # If `audio` is already a tensor and you want a separate copy
        if isinstance(audio, torch.Tensor):
            audio = audio.clone().detach()
        else:
            # If `audio` is not a tensor, convert it
            audio = torch.tensor(audio, dtype=torch.float32) #.to('cuda')

        #=========================  #  语谱图特征生成以及帧级别的标签生成之后================================
        # use this function to extract features in the dataloader and apply possibly some data augm
        # The output format is (channel=3, fea_dim=84, frames=variable, )
        spectrogram = self.spec_extract_features(audio) #  此时，生成的单个样本是 （3, 84,656,）# (chan =3, n_mels=84,  frames= variable)
        spectrogram = spectrogram.permute(0,2,1 ) #(channel=3, fea_dim=84, frames=variable, ) --> (chann, frames, fea_dim)

        # # 这里需要注意， 将frame labels 帧数的标签， 和 spectorgram 语谱图中的帧数进行对应起来；
        cur_audio_frames = spectrogram.size(1)

        # check if labels exists:  将时间标签，转化为对应的帧标签；
        if not len(labels_df):
            max_len_targets = self.encoder.n_frames
            frame_labels = torch.zeros(max_len_targets, len(self.encoder.labels)).float()
        else:
            #  Generate frame-level labels without downsampling
            # note， 该函数揭示了， 将时间级别的标签转化为 帧标签级别的精髓；
            strong = self.encoder.encode_variable_df(labels_df,cur_audio_frames) # ndarray（156， 10） 获取帧级别的标签，
            frame_labels = torch.from_numpy(strong).float()
            frame_labels = frame_labels.transpose(0,1)  # ndarray（ n_cls=10, 156，）



        # Perform chunking and node label computation
        chunks = []
        node_labels = []
        n_frames = spectrogram.size(1)
        chunk_size = 5  # Define your chunk size
        for j in range(0, n_frames - chunk_size + 1, chunk_size):
            chunk = spectrogram[:, j:j + chunk_size, :]  # (channels, chunk_size, n_mels)
            chunks.append(chunk)

            # Compute node label for the chunk
            label_chunk = frame_labels[:, j:j + chunk_size]
            node_label = self.get_node_label(label_chunk)  # Implement this method to compute labels
            node_labels.append(node_label)




        out_args  = {
            #'spectrogram': spectrogram,
            'chunks': chunks,
            'node_labels': node_labels,
            'c_ex_mixture': cur_audio_path,
            # Include any other necessary data
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

    def spec_extract_features(self, audio):
        # Placeholder for feature extraction logic
        # Convert raw audio to spectrogram format (channel=3, frames=variable, fea_dim=128)
        # spectrogram = torchaudio.transforms.MelSpectrogram(fs=self.fs, n_mels=128)(audio)

        # ----------- 生成语谱图类别的特征 ----------------------------
        mel_spec = self.mel_spec(audio)  # (n_filters=84, frames = 626)
        mel_spec = self.powerToDB(mel_spec)
        # note, 对语谱图分别在行维度上进行标准化；
        mel_norm = standardize_rows(mel_spec)
        mel_norm = mel_norm.unsqueeze(0)  # (1, 84, 626)

        cur_cqt = self.cqt_spectrogram(audio)  # (1, bins, frames =626 )
        cur_cqt = self.amplitudeToDB(cur_cqt)
        cur_cqt = cur_cqt.squeeze(0)
        cqt_norm = standardize_rows(cur_cqt)
        cqt_norm = cqt_norm.unsqueeze(0)

        cur_gamma = self.gamma_spec(audio)  # (1, n_filters, frames= 626)
        cur_gamma = self.powerToDB(cur_gamma)
        cur_gamma = cur_gamma.squeeze(0)
        gamma_norm = standardize_rows(cur_gamma)
        gamma_norm = gamma_norm.unsqueeze(0)


        spec_Aug = random.random()
        # note， 在训练阶段， 随机的对语谱图特征，在频率与时间维度上，进行随机的掩码；
        if self.train_flag and spec_Aug > 0.5:
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




# def custom_collate(batch):
#     # Example collate function that converts spectrograms to Data objects
#     processed_batch = []
#     for spectrogram, label in batch:
#         features, edge_index = extract_features_and_edges(spectrogram)  # Placeholder for feature and edge extraction
#         node_labels = label.process_labels(label, spectrogram.shape[1] // 20)  # Convert labels for graph nodes
#         graph_data = Data(x=features, edge_index=edge_index, y=torch.tensor(node_labels))
#         processed_batch.append(graph_data)
#     return torch.stack(processed_batch)
#
# # Example usage:
# list_of_spectrograms = [...]  # Your spectrogram data
# list_of_labels = [...]  # Your label data
# dataset = RespiraGnnSet(list_of_spectrograms, list_of_labels)
# loader = DataLoader(dataset, batch_size=10, collate_fn=custom_collate)
#

