from collections import OrderedDict

import numpy as np
import pandas as pd
from dcase_util.data import DecisionEncoder

# wrapper around manyhotencoder to handle multiple heterogeneous class
# dsets


from collections import OrderedDict

import numpy as np
import pandas as pd
from dcase_util.data import DecisionEncoder

# wrapper around manyhotencoder to handle multiple heterogeneous class
# dsets


class ManyHotEncoder: # note, 初始化一个专门用于对音频的编码器；
    """ "
        Adapted after DecisionEncoder.find_contiguous_regions method in
        https://github.com/DCASE-REPO/dcase_util/blob/master/dcase_util/data/decisions.py

        Encode labels into numpy arrays where 1 correspond to presence of the class and 0 absence.
        Multiple 1 can appear on the same line, it is for multi label problem.
    Args:
        labels: list, the classes which will be encoded
        n_frames: int, (Default value = None) only useful for strong labels. The number of frames of a segment.
    Attributes:
        labels: list, the classes which will be encoded
        n_frames: int, only useful for strong labels. The number of frames of a segment.
    """

    def __init__(
        self, labels, audio_len, frame_len, frame_hop, net_pooling=1, fs=16000
    ):
        if type(labels) in [np.ndarray, np.array]:
            labels = labels.tolist()
        elif isinstance(labels, (dict, OrderedDict)):
            labels = list(labels.keys())
        self.labels = labels  # list , 10个类别；
        self.audio_len = audio_len  #  10s;
        self.frame_len = frame_len  #  2048,  1024 nfft 的点数；
        self.frame_hop = frame_hop  #  256
        self.fs = fs  # 16k
        self.net_pooling = net_pooling # 4
        n_frames = self.audio_len * self.fs  # 10秒长度下总的采样点数  160k；
        self.n_frames = int(int((n_frames / self.frame_hop)) / self.net_pooling)  # 625 帧/ 4 = 156 帧； 相当于每4帧 作为一组， 生成新的特征；
    def encode_weak(self, labels):
        """Encode a list of weak labels into a numpy array

        Args:
            labels: list, list of labels to encode (to a vector of 0 and 1)

        Returns:
            numpy.array
            A vector containing 1 for each label, and 0 everywhere else
        """
        # useful for tensor empty labels
        if type(labels) is str:
            if labels == "empty":
                y = np.zeros(len(self.labels)) - 1
                return y
            else:
                labels = labels.split(",")
        if type(labels) is pd.DataFrame:
            if labels.empty:
                labels = []
            elif "event_label" in labels.columns:
                labels = labels["event_label"]
        y = np.zeros(len(self.labels))
        for label in labels:
            if not pd.isna(label):
                i = self.labels.index(label)
                y[i] = 1
        return y

    def _time_to_frame(self, time):
        samples = time * self.fs   # 时间点数 * 采样点数， 计算对应的时间点数乘以采样点数，得到该时间点在声音信号中的样本点数。
        frame = (samples) / self.frame_hop # 将样本点数除以帧跳步 self.frame_hop，得到当前时间对应的帧位置。
        return np.clip(frame / self.net_pooling, a_min=0, a_max=self.n_frames)#对帧位置除以网络池化因子 self.net_pooling，然后使用 np.clip 函数确保结果在 [0, self.n_frames] 范围内，其中 self.n_frames 是最大帧数。

    def _frame_to_time(self, frame):
        frame = frame * self.net_pooling / (self.fs / self.frame_hop)#通过乘以 self.net_pooling 并除以 (self.fs / self.frame_hop) 来计算实际时间位置。
        return np.clip(frame, a_min=0, a_max=self.audio_len) # 将157 帧， 转成10秒 范围内对应的时间点；

    def encode_strong_df(self, label_df):
        """Encode a list (or pandas Dataframe or Serie) of strong labels, they correspond to a given filename

        Args:
            label_df: pandas DataFrame or Series, contains filename, onset (in frames) and offset (in frames)
                If only filename (no onset offset) is specified, it will return the event on all the frames
                onset and offset should be in frames
        Returns:
            numpy.array
            Encoded labels, 1 where the label is present, 0 otherwise
        """

        assert any(
            [x is not None for x in [self.audio_len, self.frame_len, self.frame_hop]]
        )

        samples_len = self.n_frames  # 统一帧数为 156，
        if type(label_df) is str:
            if label_df == "empty":
                y = np.zeros((samples_len, len(self.labels))) - 1
                return y
        y = np.zeros((samples_len, len(self.labels))) #初始化一个形状为(samples_len, len(self.labels))的零矩阵y作为输出结果。
        if type(label_df) is pd.DataFrame: # note， 该函数揭示了， 将时间级别的标签转化为 帧标签级别的精髓；
            if {"onset", "offset", "event_label"}.issubset(label_df.columns):
                for _, row in label_df.iterrows():
                    if not pd.isna(row["event_label"]):
                        i = self.labels.index(row["event_label"]) #如果"event_label"不为空，则获取其在self.labels中的索引i， 即属于10个类别中的第几个索引；
                        onset = int(self._time_to_frame(row["onset"])) #note, 将起始时间onset和结束时间offset转换为帧数
                        offset = int(np.ceil(self._time_to_frame(row["offset"])))  # 得到当前事件， 对应的起始帧数和结束帧数；
                        if "confidence" in label_df.columns:  # 如果DataFrame包含"confidence"列，则用该行的"confidence"值填充矩阵y对应位置；否则填充为1。
                            y[onset:offset, i] = row["confidence"]  # support confidence
                        else:
                            y[onset:offset, i] = (
                                1  # means offset not included (hypothesis of overlapping frames, so ok)
                            )   # 最终生成一个 (156,10)，   156行代表了每份音频是 156 帧的固定长度， 10列代表了， 每一帧在10个类别上是否存在该类别对应的声音事件。
        # 最终返回编码后的二维数组y，其中1表示事件存在，0表示不存在，如果有置信度信息，则使用置信度值代替1。
        elif type(label_df) in [
            pd.Series,
            list,
            np.ndarray,
        ]:  # list of list or list of strings
            if type(label_df) is pd.Series:
                if {"onset", "offset", "event_label"}.issubset(
                    label_df.index
                ):  # means only one value
                    if not pd.isna(label_df["event_label"]):
                        i = self.labels.index(label_df["event_label"])
                        onset = int(self._time_to_frame(label_df["onset"]))
                        offset = int(np.ceil(self._time_to_frame(label_df["offset"])))

                        if "confidence" in label_df.columns:
                            y[onset:offset, i] = label_df["confidence"]
                        else:
                            y[onset:offset, i] = 1
                    return y

            for event_label in label_df:
                # List of string, so weak labels to be encoded in strong
                if type(event_label) is str:
                    if event_label != "":
                        i = self.labels.index(event_label)
                        y[:, i] = 1

                # List of list, with [label, onset, offset]
                elif len(event_label) == 3:
                    if event_label[0] != "":
                        i = self.labels.index(event_label[0])
                        onset = int(self._time_to_frame(event_label[1]))
                        offset = int(np.ceil(self._time_to_frame(event_label[2])))
                        y[onset:offset, i] = 1
                # List of list, with [label, onset, offset, confidence]
                elif len(event_label) == 4:
                    if event_label[0] != "":
                        i = self.labels.index(event_label[0])
                        onset = int(self._time_to_frame(event_label[1]))
                        offset = int(np.ceil(self._time_to_frame(event_label[2])))
                        y[onset:offset, i] = event_label[3]

                else:
                    raise NotImplementedError(
                        "cannot encode strong, type mismatch: {}".format(
                            type(event_label)
                        )
                    )

        else:
            raise NotImplementedError(
                "To encode_strong, type is pandas.Dataframe with onset, offset and event_label"
                "columns, or it is a list or pandas Series of event labels, "
                "type given: {}".format(type(label_df))
            )
        return y

    def decode_weak(self, labels):
        """Decode the encoded weak labels
        Args:
            labels: numpy.array, the encoded labels to be decoded

        Returns:
            list
            Decoded labels, list of string

        """
        result_labels = []
        for i, value in enumerate(labels):
            if value == 1:
                result_labels.append(self.labels[i])
        return result_labels

    def decode_strong(self, labels):
        """Decode the encoded strong labels
        Args:
            labels: numpy.array, the encoded labels to be decoded
        Returns:
            list
            Decoded labels, list of list: [[label, onset offset], ...]

        """ # labels: (156, 10)
        result_labels = []  # 创建一个空列表result_labels，用于存储解码后的标签信息。
        for i, label_column in enumerate(labels.T): # labels.T: (10,156) 按照类别取出， 每一个类别156帧； lable_col: 156 frames
            change_indices = DecisionEncoder().find_contiguous_regions(label_column)
            # 调用DecisionEncoder().find_contiguous_regions(label_column)方法，找出当前列中值连续相同的区域，并返回这些区域的索引信息
            # append [label, onset, offset] in the result list
            for row in change_indices:
                result_labels.append( #
                    [#转换并存储结果：对于每个找到的连续区域，将该区域对应的标签名称（来自self.labels）、起始帧转换为时间（使用self._frame_to_time方法）以及结束帧转换为时间，封装成一个列表形式，然后追加到result_labels中。
                        self.labels[i],
                        self._frame_to_time(row[0]),
                        self._frame_to_time(row[1]),
                    ]
                )
        return result_labels#返回解码结果：最终返回result_labels列表，其中每个元素是一个包含标签名称、起始时间和结束时间的子列表。

    def state_dict(self):
        return {
            "labels": self.labels,
            "audio_len": self.audio_len,
            "frame_len": self.frame_len,
            "frame_hop": self.frame_hop,
            "net_pooling": self.net_pooling,
            "fs": self.fs,
        }

    @classmethod
    def load_state_dict(cls, state_dict):
        labels = state_dict["labels"]
        audio_len = state_dict["audio_len"]
        frame_len = state_dict["frame_len"]
        frame_hop = state_dict["frame_hop"]
        net_pooling = state_dict["net_pooling"]
        fs = state_dict["fs"]
        return cls(labels, audio_len, frame_len, frame_hop, net_pooling, fs)


class CatManyHotEncoder(ManyHotEncoder):
    """
    Concatenate many ManyHotEncoders.
    """

    def __init__(self, encoders, allow_same_classes=True):
        total_labels = []
        assert len(encoders) > 0, "encoders list must not be empty."
        for enc in encoders:
            for attr in ["audio_len", "frame_len", "frame_hop", "net_pooling", "fs"]:
                assert getattr(encoders[0], attr) == getattr(enc, attr), (
                    "Encoders must have the same args (e.g. same fs and so on) "
                    "except for the classes."
                )
            total_labels.extend(enc.labels)

        if len(total_labels) != len(set(total_labels)):
            if not allow_same_classes:
                # we might test for this
                raise RuntimeError(
                    f"Encoders must not have classes in common. "
                    f"But you have {total_labels} while the unique labels are: {set(total_labels)}"
                )
            total_labels_tmp_set = {}
            i = 0
            for label in total_labels:
                if label in total_labels_tmp_set:
                    total_labels.pop(i)
                else:
                    i += 1

        total_labels = OrderedDict({x: indx for indx, x in enumerate(total_labels)})

        # instantiate only one manyhotencoder
        super().__init__(
            total_labels,
            encoders[0].audio_len,
            encoders[0].frame_len,
            encoders[0].frame_hop,
            encoders[0].net_pooling,
            encoders[0].fs,
        )

class RespiraHotEncoder:
    """ "
        Adapted after DecisionEncoder.find_contiguous_regions method in
        https://github.com/DCASE-REPO/dcase_util/blob/master/dcase_util/data/decisions.py

        Encode labels into numpy arrays where 1 correspond to presence of the class and 0 absence.
        Multiple 1 can appear on the same line, it is for multi label problem.
    Args:
        labels: list, the classes which will be encoded
        n_frames: int, (Default value = None) only useful for strong labels. The number of frames of a segment.
    Attributes:
        labels: list, the classes which will be encoded
        n_frames: int, only useful for strong labels. The number of frames of a segment.
    """

    def __init__(
        self, labels,bin_labels, audio_len, frame_len, frame_hop, frames_per_node=5, fs=8000
    ):
        if type(labels) in [np.ndarray, np.array]:
            labels = labels.tolist()
        elif isinstance(labels, (dict, OrderedDict)):
            labels = list(labels.keys())
        self.labels = labels
        self.binary_labels = bin_labels

        self.audio_len = audio_len
        self.frame_len = frame_len

        self.frame_hop = frame_hop
        self.fs = fs
        self.frames_per_node = frames_per_node
        n_frames = self.audio_len * self.fs
        self.max_frames = int(n_frames / self.frame_hop)

    def encode_weak(self, labels):
        """Encode a list of weak labels into a numpy array

        Args:
            labels: list, list of labels to encode (to a vector of 0 and 1)

        Returns:
            numpy.array
            A vector containing 1 for each label, and 0 everywhere else
        """
        # useful for tensor empty labels
        if type(labels) is str:
            if labels == "empty":
                y = np.zeros(len(self.labels)) - 1
                return y
            else:
                labels = labels.split(",")
        if type(labels) is pd.DataFrame:
            if labels.empty:
                labels = []
            elif "event_label" in labels.columns:
                labels = labels["event_label"]
        y = np.zeros(len(self.labels))
        for label in labels:
            if not pd.isna(label):
                i = self.labels.index(label)
                y[i] = 1
        return y



    def _time_to_frame(self, time):
        samples = time * self.fs
        frame = samples / self.frame_hop
        return int(np.clip(frame, a_min=0, a_max=self.max_frames))


    def _frame_to_time(self, frame):
        frame = frame * self.frames_per_node / (self.fs / self.frame_hop)
        return np.clip(frame, a_min=0, a_max=self.audio_len)

    def _node_to_time(self, node, cur_audio_dur ):
        start_frame =   node * self.frames_per_node
        start_time = start_frame * self.frame_hop / self.fs

        return np.clip(start_time, a_min=0, a_max= cur_audio_dur)




    def node_to_time_interval(self, node_labels, node_size_in_frames=5, cur_audio_dur=None):
        """
        Converts node-level labels back to time intervals for a single sample.

        Args:
        - node_labels: Labels for each node in the sample.
        - node_size_in_frames: Number of frames represented by each node.

        Returns:
        - time_intervals: A one-dimensional list containing the start time for each frame in the node sequence.
        """
        num_nodes = node_labels.size(0)  # Corrected to handle tensor input
        time_intervals = []

        for node_idx in range(num_nodes + 1):
            # Calculate the start frame for the node (end frame will be implied by the next start frame)
            start_frame = node_idx * node_size_in_frames

            # Convert frame index to time (in seconds)
            start_time = start_frame * self.frame_hop / self.fs

            # Append the time interval to the list
            time_intervals.append(start_time)

        time_intervals = np.array(time_intervals)
        #  这里的裁剪应该对当前的音频进行，即每个音频都使用自身的最大长度。
        limit = np.clip(time_intervals, a_min=0, a_max= cur_audio_dur)
        return limit

    def node_to_time_interval_v2(self, node_labels, node_size_in_frames=5, cur_audio_dur=None):
        """
        Converts node-level labels back to time intervals for a single sample.
        Args:
        - node_labels: Labels for each node in the sample.
        - node_size_in_frames: Number of frames represented by each node.
        Returns:
        - time_intervals: A two-dimensional list containing the start and end times for each node.
        """
        num_nodes = node_labels.size(0)
        time_intervals = []

        for node_idx in range(num_nodes):
            start_frame = node_idx * node_size_in_frames
            end_frame = (node_idx + 1) * node_size_in_frames

            start_time = start_frame * self.frame_hop / self.fs
            end_time = end_frame * self.frame_hop / self.fs

            if end_time > cur_audio_dur:
                end_time = cur_audio_dur

            # Append the [start_time, end_time] interval
            time_intervals.append([start_time, end_time])

        return np.array(time_intervals)

    # def node_to_time_interval_v2(self, node_labels, node_size_in_frames=5, cur_audio_dur=None, n_frames=None):
    #     """
    #     Converts node-level labels back to time intervals for a single sample.
    #
    #     Args:
    #     - node_labels: Labels for each node in the sample.
    #     - node_size_in_frames: Number of frames represented by each node.
    #     - cur_audio_dur: Duration of the current audio in seconds.
    #     - n_frames: Total number of frames in the spectrogram.
    #
    #     Returns:
    #     - time_intervals: A list of tuples containing the start and end time for each node.
    #     """
    #     if n_frames is None:
    #          raise ValueError("n_frames cannot be None. Please provide a valid frame count.")
    #
    #
    #     num_nodes = node_labels.size(0)
    #     time_intervals = []
    #
    #     for node_idx in range(num_nodes):
    #         if node_idx < num_nodes - 1:
    #             # Calculate the start and end frame for the node
    #             start_frame = node_idx * node_size_in_frames
    #             end_frame = start_frame + node_size_in_frames
    #         else:
    #             # For the last node, adjust to cover the last chunk_size frames
    #             start_frame = n_frames - node_size_in_frames
    #             end_frame = n_frames
    #
    #         # Convert frame indices to time (in seconds)
    #         start_time = start_frame * self.frame_hop / self.fs
    #         end_time = end_frame * self.frame_hop / self.fs
    #
    #         # Append the time interval to the list
    #         time_intervals.append((start_time, end_time))
    #
    #     # Clip the time intervals to the audio duration
    #     if cur_audio_dur is not None:
    #         time_intervals = [
    #             (max(0, min(start_time, cur_audio_dur)), max(0, min(end_time, cur_audio_dur)))
    #             for start_time, end_time in time_intervals
    #         ]
    #
    #     return time_intervals



    def encode_strong_df(self, label_df):
        """Encode a list (or pandas Dataframe or Serie) of strong labels, they correspond to a given filename

        Args:
            label_df: pandas DataFrame or Series, contains filename, onset (in frames) and offset (in frames)
                If only filename (no onset offset) is specified, it will return the event on all the frames
                onset and offset should be in frames
        Returns:
            numpy.array
            Encoded labels, 1 where the label is present, 0 otherwise
        """

        assert any(
            [x is not None for x in [self.audio_len, self.frame_len, self.frame_hop]]
        )


        samples_len = self.max_frames
        if type(label_df) is str:
            if label_df == "empty":
                y = np.zeros((samples_len, len(self.labels))) - 1
                return y
        y = np.zeros((samples_len, len(self.labels)))
        if type(label_df) is pd.DataFrame:
            if {"onset", "offset", "event_label"}.issubset(label_df.columns):
                for _, row in label_df.iterrows():
                    if not pd.isna(row["event_label"]):
                        i = self.labels.index(row["event_label"])
                        onset = self._time_to_frame(row["onset"])
                        offset = self._time_to_frame(row["offset"])
                        if "confidence" in label_df.columns:
                            y[onset:offset, i] = row["confidence"]
                        else:
                            y[onset:offset, i] = 1

        elif type(label_df) in [
            pd.Series,
            list,
            np.ndarray,
        ]:  # list of list or list of strings
            if type(label_df) is pd.Series:
                if {"onset", "offset", "event_label"}.issubset(
                    label_df.index
                ):  # means only one value
                    if not pd.isna(label_df["event_label"]):
                        i = self.labels.index(label_df["event_label"])
                        onset = int(self._time_to_frame(label_df["onset"]))
                        offset = int(np.ceil(self._time_to_frame(label_df["offset"])))

                        if "confidence" in label_df.columns:
                            y[onset:offset, i] = label_df["confidence"]
                        else:
                            y[onset:offset, i] = 1
                    return y

            for event_label in label_df:
                # List of string, so weak labels to be encoded in strong
                if type(event_label) is str:
                    if event_label != "":
                        i = self.labels.index(event_label)
                        y[:, i] = 1

                # List of list, with [label, onset, offset]
                elif len(event_label) == 3:
                    if event_label[0] != "":
                        i = self.labels.index(event_label[0])
                        onset = int(self._time_to_frame(event_label[1]))
                        offset = int(np.ceil(self._time_to_frame(event_label[2])))
                        y[onset:offset, i] = 1
                # List of list, with [label, onset, offset, confidence]
                elif len(event_label) == 4:
                    if event_label[0] != "":
                        i = self.labels.index(event_label[0])
                        onset = int(self._time_to_frame(event_label[1]))
                        offset = int(np.ceil(self._time_to_frame(event_label[2])))
                        y[onset:offset, i] = event_label[3]

                else:
                    raise NotImplementedError(
                        "cannot encode strong, type mismatch: {}".format(
                            type(event_label)
                        )
                    )

        else:
            raise NotImplementedError(
                "To encode_strong, type is pandas.Dataframe with onset, offset and event_label"
                "columns, or it is a list or pandas Series of event labels, "
                "type given: {}".format(type(label_df))
            )
        return y

    def encode_variable_df(self, label_df, cur_audio_frames ):
        """Encode a list (or pandas Dataframe or Serie) of strong labels, they correspond to a given filename

        Args:
            label_df: pandas DataFrame or Series, contains filename, onset (in frames) and offset (in frames)
                If only filename (no onset offset) is specified, it will return the event on all the frames
                onset and offset should be in frames
        Returns:
            numpy.array
            Encoded labels, 1 where the label is present, 0 otherwise
        """

        assert any(
            [x is not None for x in [self.audio_len, self.frame_len, self.frame_hop]]
        )


        sample_frames = cur_audio_frames
        if type(label_df) is str:
            if label_df == "empty":
                y = np.zeros((sample_frames, len(self.labels))) - 1
                return y
        y = np.zeros((sample_frames, len(self.labels)))
        if type(label_df) is pd.DataFrame:
            if {"onset", "offset", "event_label"}.issubset(label_df.columns):
                for _, row in label_df.iterrows():
                    if not pd.isna(row["event_label"]):
                        i = self.labels.index(row["event_label"])
                        onset = self._time_to_frame(row["onset"])
                        offset = self._time_to_frame(row["offset"])
                        if "confidence" in label_df.columns:
                            y[onset:offset, i] = row["confidence"]
                        else:
                            y[onset:offset, i] = 1

        elif type(label_df) in [
            pd.Series,
            list,
            np.ndarray,
        ]:  # list of list or list of strings
            if type(label_df) is pd.Series:
                if {"onset", "offset", "event_label"}.issubset(
                    label_df.index
                ):  # means only one value
                    if not pd.isna(label_df["event_label"]):
                        i = self.labels.index(label_df["event_label"])
                        onset = int(self._time_to_frame(label_df["onset"]))
                        offset = int(np.ceil(self._time_to_frame(label_df["offset"])))

                        if "confidence" in label_df.columns:
                            y[onset:offset, i] = label_df["confidence"]
                        else:
                            y[onset:offset, i] = 1
                    return y

            for event_label in label_df:
                # List of string, so weak labels to be encoded in strong
                if type(event_label) is str:
                    if event_label != "":
                        i = self.labels.index(event_label)
                        y[:, i] = 1

                # List of list, with [label, onset, offset]
                elif len(event_label) == 3:
                    if event_label[0] != "":
                        i = self.labels.index(event_label[0])
                        onset = int(self._time_to_frame(event_label[1]))
                        offset = int(np.ceil(self._time_to_frame(event_label[2])))
                        y[onset:offset, i] = 1
                # List of list, with [label, onset, offset, confidence]
                elif len(event_label) == 4:
                    if event_label[0] != "":
                        i = self.labels.index(event_label[0])
                        onset = int(self._time_to_frame(event_label[1]))
                        offset = int(np.ceil(self._time_to_frame(event_label[2])))
                        y[onset:offset, i] = event_label[3]

                else:
                    raise NotImplementedError(
                        "cannot encode strong, type mismatch: {}".format(
                            type(event_label)
                        )
                    )

        else:
            raise NotImplementedError(
                "To encode_strong, type is pandas.Dataframe with onset, offset and event_label"
                "columns, or it is a list or pandas Series of event labels, "
                "type given: {}".format(type(label_df))
            )
        return y



    def decode_weak(self, labels):
        """Decode the encoded weak labels
        Args:
            labels: numpy.array, the encoded labels to be decoded

        Returns:
            list
            Decoded labels, list of string

        """
        result_labels = []
        for i, value in enumerate(labels):
            if value == 1:
                result_labels.append(self.labels[i])
        return result_labels

    def decode_strong(self, labels, cur_dur):
        """Decode the encoded strong labels
        Args:
            labels: numpy.array, the encoded labels to be decoded
        Returns:
            list
            Decoded labels, list of list: [[label, onset offset], ...]

        """
        result_labels = []
        for i, label_column in enumerate(labels.T):
            change_indices = DecisionEncoder().find_contiguous_regions(label_column)

            # append [label, onset, offset] in the result list
            for row in change_indices:
                result_labels.append(
                    [
                        self.labels[i],
                        self._node_to_time(row[0], cur_dur),
                        self._node_to_time(row[1], cur_dur),
                    ]
                )
        return result_labels

    def state_dict(self):
        return {
            "labels": self.labels,
            "audio_len": self.audio_len,
            "frame_len": self.frame_len,
            "frame_hop": self.frame_hop,
            "frames_per_node": self.frames_per_node,
            "fs": self.fs,
        }

    @classmethod
    def load_state_dict(cls, state_dict):
        labels = state_dict["labels"]
        audio_len = state_dict["audio_len"]
        frame_len = state_dict["frame_len"]
        frame_hop = state_dict["frame_hop"]
        frames_per_node = state_dict["frames_per_node"]
        fs = state_dict["fs"]
        return cls(labels, audio_len, frame_len, frame_hop, frames_per_node, fs)
