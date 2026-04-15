import pickle

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from chsims_v1_raw_dataset import _masked_mean_text, _masked_mean_sequence


class MOSIDataset(Dataset):

    def __init__(self,
                 path,
                 train=True,
                 split=None,
                 num_emotion_classes=5,
                 binary_mode='has0'):
        data = pickle.load(open(path, "rb"), encoding="latin1")

        self.videoIDs = data["videoIDs"]
        self.videoSpeakers = data["videoSpeakers"]
        self.videoRegressionLabels = data["videoRegressionLabels"]
        self.videoText0 = data["videoText0"]
        self.videoText1 = data["videoText1"]
        self.videoText2 = data["videoText2"]
        self.videoText3 = data["videoText3"]
        self.videoAudio = data["videoAudio"]
        self.videoVisual = data["videoVisual"]
        self.videoSentence = data["videoSentence"]
        self.trainVid = data["trainVid"]
        self.devVid = data.get("devVid", [])
        self.testVid = data["testVid"]
        self.has_dev_split = len(self.devVid) > 0
        self.num_emotion_classes = int(num_emotion_classes)
        self.binary_mode = str(binary_mode)
        self.videoEmotionLabels2 = data.get("videoEmotionLabels2")
        self.videoEmotionLabels5 = data.get("videoEmotionLabels5")
        self.videoEmotionLabels7 = data.get("videoEmotionLabels7")

        if split is None:
            split = "train" if train else "test"
        if split == "train":
            self.keys = self.trainVid
        elif split in ["dev", "valid"]:
            self.keys = self.devVid
        elif split == "test":
            self.keys = self.testVid
        else:
            raise ValueError(f"Unsupported split: {split}")

        self.keep_indices = {
            vid: self._build_keep_indices(vid)
            for vid in self.videoRegressionLabels
        }
        if self.num_emotion_classes == 2 and self.binary_mode == 'non0':
            self.keys = [
                vid for vid in self.keys
                if len(self.keep_indices.get(vid, [])) > 0
            ]
        self.len = len(self.keys)
        self.labels_emotion = {
            vid: self._get_emotion_labels(vid)
            for vid in self.keys
        }
        self.labels_sentiment = {
            vid: [self._map_sentiment_label(x) for x in self._get_regression_values(vid)]
            for vid in self.keys
        }

    def _build_keep_indices(self, vid):
        if self.num_emotion_classes == 2 and self.binary_mode == 'non0':
            return [
                idx for idx, value in enumerate(self.videoRegressionLabels[vid])
                if float(value) != 0
            ]
        return list(range(len(self.videoRegressionLabels[vid])))

    def _get_regression_values(self, vid):
        return [
            float(self.videoRegressionLabels[vid][idx])
            for idx in self.keep_indices[vid]
        ]

    def _slice_sequence(self, values, vid):
        indices = self.keep_indices[vid]
        if isinstance(values, np.ndarray):
            return values[indices]
        return [values[idx] for idx in indices]

    def _get_emotion_labels(self, vid):
        if self.num_emotion_classes == 2 and self.videoEmotionLabels2 is not None and self.binary_mode == 'has0':
            return [int(self.videoEmotionLabels2[vid][idx]) for idx in self.keep_indices[vid]]
        if self.num_emotion_classes == 5 and self.videoEmotionLabels5 is not None:
            return [int(self.videoEmotionLabels5[vid][idx]) for idx in self.keep_indices[vid]]
        if self.num_emotion_classes == 7 and self.videoEmotionLabels7 is not None:
            return [int(self.videoEmotionLabels7[vid][idx]) for idx in self.keep_indices[vid]]
        return [self._map_emotion_label(x) for x in self._get_regression_values(vid)]

    def _map_emotion_label(self, value):
        value = float(value)
        value_clip_7 = np.clip(value, a_min=-3.0, a_max=3.0)
        value_clip_5 = np.clip(value, a_min=-2.0, a_max=2.0)
        if self.num_emotion_classes == 2:
            if self.binary_mode == 'non0':
                return 0 if value_clip_7 < 0 else 1
            return 0 if value_clip_7 < 0 else 1
        if self.num_emotion_classes == 5:
            return int(np.round(value_clip_5)) + 2
        if self.num_emotion_classes == 7:
            return int(np.round(value_clip_7)) + 3
        raise ValueError(
            f"Unsupported MOSI class setting: {self.num_emotion_classes}")

    def _map_sentiment_label(self, value):
        value = float(value)
        if value < 0:
            return 0
        if value == 0:
            return 1
        return 2

    def __getitem__(self, index):
        vid = self.keys[index]
        text0 = np.array(self._slice_sequence(self.videoText0[vid], vid))
        text1 = np.array(self._slice_sequence(self.videoText1[vid], vid))
        text2 = np.array(self._slice_sequence(self.videoText2[vid], vid))
        text3 = np.array(self._slice_sequence(self.videoText3[vid], vid))
        visual = np.array(self._slice_sequence(self.videoVisual[vid], vid))
        audio = np.array(self._slice_sequence(self.videoAudio[vid], vid))
        speakers = np.array(self._slice_sequence(self.videoSpeakers[vid], vid))
        return (
            torch.FloatTensor(text0),
            torch.FloatTensor(text1),
            torch.FloatTensor(text2),
            torch.FloatTensor(text3),
            torch.FloatTensor(visual),
            torch.FloatTensor(audio),
            torch.FloatTensor(
                [
                    [1, 0] if x == "M" else [0, 1]
                    for x in speakers
                ]
            ),
            torch.FloatTensor([1] * len(np.array(self.labels_emotion[vid]))),
            torch.LongTensor(np.array(self.labels_emotion[vid])),
            torch.LongTensor(np.array(self.labels_sentiment[vid])),
            vid,
        )

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)

        return [
            (
                pad_sequence(dat[i])
                if i < 7
                else pad_sequence(dat[i]) if i < 10 else dat[i].tolist()
            )
            for i in dat
        ]


class MOSEIRegressionDataset(Dataset):

    def __init__(self, path, train=True, split=None):
        data = pickle.load(open(path, "rb"), encoding="latin1")
        self.has_dev_split = True

        self.videoIDs = {}
        self.videoSpeakers = {}
        self.videoRegressionLabels = {}
        self.videoText0 = {}
        self.videoText1 = {}
        self.videoText2 = {}
        self.videoText3 = {}
        self.videoAudio = {}
        self.videoVisual = {}
        self.videoSentence = {}
        self.videoAnnotations = {}
        self.trainVid = []
        self.devVid = []
        self.testVid = []

        split_mapping = [("train", "train"), ("valid", "dev"), ("test", "test")]
        for raw_split_name, target_split in split_mapping:
            split_data = data[raw_split_name]
            target_keys = getattr(self, f"{target_split}Vid")
            grouped = {}
            for idx, sample_id in enumerate(split_data["id"]):
                sample_id = str(sample_id)
                video_id, utterance_id = sample_id.split("$_$")
                dialogue_id = f"{target_split}_{video_id}"
                if dialogue_id not in grouped:
                    grouped[dialogue_id] = {
                        "video_ids": [],
                        "speakers": [],
                        "regression": [],
                        "text0": [],
                        "text1": [],
                        "text2": [],
                        "text3": [],
                        "audio": [],
                        "visual": [],
                        "sentences": [],
                        "annotations": [],
                    }
                    target_keys.append(dialogue_id)
                packed = grouped[dialogue_id]
                text_feature = _masked_mean_text(
                    split_data["text"][idx],
                    split_data["text_bert"][idx],
                )
                audio_feature = _masked_mean_sequence(
                    split_data["audio"][idx],
                    split_data["audio_lengths"][idx],
                )
                visual_feature = _masked_mean_sequence(
                    split_data["vision"][idx],
                    split_data["vision_lengths"][idx],
                )
                packed["video_ids"].append(utterance_id)
                packed["speakers"].append("M")
                packed["regression"].append(
                    float(split_data["regression_labels"][idx]))
                packed["text0"].append(text_feature)
                packed["text1"].append(text_feature)
                packed["text2"].append(text_feature)
                packed["text3"].append(text_feature)
                packed["audio"].append(audio_feature)
                packed["visual"].append(visual_feature)
                packed["sentences"].append(str(split_data["raw_text"][idx]))
                packed["annotations"].append(str(split_data["annotations"][idx]))

            for dialogue_id in target_keys:
                if dialogue_id not in grouped:
                    continue
                packed = grouped[dialogue_id]
                self.videoIDs[dialogue_id] = packed["video_ids"]
                self.videoSpeakers[dialogue_id] = packed["speakers"]
                self.videoRegressionLabels[dialogue_id] = packed["regression"]
                self.videoText0[dialogue_id] = np.stack(
                    packed["text0"]).astype(np.float32)
                self.videoText1[dialogue_id] = np.stack(
                    packed["text1"]).astype(np.float32)
                self.videoText2[dialogue_id] = np.stack(
                    packed["text2"]).astype(np.float32)
                self.videoText3[dialogue_id] = np.stack(
                    packed["text3"]).astype(np.float32)
                self.videoAudio[dialogue_id] = np.stack(
                    packed["audio"]).astype(np.float32)
                self.videoVisual[dialogue_id] = np.stack(
                    packed["visual"]).astype(np.float32)
                self.videoSentence[dialogue_id] = packed["sentences"]
                self.videoAnnotations[dialogue_id] = packed["annotations"]

        if split is None:
            split = "train" if train else "test"
        if split == "train":
            self.keys = self.trainVid
        elif split in ["dev", "valid"]:
            self.keys = self.devVid
        elif split == "test":
            self.keys = self.testVid
        else:
            raise ValueError(f"Unsupported split: {split}")

        self.len = len(self.keys)
        self.labels_emotion = {
            vid: [float(x) for x in self.videoRegressionLabels[vid]]
            for vid in self.videoRegressionLabels
        }
        self.labels_sentiment = {
            vid: [self._map_sentiment_label(x) for x in self.videoRegressionLabels[vid]]
            for vid in self.videoRegressionLabels
        }

    def _map_sentiment_label(self, value):
        value = float(value)
        if value < 0:
            return 0
        if value == 0:
            return 1
        return 2

    def __getitem__(self, index):
        vid = self.keys[index]
        return (
            torch.FloatTensor(np.array(self.videoText0[vid])),
            torch.FloatTensor(np.array(self.videoText1[vid])),
            torch.FloatTensor(np.array(self.videoText2[vid])),
            torch.FloatTensor(np.array(self.videoText3[vid])),
            torch.FloatTensor(np.array(self.videoVisual[vid])),
            torch.FloatTensor(np.array(self.videoAudio[vid])),
            torch.FloatTensor(
                [
                    [1, 0] if x == "M" else [0, 1]
                    for x in np.array(self.videoSpeakers[vid])
                ]
            ),
            torch.FloatTensor([1] * len(np.array(self.labels_emotion[vid]))),
            torch.FloatTensor(np.array(self.labels_emotion[vid])),
            torch.LongTensor(np.array(self.labels_sentiment[vid])),
            vid,
        )

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)

        return [
            (
                pad_sequence(dat[i])
                if i < 7
                else pad_sequence(dat[i]) if i < 10 else dat[i].tolist()
            )
            for i in dat
        ]
