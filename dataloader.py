import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd
import numpy as np


class IEMOCAPDataset_BERT4(Dataset):

    def __init__(self, path, train=True):

        (
            self.videoIDs,
            self.videoSpeakers,
            self.videoLabels,
            self.videoText,
            self.videoAudio,
            self.videoVisual,
            self.videoSentence,
            self.trainVid,
            self.testVid,
        ) = pickle.load(open(path, "rb"), encoding="latin1")
        self.keys = self.trainVid if train else self.testVid

        self.len = len(self.keys)

        self.labels_emotion = self.videoLabels

        labels_sentiment = {}
        for item in self.videoLabels:
            array = []
            for e in self.videoLabels[item]:
                if e in [1, 3]:
                    array.append(0)
                elif e == 2:
                    array.append(1)
                elif e in [0]:
                    array.append(2)
            labels_sentiment[item] = array
        self.labels_sentiment = labels_sentiment

    def __getitem__(self, index):
        vid = self.keys[index]
        return (
            torch.FloatTensor(np.array(self.videoText[vid])),
            torch.FloatTensor(np.array(self.videoText[vid])),
            torch.FloatTensor(np.array(self.videoText[vid])),
            torch.FloatTensor(np.array(self.videoText[vid])),
            torch.FloatTensor(np.array(self.videoVisual[vid])),
            torch.FloatTensor(np.array(self.videoAudio[vid])),
            torch.FloatTensor(
                [
                    [1, 0] if x == "M" else [0, 1]
                    for x in np.array(self.videoSpeakers[vid])
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


class IEMOCAPDataset_BERT(Dataset):

    def __init__(self, path, train=True, split=None):

        data = pickle.load(open(path, "rb"), encoding="latin1")

        self.devVid = []
        self.has_dev_split = False
        if isinstance(data, dict):
            self.videoIDs = data["videoIDs"]
            self.videoSpeakers = data["videoSpeakers"]
            self.videoLabels = data["videoLabels"]
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
        else:
            (
                self.videoIDs,
                self.videoSpeakers,
                self.videoLabels,
                self.videoText0,
                self.videoText1,
                self.videoText2,
                self.videoText3,
                self.videoAudio,
                self.videoVisual,
                self.videoSentence,
                self.trainVid,
                self.testVid,
            ) = data

        if split is None:
            split = "train" if train else "test"
        if split == "train":
            self.keys = self.trainVid
        elif split in ["dev", "valid"]:
            if not self.has_dev_split:
                raise ValueError("Requested dev split but dataset file does not contain devVid")
            self.keys = self.devVid
        elif split == "test":
            self.keys = self.testVid
        else:
            raise ValueError(f"Unsupported split: {split}")

        self.len = len(self.keys)

        self.labels_emotion = self.videoLabels

        labels_sentiment = {}
        for item in self.videoLabels:
            array = []
            for e in self.videoLabels[item]:
                if e in [1, 3, 5]:
                    array.append(0)
                elif e == 2:
                    array.append(1)
                elif e in [0, 4]:
                    array.append(2)
            labels_sentiment[item] = array
        self.labels_sentiment = labels_sentiment

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


class MELDDataset_BERT(Dataset):

    def __init__(self, path, train=True, split=None):
        """
        label index mapping = {0:neutral, 1:surprise, 2:fear, 3:sadness, 4:joy, 5:disgust, 6:anger}
        """
        data = pickle.load(open(path, "rb"), encoding="latin1")

        self.devVid = []
        self.has_dev_split = False
        if isinstance(data, dict):
            self.videoIDs = data["videoIDs"]
            self.videoSpeakers = data["videoSpeakers"]
            self.videoLabels = data["videoLabels"]
            self.videoSentiments = data["videoSentiments"]
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
        else:
            (
                self.videoIDs,
                self.videoSpeakers,
                self.videoLabels,
                self.videoSentiments,
                self.videoText0,
                self.videoText1,
                self.videoText2,
                self.videoText3,
                self.videoAudio,
                self.videoVisual,
                self.videoSentence,
                self.trainVid,
                self.testVid,
                _,
            ) = data

        if split is None:
            split = "train" if train else "test"
        if split == "train":
            self.keys = [x for x in self.trainVid]
        elif split in ["dev", "valid"]:
            if not self.has_dev_split:
                raise ValueError("Requested dev split but dataset file does not contain devVid")
            self.keys = [x for x in self.devVid]
        elif split == "test":
            self.keys = [x for x in self.testVid]
        else:
            raise ValueError(f"Unsupported split: {split}")

        self.len = len(self.keys)

        self.labels_emotion = self.videoLabels

        self.labels_sentiment = self.videoSentiments

    def __getitem__(self, index):
        vid = self.keys[index]
        return (
            torch.FloatTensor(np.array(self.videoText0[vid])),
            torch.FloatTensor(np.array(self.videoText1[vid])),
            torch.FloatTensor(np.array(self.videoText2[vid])),
            torch.FloatTensor(np.array(self.videoText3[vid])),
            torch.FloatTensor(np.array(self.videoVisual[vid])),
            torch.FloatTensor(np.array(self.videoAudio[vid])),
            torch.FloatTensor(np.array(self.videoSpeakers[vid])),
            torch.FloatTensor([1] * len(np.array(self.labels_emotion[vid]))),
            torch.LongTensor(np.array(self.labels_emotion[vid])),
            torch.LongTensor(np.array(self.labels_sentiment[vid])),
            vid,
        )

    def __len__(self):
        return self.len

    def return_labels(self):
        return_label = []
        for key in self.keys:
            return_label += self.videoLabels[key]
        return return_label

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


class CMUMOSEIDataset7(Dataset):

    def __init__(self, path, train=True):

        (
            self.videoIDs,
            self.videoSpeakers,
            self.videoLabels,
            self.videoText,
            self.videoAudio,
            self.videoVisual,
            self.videoSentence,
            self.trainVid,
            self.testVid,
        ) = pickle.load(open(path, "rb"), encoding="latin1")

        self.keys = self.trainVid if train else self.testVid

        self.len = len(self.keys)

        labels_emotion = {}
        for item in self.videoLabels:
            labels_emotion[item] = [
                int(np.round(np.clip(float(a), a_min=-3.0, a_max=3.0))) + 3
                for a in self.videoLabels[item]
            ]
        self.labels_emotion = labels_emotion

        labels_sentiment = {}
        for item in self.videoLabels:
            array = []
            for a in self.videoLabels[item]:
                if a < 0:
                    array.append(0)
                elif 0 <= a and a <= 0:
                    array.append(1)
                elif a > 0:
                    array.append(2)
            labels_sentiment[item] = array
        self.labels_sentiment = labels_sentiment

    def __getitem__(self, index):
        vid = self.keys[index]
        return (
            torch.FloatTensor(np.array(self.videoText[vid])),
            torch.FloatTensor(np.array(self.videoText[vid])),
            torch.FloatTensor(np.array(self.videoText[vid])),
            torch.FloatTensor(np.array(self.videoText[vid])),
            torch.FloatTensor(np.array(self.videoVisual[vid])),
            torch.FloatTensor(np.array(self.videoAudio[vid])),
            torch.FloatTensor(
                [
                    [1, 0] if x == "M" else [0, 1]
                    for x in np.array(self.videoSpeakers[vid])
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
