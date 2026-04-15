import logging
import os
import sys
import atexit
import numpy as np
import pickle as pk
import pickle
import datetime
from copy import deepcopy
import torch.nn as nn
import torch.optim as optim
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
import time
from utils import AutomaticWeightedLoss
from model import GraphSmile
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
from trainer import train_or_eval_model, seed_everything
from dataloader import (
    IEMOCAPDataset_BERT,
    IEMOCAPDataset_BERT4,
    MELDDataset_BERT,
    CMUMOSEIDataset7,
)
from functools import partial
import argparse
import matplotlib
import warnings
warnings.filterwarnings('ignore', category=UserWarning, 
                       module='torch.distributed.distributed_c10d')

matplotlib.use('Agg')
import matplotlib.pyplot as plt

CHSIMS_DATASET_IMPORT_ERROR = None
CHSIMS_RAW_IMPORT_ERROR = None
MOSI_DATASET_IMPORT_ERROR = None

try:
    from chsims_dataset import CHSIMSDataset
except ModuleNotFoundError as exc:
    CHSIMSDataset = None
    CHSIMS_DATASET_IMPORT_ERROR = exc

try:
    from chsims_v1_raw_dataset import (
        CHSIMSV1RawDataset,
        CHSIMSV2NoSuperDataset,
        CHSIMSV2RegressionDataset,
    )
except ModuleNotFoundError as exc:
    CHSIMSV1RawDataset = None
    CHSIMSV2NoSuperDataset = None
    CHSIMSV2RegressionDataset = None
    CHSIMS_RAW_IMPORT_ERROR = exc

try:
    from mosi_dataset import MOSIDataset, MOSEIRegressionDataset
except ModuleNotFoundError as exc:
    MOSIDataset = None
    MOSEIRegressionDataset = None
    MOSI_DATASET_IMPORT_ERROR = exc


def _flatten_text_bert_feature_nosuper(text_bert_feature):
    text_bert_feature = np.asarray(text_bert_feature, dtype=np.float32)
    if text_bert_feature.ndim == 0:
        return np.zeros((1, ), dtype=np.float32)
    flat_feature = text_bert_feature.reshape(-1)
    scale = max(float(np.max(np.abs(flat_feature))), 1.0)
    return (flat_feature / scale).astype(np.float32)


def _masked_mean_sequence_nosuper(sequence_feature, valid_length):
    sequence_feature = np.asarray(sequence_feature, dtype=np.float32)
    valid_length = int(valid_length)
    valid_length = max(1, min(valid_length, sequence_feature.shape[0]))
    return sequence_feature[:valid_length].mean(axis=0).astype(np.float32)


def _split_dialogue_ids_nosuper(dialogue_ids):
    sorted_ids = sorted(dialogue_ids)
    total = len(sorted_ids)
    if total <= 2:
        train_ids = sorted_ids[:1]
        dev_ids = sorted_ids[1:2]
        test_ids = sorted_ids[2:]
        return train_ids, dev_ids, test_ids
    dev_count = max(1, int(round(total * 0.1)))
    test_count = max(1, int(round(total * 0.1)))
    if dev_count + test_count >= total:
        overflow = dev_count + test_count - total + 1
        if test_count >= dev_count:
            test_count = max(1, test_count - overflow)
        else:
            dev_count = max(1, dev_count - overflow)
    train_count = total - dev_count - test_count
    if train_count <= 0:
        train_count = max(1, total - 2)
        dev_count = 1 if total - train_count >= 1 else 0
        test_count = total - train_count - dev_count
    train_ids = sorted_ids[:train_count]
    dev_ids = sorted_ids[train_count:train_count + dev_count]
    test_ids = sorted_ids[train_count + dev_count:]
    return train_ids, dev_ids, test_ids


if CHSIMSV2NoSuperDataset is None:

    class CHSIMSV2NoSuperDataset(Dataset):

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

            grouped = {}
            seen_sample_ids = set()
            for idx, sample_id in enumerate(data["id"]):
                sample_id = str(sample_id)
                if sample_id in seen_sample_ids or "$_$" not in sample_id:
                    continue
                seen_sample_ids.add(sample_id)
                video_id, utterance_id = sample_id.split("$_$", 1)
                if video_id not in grouped:
                    grouped[video_id] = []
                grouped[video_id].append(
                    {
                        "utterance_id":
                        utterance_id,
                        "text_feature":
                        _flatten_text_bert_feature_nosuper(data["text_bert"][idx]),
                        "audio_feature":
                        _masked_mean_sequence_nosuper(data["audio"][idx],
                                                      data["audio_lengths"][idx]),
                        "visual_feature":
                        _masked_mean_sequence_nosuper(data["vision"][idx],
                                                      data["vision_lengths"][idx]),
                        "sample_id":
                        sample_id,
                    })

            train_video_ids, dev_video_ids, test_video_ids = _split_dialogue_ids_nosuper(
                grouped.keys())
            self.trainVid = [f"train_{vid}" for vid in train_video_ids]
            self.devVid = [f"dev_{vid}" for vid in dev_video_ids]
            self.testVid = [f"test_{vid}" for vid in test_video_ids]

            for target_split, video_ids in [
                ("train", train_video_ids),
                ("dev", dev_video_ids),
                ("test", test_video_ids),
            ]:
                target_keys = getattr(self, f"{target_split}Vid")
                for dialogue_id in target_keys:
                    raw_video_id = dialogue_id.split("_", 1)[1]
                    turns = sorted(grouped[raw_video_id],
                                   key=lambda item: item["utterance_id"])
                    text_matrix = np.stack(
                        [item["text_feature"] for item in turns]).astype(np.float32)
                    audio_matrix = np.stack(
                        [item["audio_feature"] for item in turns]).astype(np.float32)
                    visual_matrix = np.stack(
                        [item["visual_feature"] for item in turns]).astype(np.float32)
                    utterance_ids = [item["utterance_id"] for item in turns]
                    sample_ids = [item["sample_id"] for item in turns]
                    dummy_labels = [0 for _ in turns]

                    self.videoIDs[dialogue_id] = utterance_ids
                    self.videoSpeakers[dialogue_id] = ["M"] * len(turns)
                    self.videoRegressionLabels[dialogue_id] = dummy_labels
                    self.videoText0[dialogue_id] = text_matrix
                    self.videoText1[dialogue_id] = text_matrix.copy()
                    self.videoText2[dialogue_id] = text_matrix.copy()
                    self.videoText3[dialogue_id] = text_matrix.copy()
                    self.videoAudio[dialogue_id] = audio_matrix
                    self.videoVisual[dialogue_id] = visual_matrix
                    self.videoSentence[dialogue_id] = sample_ids
                    self.videoAnnotations[dialogue_id] = sample_ids

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
                vid: [0 for _ in self.videoRegressionLabels[vid]]
                for vid in self.videoRegressionLabels
            }
            self.labels_sentiment = {
                vid: [0 for _ in self.videoRegressionLabels[vid]]
                for vid in self.videoRegressionLabels
            }

        def _get_speaker_mask(self, vid):
            speakers = list(self.videoSpeakers.get(vid, []))
            target_len = len(self.labels_emotion[vid])
            if len(speakers) < target_len:
                speakers = speakers + ["M"] * (target_len - len(speakers))
            elif len(speakers) > target_len:
                speakers = speakers[:target_len]
            return torch.FloatTensor(
                [[1, 0] if speaker == "M" else [0, 1] for speaker in speakers])

        def __getitem__(self, index):
            vid = self.keys[index]
            return (
                torch.FloatTensor(np.array(self.videoText0[vid])),
                torch.FloatTensor(np.array(self.videoText1[vid])),
                torch.FloatTensor(np.array(self.videoText2[vid])),
                torch.FloatTensor(np.array(self.videoText3[vid])),
                torch.FloatTensor(np.array(self.videoVisual[vid])),
                torch.FloatTensor(np.array(self.videoAudio[vid])),
                self._get_speaker_mask(vid),
                torch.FloatTensor([1] * len(np.array(self.labels_emotion[vid]))),
                torch.LongTensor(np.array(self.labels_emotion[vid])),
                torch.LongTensor(np.array(self.labels_sentiment[vid])),
                vid,
            )

        def __len__(self):
            return self.len

        def collate_fn(self, data):
            columns = list(zip(*data))
            return [
                (
                    pad_sequence(list(columns[i]))
                    if i < 7
                    else pad_sequence(list(columns[i]))
                    if i < 10
                    else list(columns[i])
                )
                for i in range(len(columns))
            ]

parser = argparse.ArgumentParser()

parser.add_argument('--no_cuda',
                    action='store_true',
                    default=False,
                    help='does not use GPU')
parser.add_argument('--gpu', default='2', type=str, help='GPU ids')
parser.add_argument('--port', default='15301', help='MASTER_PORT')
parser.add_argument('--classify',
                    default='emotion',
                    help='sentiment, emotion, regression, unsupervised')
parser.add_argument('--lr',
                    type=float,
                    default=0.00001,
                    metavar='LR',
                    help='learning rate')
parser.add_argument('--l2',
                    type=float,
                    default=0.0001,
                    metavar='L2',
                    help='L2 regularization weight')
parser.add_argument('--batch_size',
                    type=int,
                    default=32,
                    metavar='BS',
                    help='batch size')
parser.add_argument('--epochs',
                    type=int,
                    default=100,
                    metavar='E',
                    help='number of epochs')
parser.add_argument('--tensorboard',
                    action='store_true',
                    default=False,
                    help='Enables tensorboard log')
parser.add_argument('--modals', default='avl', help='modals')
parser.add_argument(
    '--dataset',
    default='IEMOCAP',
    help='dataset to train and test.MELD/IEMOCAP/IEMOCAP4/CMUMOSEI7/CHSIMSV2REG/CHSIMSV2_NOSUPER2/CHSIMSV2_NOSUPER3/CHSIMSV2_NOSUPER5/MOSI2_HAS0/MOSI2_NON0/MOSI5/MOSI7/MOSEI2_HAS0/MOSEI2_NON0/MOSEI5/MOSEI7/MOSEIREG',
)
parser.add_argument(
    '--textf_mode',
    default='textf0',
    help='concat4/concat2/textf0/textf1/textf2/textf3/sum2/sum4',
)

parser.add_argument(
    '--conv_fpo',
    nargs='+',
    type=int,
    default=[3, 1, 1],
    help='n_filter,n_padding; n_out = (n_in + 2*n_padding -n_filter)/stride +1',
)

parser.add_argument('--hidden_dim', type=int, default=256, help='hidden_dim')
parser.add_argument(
    '--win',
    nargs='+',
    type=int,
    default=[17, 17],
    help='[win_p, win_f], -1 denotes all nodes',
)
parser.add_argument('--heter_n_layers',
                    nargs='+',
                    type=int,
                    default=[6, 6, 6],
                    help='heter_n_layers')

parser.add_argument('--drop',
                    type=float,
                    default=0.3,
                    metavar='dropout',
                    help='dropout rate')

parser.add_argument('--shift_win',
                    type=int,
                    default=12,
                    help='windows of sentiment shift')
parser.add_argument('--refine_heads',
                    type=int,
                    default=4,
                    help='heads for cross-modal refinement')
parser.add_argument('--dropedge',
                    type=float,
                    default=0.1,
                    help='drop ratio for graph edges during training')
parser.add_argument('--gcn_residual_drop',
                    type=float,
                    default=0.3,
                    help='dropout on gcn residual branch')
parser.add_argument('--contrastive_weight',
                    type=float,
                    default=0.05,
                    help='weight for supervised contrastive loss')
parser.add_argument('--decouple_weight',
                    type=float,
                    default=0.02,
                    help='weight for decoupling loss')
parser.add_argument('--warmup_epochs',
                    type=int,
                    default=8,
                    help='epochs for warmup stage')
parser.add_argument('--supcon_temperature',
                    type=float,
                    default=0.07,
                    help='temperature for supervised contrastive loss')
parser.add_argument('--label_smoothing',
                    type=float,
                    default=0.1,
                    help='label smoothing for classification loss')
parser.add_argument('--class_balance_beta',
                    type=float,
                    default=0.9999,
                    help='beta for class-balanced reweighting')
parser.add_argument('--hard_negative_weight',
                    type=float,
                    default=2.0,
                    help='weight for hard negative pairs in SupCon')
parser.add_argument('--use_pair_expert',
                    action='store_true',
                    default=False,
                    help='enable pair experts for 0/4 and 2/5')
parser.add_argument('--pair_logit_scale',
                    type=float,
                    default=0.3,
                    help='residual logit correction scale from pair experts')
parser.add_argument('--pair_aux_weight_04',
                    type=float,
                    default=0.2,
                    help='auxiliary loss weight for pair expert 0/4')
parser.add_argument('--pair_aux_weight_25',
                    type=float,
                    default=0.2,
                    help='auxiliary loss weight for pair expert 2/5')
parser.add_argument('--use_proto_prior',
                    action='store_true',
                    default=False,
                    help='enable prototype prior logit calibration')
parser.add_argument('--proto_alpha',
                    type=float,
                    default=0.25,
                    help='fusion scale for prototype logits')
parser.add_argument('--proto_temp',
                    type=float,
                    default=0.07,
                    help='temperature for prototype cosine logits')
parser.add_argument('--proto_momentum',
                    type=float,
                    default=0.95,
                    help='ema momentum for prototype update')
parser.add_argument('--proto_aux_weight',
                    type=float,
                    default=0.1,
                    help='auxiliary loss weight for prototype logits')
parser.add_argument('--use_hierarchical',
                    action='store_true',
                    default=False,
                    help='enable hierarchical coarse-to-fine calibration')
parser.add_argument('--hier_alpha',
                    type=float,
                    default=0.2,
                    help='fusion scale for hierarchical calibration logits')
parser.add_argument('--hier_temp',
                    type=float,
                    default=1.0,
                    help='temperature for hierarchical coarse probabilities')
parser.add_argument('--hier_aux_weight',
                    type=float,
                    default=0.12,
                    help='auxiliary CE weight for coarse hierarchy')
parser.add_argument('--hier_kl_weight',
                    type=float,
                    default=0.05,
                    help='KL consistency weight for hierarchy')
parser.add_argument('--use_temporal_fusion',
                    action='store_true',
                    default=False,
                    help='enable temporal branch and late fusion')
parser.add_argument('--temporal_logit_scale',
                    type=float,
                    default=0.35,
                    help='fusion scale for temporal logits')
parser.add_argument('--temporal_aux_weight',
                    type=float,
                    default=0.15,
                    help='auxiliary loss weight for temporal branch')
parser.add_argument('--grl_alpha',
                    type=float,
                    default=1.0,
                    help='gradient reversal strength for modality adversarial branch')
parser.add_argument('--modality_adv_weight',
                    type=float,
                    default=0.0,
                    help='loss weight for modality adversarial branch')
parser.add_argument('--pair04_sample_boost',
                    type=float,
                    default=1.5,
                    help='sample boost weight for pair 0/4 loss')
parser.add_argument('--pair04_margin',
                    type=float,
                    default=0.25,
                    help='contrastive margin for pair 0/4 features')
parser.add_argument('--pair04_margin_weight',
                    type=float,
                    default=0.10,
                    help='loss weight for pair 0/4 contrastive margin')
parser.add_argument('--disable_auto_iemocap4_lite',
                    action='store_true',
                    default=False,
                    help='disable the balanced preset automatically applied to IEMOCAP4')
parser.add_argument('--use_lite_small_set_arch',
                    action='store_true',
                    default=False,
                    help='use the lighter fusion architecture for small-class datasets')
parser.add_argument('--early_stop_patience',
                    type=int,
                    default=0,
                    help='stop training after N epochs without dev improvement, 0 disables early stopping')

parser.add_argument(
    '--loss_type',
    default='emo_sen_sft',
    help='auto/epoch/emo_sen_sft/emo_sen/emo_sft/emo/sen_sft/sen',
)
parser.add_argument(
    '--lambd',
    nargs='+',
    type=float,
    default=[1.0, 1.0, 1.0],
    help='[loss_emotion, loss_sentiment, loss_shift]',
)
parser.add_argument('--select_best_by',
                    type=str,
                    default='dev',
                    help='how to select best epoch')
parser.add_argument('--plot_dir', type=str, default='results/plots')
parser.add_argument('--checkpoint_dir',
                    type=str,
                    default='results/checkpoints')
parser.add_argument('--disable_tsne',
                    action='store_true',
                    default=False,
                    help='skip t-SNE drawing at the end of training')
parser.add_argument('--tsne_max_points',
                    type=int,
                    default=0,
                    help='subsample test features before t-SNE, 0 means use all points')
parser.add_argument('--data_path',
                    type=str,
                    default='',
                    help='optional dataset file path override for IEMOCAP/IEMOCAP4/CMUMOSEI2/CMUMOSEI5/CMUMOSEI7/CHSIMS2/CHSIMS3/CHSIMS5/CHSIMSV2REG/CHSIMSV2_NOSUPER2/CHSIMSV2_NOSUPER3/CHSIMSV2_NOSUPER5/MOSI2_HAS0/MOSI2_NON0/MOSI5/MOSI7/MOSEI2_HAS0/MOSEI2_NON0/MOSEI5/MOSEI7/MOSEIREG')
args = parser.parse_args()


def arg_was_provided(flag):
    return flag in sys.argv


def maybe_apply_iemocap4_lite_preset(args):
    args.auto_lite_small_sets = bool(
        getattr(args, 'use_lite_small_set_arch', False))
    args.auto_iemocap4_lite_notes = []
    if args.dataset != 'IEMOCAP4' or args.disable_auto_iemocap4_lite:
        return args

    def maybe_set(flag, attr, value):
        if arg_was_provided(flag):
            return
        current = getattr(args, attr)
        if current != value:
            setattr(args, attr, value)
            args.auto_iemocap4_lite_notes.append(f'{attr}={value}')

    maybe_set('--hidden_dim', 'hidden_dim', 256)
    maybe_set('--win', 'win', [5, 5])
    maybe_set('--heter_n_layers', 'heter_n_layers', [4, 4, 4])
    maybe_set('--drop', 'drop', 0.22)
    maybe_set('--shift_win', 'shift_win', 10)
    maybe_set('--refine_heads', 'refine_heads', 4)
    maybe_set('--contrastive_weight', 'contrastive_weight', 0.0)
    maybe_set('--decouple_weight', 'decouple_weight', 0.0)
    maybe_set('--pair_aux_weight_04', 'pair_aux_weight_04', 0.0)
    maybe_set('--pair_aux_weight_25', 'pair_aux_weight_25', 0.0)
    maybe_set('--proto_aux_weight', 'proto_aux_weight', 0.0)
    maybe_set('--hier_aux_weight', 'hier_aux_weight', 0.0)
    maybe_set('--hier_kl_weight', 'hier_kl_weight', 0.0)
    maybe_set('--temporal_aux_weight', 'temporal_aux_weight', 0.0)
    maybe_set('--label_smoothing', 'label_smoothing', 0.05)
    maybe_set('--class_balance_beta', 'class_balance_beta', 0.9992)
    maybe_set('--hard_negative_weight', 'hard_negative_weight', 1.0)
    maybe_set('--early_stop_patience', 'early_stop_patience', 18)
    return args


def maybe_apply_chsims2_binary_preset(args):
    args.auto_chsims2_binary_notes = []
    if args.dataset != 'CHSIMS2':
        return args

    def maybe_set(flag, attr, value):
        if arg_was_provided(flag):
            return
        current = getattr(args, attr)
        if current != value:
            setattr(args, attr, value)
            args.auto_chsims2_binary_notes.append(f'{attr}={value}')

    maybe_set('--pair_aux_weight_04', 'pair_aux_weight_04', 0.0)
    maybe_set('--pair_aux_weight_25', 'pair_aux_weight_25', 0.0)
    maybe_set('--pair04_margin_weight', 'pair04_margin_weight', 0.0)
    maybe_set('--warmup_epochs', 'warmup_epochs', 6)
    maybe_set('--label_smoothing', 'label_smoothing', 0.03)
    maybe_set('--class_balance_beta', 'class_balance_beta', 0.9990)
    maybe_set('--hard_negative_weight', 'hard_negative_weight', 1.0)
    maybe_set('--temporal_logit_scale', 'temporal_logit_scale', 0.28)
    maybe_set('--temporal_aux_weight', 'temporal_aux_weight', 0.05)
    return args


def maybe_apply_chsimsv2_nosuper_preset(args):
    args.auto_chsimsv2_nosuper_notes = []
    nosuper_datasets = {
        'CHSIMSV2_NOSUPER2',
        'CHSIMSV2_NOSUPER3',
        'CHSIMSV2_NOSUPER5',
    }
    if args.dataset not in nosuper_datasets:
        return args

    if args.classify != 'unsupervised':
        args.classify = 'unsupervised'
        args.auto_chsimsv2_nosuper_notes.append('classify=unsupervised')

    def maybe_set(flag, attr, value):
        if arg_was_provided(flag):
            return
        current = getattr(args, attr)
        if current != value:
            setattr(args, attr, value)
            args.auto_chsimsv2_nosuper_notes.append(f'{attr}={value}')

    maybe_set('--textf_mode', 'textf_mode', 'textf0')
    maybe_set('--decouple_weight', 'decouple_weight', 1.0)
    maybe_set('--modality_adv_weight', 'modality_adv_weight', 0.05)
    maybe_set('--contrastive_weight', 'contrastive_weight', 0.0)
    maybe_set('--pair_aux_weight_04', 'pair_aux_weight_04', 0.0)
    maybe_set('--pair_aux_weight_25', 'pair_aux_weight_25', 0.0)
    maybe_set('--pair04_margin_weight', 'pair04_margin_weight', 0.0)
    maybe_set('--proto_aux_weight', 'proto_aux_weight', 0.0)
    maybe_set('--hier_aux_weight', 'hier_aux_weight', 0.0)
    maybe_set('--hier_kl_weight', 'hier_kl_weight', 0.0)
    maybe_set('--temporal_aux_weight', 'temporal_aux_weight', 0.0)
    maybe_set('--early_stop_patience', 'early_stop_patience', 12)
    return args


args = maybe_apply_iemocap4_lite_preset(args)
args = maybe_apply_chsims2_binary_preset(args)
args = maybe_apply_chsimsv2_nosuper_preset(args)

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = args.port
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
world_size = torch.cuda.device_count()
os.environ['WORLD_SIZE'] = str(world_size)

MELD_path = 'data/meld_multi_features_with_dev.pkl'
IEMOCAP_path = 'data/iemocap_multi_features.pkl'
IEMOCAP4_path = 'data/iemocap_multi_features_4.pkl'
CMUMOSEI7_path = ''
CHSIMS_path = 'data/CH-SIMSv2/chsims_v2_supervised_dialogue_features_paper_labels.pkl'
CHSIMS_SUPER_PATH = 'data/CH-SIMSv2/super.pkl'
CHSIMS_NOSUPER_PATH = 'data/CH-SIMSv2/nosuper.pkl'
CHSIMSV1_path = 'data/CH-SIMS/chsims_v1_dialogue_features_acc235.pkl'
MOSI_path = 'data/MOSI/mosi_dialogue_features.pkl'
MOSEI_path = 'data/MOSEI/mosei_dialogue_features_metricstop.pkl'
MOSEI_raw_path = 'data/MOSEI/unaligned_50.pkl'
NEWDATA_SPLIT_PATH = os.path.join(os.path.dirname(__file__), 'newdata.pkl')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def resolve_dataset_path(dataset_name, data_path_override):
    default_paths = {
        'MELD': MELD_path,
        'IEMOCAP': 'data/newdata_iemocap_features.pkl',
        'IEMOCAP4': IEMOCAP4_path,
        'CMUMOSEI2': 'data/cmumosei_multi_regression_features_acc25.pkl',
        'CMUMOSEI5': 'data/cmumosei_multi_regression_features_acc25.pkl',
        'CMUMOSEI7': CMUMOSEI7_path,
        'CHSIMS2': CHSIMS_path,
        'CHSIMS3': CHSIMS_path,
        'CHSIMS5': CHSIMS_path,
        'CHSIMSV2REG': CHSIMS_SUPER_PATH,
        'CHSIMSV2_NOSUPER2': CHSIMS_NOSUPER_PATH,
        'CHSIMSV2_NOSUPER3': CHSIMS_NOSUPER_PATH,
        'CHSIMSV2_NOSUPER5': CHSIMS_NOSUPER_PATH,
        'CHSIMSV1_2': CHSIMSV1_path,
        'CHSIMSV1_3': CHSIMSV1_path,
        'CHSIMSV1_5': CHSIMSV1_path,
        'MOSI2': MOSI_path,
        'MOSI2_HAS0': MOSI_path,
        'MOSI2_NON0': MOSI_path,
        'MOSI5': MOSI_path,
        'MOSI7': MOSI_path,
        'MOSEI2': MOSEI_path,
        'MOSEI2_HAS0': MOSEI_path,
        'MOSEI2_NON0': MOSEI_path,
        'MOSEI5': MOSEI_path,
        'MOSEI7': MOSEI_path,
        'MOSEIREG': MOSEI_raw_path,
    }
    if data_path_override:
        return data_path_override
    return default_paths.get(dataset_name, '')


def init_ddp(local_rank):
    try:
        if not dist.is_initialized():
            torch.cuda.set_device(local_rank)
            os.environ['RANK'] = str(local_rank)
            dist.init_process_group(backend='nccl', init_method='env://')
        else:
            logger.info('Distributed process group already initialized.')
    except Exception as e:
        logger.error(f'Failed to initialize distributed process group: {e}')
        raise


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def reduce_tensor(tensor: torch.Tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()

    return rt


def get_ddp_generator(seed=3407):
    local_rank = dist.get_rank()
    g = torch.Generator()
    g.manual_seed(seed + local_rank)

    return g


def _build_leave_one_split(sorted_vids, valid_count):
    stride = max(1, len(sorted_vids) // valid_count)
    valid_vids, selected = [], set()
    for idx in range(stride - 1, len(sorted_vids), stride):
        vid = sorted_vids[idx]
        valid_vids.append(vid)
        selected.add(vid)
        if len(valid_vids) == valid_count:
            break
    if len(valid_vids) < valid_count:
        for vid in sorted_vids:
            if vid in selected:
                continue
            valid_vids.append(vid)
            if len(valid_vids) == valid_count:
                break
    valid_vid_set = set(valid_vids)
    train_vids = [vid for vid in sorted_vids if vid not in valid_vid_set]

    return train_vids, valid_vids


def _build_target_utterance_split(dataset, sorted_vids, valid_count,
                                  target_valid_utterances):
    utt_lengths = [len(dataset.labels_emotion[vid]) for vid in sorted_vids]
    dp = [dict() for _ in range(valid_count + 1)]
    dp[0][0] = ()
    for idx, utt_len in enumerate(utt_lengths):
        upper = min(valid_count - 1, idx)
        for cnt in range(upper, -1, -1):
            current = list(dp[cnt].items())
            for utt_sum, picked in current:
                next_sum = utt_sum + utt_len
                if next_sum > target_valid_utterances:
                    continue
                if next_sum in dp[cnt + 1]:
                    continue
                dp[cnt + 1][next_sum] = picked + (idx,)
    chosen_indices = dp[valid_count].get(target_valid_utterances)
    if chosen_indices is None:
        return None, None
    valid_vids = [sorted_vids[idx] for idx in chosen_indices]
    valid_vid_set = set(valid_vids)
    train_vids = [vid for vid in sorted_vids if vid not in valid_vid_set]

    return train_vids, valid_vids


class _NewdataPlaceholder:
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.state = state


class _NewdataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            return _NewdataPlaceholder


def _load_pickle_fallback(path):
    with open(path, 'rb') as f:
        try:
            return pk.load(f, encoding='latin1')
        except Exception:
            f.seek(0)
            return _NewdataUnpickler(f).load()


def _extract_vid_from_item(item):
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        vid = item.get('vid')
        return vid if isinstance(vid, str) else None
    vid = getattr(item, 'vid', None)
    return vid if isinstance(vid, str) else None


def _load_newdata_split_vids(path):
    if not os.path.exists(path):
        return None
    data = _load_pickle_fallback(path)
    if not isinstance(data, dict):
        return None
    split_vids = {}
    for split_name in ('train', 'dev', 'test'):
        split_items = data.get(split_name)
        if not isinstance(split_items, (list, tuple)):
            return None
        vids = []
        for item in split_items:
            vid = _extract_vid_from_item(item)
            if vid is None:
                return None
            vids.append(vid)
        split_vids[split_name] = vids
    return split_vids


def _build_newdata_aligned_split(dataset, sorted_vids):
    split_vids = _load_newdata_split_vids(NEWDATA_SPLIT_PATH)
    if split_vids is None:
        return None, None
    dev_vids = split_vids['dev']
    if len(dev_vids) == 0:
        return None, None
    sorted_vid_set = set(sorted_vids)
    if any(vid not in sorted_vid_set for vid in dev_vids):
        return None, None
    dev_vid_set = set(dev_vids)
    train_vids = [vid for vid in sorted_vids if vid not in dev_vid_set]
    if len(train_vids) + len(dev_vids) != len(sorted_vids):
        return None, None
    return train_vids, dev_vids


def _build_standard_split_indices(dataset, dataset_name):
    standard_valid_dialogues = {
        'IEMOCAP': 12,
        'MELD': 114,
    }
    standard_valid_utterances = {
        'IEMOCAP': 647,
    }
    raw_vids = list(dataset.keys)
    sorted_vids = sorted(raw_vids)
    if dataset_name in standard_valid_dialogues:
        valid_count = standard_valid_dialogues[dataset_name]
    else:
        valid_count = max(1, int(round(len(sorted_vids) * 0.1)))
    valid_count = min(valid_count, len(sorted_vids) - 1)
    train_vids, valid_vids = None, None
    if dataset_name == 'IEMOCAP':
        train_vids, valid_vids = _build_newdata_aligned_split(dataset, sorted_vids)
    if (train_vids is None or valid_vids is None) and dataset_name in standard_valid_utterances:
        train_vids, valid_vids = _build_target_utterance_split(
            dataset=dataset,
            sorted_vids=sorted_vids,
            valid_count=valid_count,
            target_valid_utterances=standard_valid_utterances[dataset_name],
        )
    if train_vids is None or valid_vids is None:
        train_vids, valid_vids = _build_leave_one_split(sorted_vids, valid_count)

    idx_by_vid = {vid: i for i, vid in enumerate(dataset.keys)}
    train_indices = [idx_by_vid[vid] for vid in train_vids]
    valid_indices = [idx_by_vid[vid] for vid in valid_vids]

    return train_indices, valid_indices, train_vids, valid_vids


def _count_utterances(dataset, vids):
    return int(sum(len(dataset.labels_emotion[vid]) for vid in vids))



def get_data_loaders(path, dataset_class, dataset_name, batch_size, num_workers,
                     pin_memory, use_explicit_dev_split=False):
    if use_explicit_dev_split:
        base_trainset = dataset_class(path, split='train')
        if getattr(base_trainset, 'has_dev_split', False):
            trainset = base_trainset
            validset = dataset_class(path, split='dev')
            testset = dataset_class(path, split='test')
            train_vids = list(trainset.keys)
            valid_vids = list(validset.keys)
        else:
            train_indices, valid_indices, train_vids, valid_vids = _build_standard_split_indices(
                base_trainset, dataset_name)
            trainset = Subset(base_trainset, train_indices)
            validset = Subset(base_trainset, valid_indices)
            testset = dataset_class(path, train=False)
    else:
        base_trainset = dataset_class(path)
        train_indices, valid_indices, train_vids, valid_vids = _build_standard_split_indices(
            base_trainset, dataset_name)
        trainset = Subset(base_trainset, train_indices)
        validset = Subset(base_trainset, valid_indices)
        testset = dataset_class(path, train=False)
    train_sampler = DistributedSampler(trainset, shuffle=True)
    valid_sampler = DistributedSampler(validset, shuffle=False)

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=base_trainset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        validset,
        batch_size=batch_size,
        sampler=valid_sampler,
        collate_fn=base_trainset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        collate_fn=testset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    split_stats = {
        'train_dialogues': len(train_vids),
        'valid_dialogues': len(valid_vids),
        'test_dialogues': len(testset.keys),
        'train_utterances': _count_utterances(base_trainset, train_vids),
        'valid_utterances': _count_utterances(validset if getattr(base_trainset, 'has_dev_split', False) and use_explicit_dev_split else base_trainset, valid_vids),
        'test_utterances': _count_utterances(testset, testset.keys),
    }

    return train_loader, valid_loader, test_loader, split_stats


def get_checkpoint_state_dict(state_dict):
    return {k: v for k, v in state_dict.items() if not k.endswith('edge_weight')}


def draw_training_curves(plot_dir, epochs, train_loss_hist, valid_loss_hist,
                         train_f1_hist, valid_f1_hist, best_epoch,
                         test_loss_hist=None, test_f1_hist=None):
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss_hist, label='train_loss')
    plt.plot(epochs, valid_loss_hist, label='dev_loss')
    if test_loss_hist is not None:
        plt.plot(epochs, test_loss_hist, label='test_loss')
    plt.axvline(best_epoch, color='r', linestyle='--', label='best_dev_epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'loss_curve.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_f1_hist, label='train_f1_emo')
    plt.plot(epochs, valid_f1_hist, label='dev_f1_emo')
    if test_f1_hist is not None:
        plt.plot(epochs, test_f1_hist, label='test_f1_emo')
    plt.axvline(best_epoch, color='r', linestyle='--', label='best_dev_epoch')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'f1_curve.png'), dpi=300)
    plt.close()


def draw_tsne(plot_dir, features, labels, fig_name, fig_title, max_points=0):
    if max_points and len(features) > max_points:
        rng = np.random.RandomState(2024)
        selected_indices = rng.choice(len(features), size=max_points, replace=False)
        features = features[selected_indices]
        labels = labels[selected_indices]
    tsne = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=2024)
    points = tsne.fit_transform(features)
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(points[:, 0], points[:, 1], c=labels, s=8, cmap='tab10')
    plt.title(fig_title)
    plt.legend(*scatter.legend_elements(), title='Class', loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, fig_name), dpi=300)
    plt.close()


def infer_chsimsv1_embedding_dims(data_path):
    candidate_path = data_path or CHSIMSV1_path
    try:
        with open(candidate_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        if isinstance(data, dict) and 'trainVid' in data:
            train_vids = data['trainVid']
            if train_vids is not None and len(train_vids) > 0:
                sample_vid = train_vids[0]
                text_key = 'videoText0' if 'videoText0' in data else 'videoText'
                text_dim = int(np.asarray(data[text_key][sample_vid]).shape[-1])
                visual_dim = int(np.asarray(data['videoVisual'][sample_vid]).shape[-1])
                audio_dim = int(np.asarray(data['videoAudio'][sample_vid]).shape[-1])
                return [text_dim, visual_dim, audio_dim]
        if isinstance(data, dict) and 'train' in data and len(data['train']['id']) > 0:
            text_dim = int(np.asarray(data['train']['text'][0]).shape[-1])
            visual_dim = int(np.asarray(data['train']['vision'][0]).shape[-1])
            audio_dim = int(np.asarray(data['train']['audio'][0]).shape[-1])
            return [text_dim, visual_dim, audio_dim]
    except Exception as exc:
        logger.warning('Failed to infer CHSIMSV1 embedding dims from %s: %s',
                       candidate_path, exc)
    return [768, 709, 33]


def main(local_rank):
    print(f'Running main(**args) on rank {local_rank}.')
    init_ddp(local_rank)  # 初始化
    atexit.register(cleanup_distributed)

    today = datetime.datetime.now()
    name_ = args.modals + '_' + args.dataset

    cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device('cuda:{}'.format(local_rank) if cuda else 'cpu')
    if args.tensorboard:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter()

    n_epochs = args.epochs
    batch_size = args.batch_size
    modals = args.modals

    if args.dataset == 'IEMOCAP':
        embedding_dims = [1024, 342, 1582]
    elif args.dataset == 'IEMOCAP4':
        embedding_dims = [1024, 512, 100]
    elif args.dataset == 'MELD':
        embedding_dims = [1024, 342, 300]
    elif args.dataset == 'CMUMOSEI7':
        embedding_dims = [1024, 35, 384]
    elif args.dataset in ['CMUMOSEI2', 'CMUMOSEI5']:
        embedding_dims = infer_chsimsv1_embedding_dims(args.data_path)
    elif args.dataset in ['CHSIMS2', 'CHSIMS3', 'CHSIMS5']:
        embedding_dims = [768, 177, 25]
    elif args.dataset == 'CHSIMSV2REG':
        embedding_dims = [768, 177, 25]
    elif args.dataset in ['CHSIMSV2_NOSUPER2', 'CHSIMSV2_NOSUPER3',
                          'CHSIMSV2_NOSUPER5']:
        embedding_dims = [150, 177, 25]
    elif args.dataset in ['CHSIMSV1_2', 'CHSIMSV1_3', 'CHSIMSV1_5']:
        embedding_dims = infer_chsimsv1_embedding_dims(args.data_path)
    elif args.dataset in ['MOSI2', 'MOSI2_HAS0', 'MOSI2_NON0', 'MOSI5', 'MOSI7']:
        embedding_dims = [768, 20, 5]
    elif args.dataset in ['MOSEI2', 'MOSEI2_HAS0', 'MOSEI2_NON0', 'MOSEI5', 'MOSEI7', 'MOSEIREG']:
        embedding_dims = [768, 35, 74]

    if args.dataset == 'MELD' or args.dataset == 'CMUMOSEI7':
        n_classes_emo = 7
    elif args.dataset == 'CMUMOSEI2':
        n_classes_emo = 2
    elif args.dataset == 'CMUMOSEI5':
        n_classes_emo = 5
    elif args.dataset == 'IEMOCAP':
        n_classes_emo = 6
    elif args.dataset == 'IEMOCAP4':
        n_classes_emo = 4
    elif args.dataset == 'CHSIMS2':
        n_classes_emo = 2
    elif args.dataset == 'CHSIMS3':
        n_classes_emo = 3
    elif args.dataset == 'CHSIMS5':
        n_classes_emo = 5
    elif args.dataset == 'CHSIMSV2REG':
        n_classes_emo = 1
    elif args.dataset == 'CHSIMSV2_NOSUPER2':
        n_classes_emo = 2
    elif args.dataset == 'CHSIMSV2_NOSUPER3':
        n_classes_emo = 3
    elif args.dataset == 'CHSIMSV2_NOSUPER5':
        n_classes_emo = 5
    elif args.dataset == 'CHSIMSV1_2':
        n_classes_emo = 2
    elif args.dataset == 'CHSIMSV1_3':
        n_classes_emo = 3
    elif args.dataset == 'CHSIMSV1_5':
        n_classes_emo = 5
    elif args.dataset in ['MOSI2', 'MOSI2_HAS0', 'MOSI2_NON0']:
        n_classes_emo = 2
    elif args.dataset == 'MOSI5':
        n_classes_emo = 5
    elif args.dataset == 'MOSI7':
        n_classes_emo = 7
    elif args.dataset in ['MOSEI2', 'MOSEI2_HAS0', 'MOSEI2_NON0']:
        n_classes_emo = 2
    elif args.dataset == 'MOSEI5':
        n_classes_emo = 5
    elif args.dataset == 'MOSEI7':
        n_classes_emo = 7
    elif args.dataset == 'MOSEIREG':
        n_classes_emo = 1

    seed_everything()
    model = GraphSmile(args, embedding_dims, n_classes_emo)

    model = model.to(local_rank)
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )

    loss_function_emo = nn.NLLLoss()
    loss_function_sen = nn.NLLLoss()
    loss_function_shift = nn.NLLLoss()

    if args.loss_type == 'auto_loss':
        awl = AutomaticWeightedLoss(3)
        optimizer = optim.AdamW(
            [
                {
                    'params': model.parameters()
                },
                {
                    'params': awl.parameters(),
                    'weight_decay': 0
                },
            ],
            lr=args.lr,
            weight_decay=args.l2,
            amsgrad=True,
        )
    else:
        optimizer = optim.AdamW(model.parameters(),
                                lr=args.lr,
                                weight_decay=args.l2,
                                amsgrad=True)

    dataset_path = resolve_dataset_path(args.dataset, args.data_path)

    if args.dataset == 'MELD':
        train_loader, valid_loader, test_loader, split_stats = get_data_loaders(
            path=dataset_path,
            dataset_class=MELDDataset_BERT,
            dataset_name=args.dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
            use_explicit_dev_split=True,
        )
    elif args.dataset == 'IEMOCAP':
        train_loader, valid_loader, test_loader, split_stats = get_data_loaders(
            path=dataset_path,
            dataset_class=IEMOCAPDataset_BERT,
            dataset_name=args.dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
            use_explicit_dev_split=True,
        )
    elif args.dataset == 'IEMOCAP4':
        train_loader, valid_loader, test_loader, split_stats = get_data_loaders(
            path=dataset_path,
            dataset_class=IEMOCAPDataset_BERT4,
            dataset_name=args.dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
        )
    elif args.dataset == 'CMUMOSEI7':
        train_loader, valid_loader, test_loader, split_stats = get_data_loaders(
            path=dataset_path,
            dataset_class=CMUMOSEIDataset7,
            dataset_name=args.dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
        )
    elif args.dataset == 'CMUMOSEI2':
        train_loader, valid_loader, test_loader, split_stats = get_data_loaders(
            path=dataset_path,
            dataset_class=partial(MOSIDataset, num_emotion_classes=2),
            dataset_name=args.dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
            use_explicit_dev_split=True,
        )
    elif args.dataset == 'CMUMOSEI5':
        train_loader, valid_loader, test_loader, split_stats = get_data_loaders(
            path=dataset_path,
            dataset_class=partial(MOSIDataset, num_emotion_classes=5),
            dataset_name=args.dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
            use_explicit_dev_split=True,
        )
    elif args.dataset == 'CHSIMS2':
        train_loader, valid_loader, test_loader, split_stats = get_data_loaders(
            path=dataset_path,
            dataset_class=partial(CHSIMSDataset, num_emotion_classes=2),
            dataset_name=args.dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
            use_explicit_dev_split=True,
        )
    elif args.dataset == 'CHSIMS3':
        train_loader, valid_loader, test_loader, split_stats = get_data_loaders(
            path=dataset_path,
            dataset_class=partial(CHSIMSDataset, num_emotion_classes=3),
            dataset_name=args.dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
            use_explicit_dev_split=True,
        )
    elif args.dataset == 'CHSIMS5':
        train_loader, valid_loader, test_loader, split_stats = get_data_loaders(
            path=dataset_path,
            dataset_class=partial(CHSIMSDataset, num_emotion_classes=5),
            dataset_name=args.dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
            use_explicit_dev_split=True,
        )
    elif args.dataset == 'CHSIMSV2REG':
        train_loader, valid_loader, test_loader, split_stats = get_data_loaders(
            path=dataset_path,
            dataset_class=CHSIMSV2RegressionDataset,
            dataset_name=args.dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
            use_explicit_dev_split=True,
        )
    elif args.dataset in ['CHSIMSV2_NOSUPER2', 'CHSIMSV2_NOSUPER3',
                          'CHSIMSV2_NOSUPER5']:
        train_loader, valid_loader, test_loader, split_stats = get_data_loaders(
            path=dataset_path,
            dataset_class=CHSIMSV2NoSuperDataset,
            dataset_name=args.dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
            use_explicit_dev_split=True,
        )
    elif args.dataset == 'CHSIMSV1_2':
        train_loader, valid_loader, test_loader, split_stats = get_data_loaders(
            path=dataset_path,
            dataset_class=partial(CHSIMSV1RawDataset, num_emotion_classes=2),
            dataset_name=args.dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
            use_explicit_dev_split=True,
        )
    elif args.dataset == 'CHSIMSV1_3':
        train_loader, valid_loader, test_loader, split_stats = get_data_loaders(
            path=dataset_path,
            dataset_class=partial(CHSIMSV1RawDataset, num_emotion_classes=3),
            dataset_name=args.dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
            use_explicit_dev_split=True,
        )
    elif args.dataset == 'CHSIMSV1_5':
        train_loader, valid_loader, test_loader, split_stats = get_data_loaders(
            path=dataset_path,
            dataset_class=partial(CHSIMSV1RawDataset, num_emotion_classes=5),
            dataset_name=args.dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
            use_explicit_dev_split=True,
        )
    elif args.dataset in ['MOSI2', 'MOSI2_HAS0', 'MOSI2_NON0']:
        train_loader, valid_loader, test_loader, split_stats = get_data_loaders(
            path=dataset_path,
            dataset_class=partial(MOSIDataset,
                                  num_emotion_classes=2,
                                  binary_mode='non0'
                                  if args.dataset == 'MOSI2_NON0' else 'has0'),
            dataset_name=args.dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
            use_explicit_dev_split=True,
        )
    elif args.dataset == 'MOSI5':
        train_loader, valid_loader, test_loader, split_stats = get_data_loaders(
            path=dataset_path,
            dataset_class=partial(MOSIDataset, num_emotion_classes=5),
            dataset_name=args.dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
            use_explicit_dev_split=True,
        )
    elif args.dataset == 'MOSI7':
        train_loader, valid_loader, test_loader, split_stats = get_data_loaders(
            path=dataset_path,
            dataset_class=partial(MOSIDataset, num_emotion_classes=7),
            dataset_name=args.dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
            use_explicit_dev_split=True,
        )
    elif args.dataset in ['MOSEI2', 'MOSEI2_HAS0', 'MOSEI2_NON0']:
        train_loader, valid_loader, test_loader, split_stats = get_data_loaders(
            path=dataset_path,
            dataset_class=partial(MOSIDataset,
                                  num_emotion_classes=2,
                                  binary_mode='non0'
                                  if args.dataset == 'MOSEI2_NON0' else 'has0'),
            dataset_name=args.dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
            use_explicit_dev_split=True,
        )
    elif args.dataset == 'MOSEI5':
        train_loader, valid_loader, test_loader, split_stats = get_data_loaders(
            path=dataset_path,
            dataset_class=partial(MOSIDataset, num_emotion_classes=5),
            dataset_name=args.dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
            use_explicit_dev_split=True,
        )
    elif args.dataset == 'MOSEI7':
        train_loader, valid_loader, test_loader, split_stats = get_data_loaders(
            path=dataset_path,
            dataset_class=partial(MOSIDataset, num_emotion_classes=7),
            dataset_name=args.dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
            use_explicit_dev_split=True,
        )
    elif args.dataset == 'MOSEIREG':
        train_loader, valid_loader, test_loader, split_stats = get_data_loaders(
            path=dataset_path,
            dataset_class=MOSEIRegressionDataset,
            dataset_name=args.dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
            use_explicit_dev_split=True,
        )
    else:
        print('There is no such dataset')
        return

    if local_rank == 0:
        print('Dataset path:', dataset_path)
        if getattr(args, 'auto_iemocap4_lite_notes', None):
            print('Auto IEMOCAP4 lite preset:',
                  ', '.join(args.auto_iemocap4_lite_notes))
        if getattr(args, 'auto_chsims2_binary_notes', None):
            print('Auto CHSIMS2 binary preset:',
                  ', '.join(args.auto_chsims2_binary_notes))
        if getattr(args, 'auto_chsimsv2_nosuper_notes', None):
            print('Auto CHSIMSv2 nosuper preset:',
                  ', '.join(args.auto_chsimsv2_nosuper_notes))
        print('Standard split for {} -> dialogues(train/valid/test): {}/{}/{}, utterances(train/valid/test): {}/{}/{}'.
              format(
                  args.dataset,
                  split_stats['train_dialogues'],
                  split_stats['valid_dialogues'],
                  split_stats['test_dialogues'],
                  split_stats['train_utterances'],
                  split_stats['valid_utterances'],
                  split_stats['test_utterances'],
              ))

    run_stamp = '{}{}{}_{}{}{}'.format(today.year, today.month, today.day,
                                       today.hour, today.minute, today.second)
    plot_dir = os.path.join(args.plot_dir,
                            '{}_{}_{}'.format(args.modals, args.dataset,
                                              run_stamp))
    checkpoint_path = os.path.join(
        args.checkpoint_dir,
        'best_{}_{}_{}.pt'.format(args.modals, args.dataset, args.classify))

    if local_rank == 0:
        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    best_dev_epoch = -1
    best_dev_metric = None
    train_loss_hist, valid_loss_hist = [], []
    train_f1_emo_hist, valid_f1_emo_hist = [], []
    test_loss_hist, test_f1_emo_hist = [], []
    best_test_metrics = None
    best_label_emo, best_pred_emo = None, None
    best_label_sen, best_pred_sen = None, None
    best_test_extracted_feats = None
    no_improve_epochs = 0
    early_stop_patience = max(int(getattr(args, 'early_stop_patience', 0)), 0)
    regression_metric_mode = 'corr' if args.dataset == 'CHSIMSV2REG' else 'msa'

    for epoch in range(n_epochs):
        train_loader.sampler.set_epoch(epoch)
        valid_loader.sampler.set_epoch(epoch)

        start_time = time.time()

        train_loss, train_loss_adv, _, _, train_acc_emo, train_f1_emo, _, _, train_acc_sen, train_f1_sen, train_acc_sft, train_f1_sft, _, _, _ = train_or_eval_model(
            model,
            loss_function_emo,
            loss_function_sen,
            loss_function_shift,
            train_loader,
            epoch,
            cuda,
            args.modals,
            optimizer,
            True,
            args.dataset,
            args.loss_type,
            args.lambd,
            args.epochs,
            args.classify,
            args.shift_win,
            args.contrastive_weight,
            args.decouple_weight,
            args.warmup_epochs,
            args.supcon_temperature,
            args.label_smoothing,
            args.class_balance_beta,
            args.hard_negative_weight,
            args.pair_aux_weight_04,
            args.pair_aux_weight_25,
            args.proto_aux_weight,
            args.hier_aux_weight,
            args.hier_kl_weight,
            args.temporal_aux_weight,
            args.modality_adv_weight,
            args.pair04_sample_boost,
            args.pair04_margin,
            args.pair04_margin_weight,
            True,
        )

        valid_loss, valid_loss_adv, _, _, valid_acc_emo, valid_f1_emo, _, _, valid_acc_sen, valid_f1_sen, valid_acc_sft, valid_f1_sft, _, _, _ = train_or_eval_model(
            model,
            loss_function_emo,
            loss_function_sen,
            loss_function_shift,
            valid_loader,
            epoch,
            cuda,
            args.modals,
            None,
            False,
            args.dataset,
            args.loss_type,
            args.lambd,
            args.epochs,
            args.classify,
            args.shift_win,
            args.contrastive_weight,
            args.decouple_weight,
            args.warmup_epochs,
            args.supcon_temperature,
            args.label_smoothing,
            args.class_balance_beta,
            args.hard_negative_weight,
            args.pair_aux_weight_04,
            args.pair_aux_weight_25,
            args.proto_aux_weight,
            args.hier_aux_weight,
            args.hier_kl_weight,
            args.temporal_aux_weight,
            args.modality_adv_weight,
            args.pair04_sample_boost,
            args.pair04_margin,
            args.pair04_margin_weight,
        )

        train_loss_avg = reduce_tensor(torch.tensor(train_loss,
                                                    device=device)).item()
        valid_loss_avg = reduce_tensor(torch.tensor(valid_loss,
                                                    device=device)).item()
        train_metric_main_avg = reduce_tensor(torch.tensor(train_f1_emo,
                                                           device=device)).item()
        valid_metric_main_avg = reduce_tensor(torch.tensor(valid_f1_emo,
                                                           device=device)).item()
        valid_metric_aux_avg = reduce_tensor(torch.tensor(valid_f1_sen,
                                                          device=device)).item()

        if args.classify == 'regression':
            train_metric_a_avg = reduce_tensor(torch.tensor(train_acc_emo,
                                                            device=device)).item()
            valid_metric_a_avg = reduce_tensor(torch.tensor(valid_acc_emo,
                                                            device=device)).item()
            train_metric_b_avg = train_metric_main_avg
            valid_metric_b_avg = valid_metric_main_avg
            train_metric_c_avg = reduce_tensor(torch.tensor(train_acc_sen,
                                                            device=device)).item()
            valid_metric_c_avg = reduce_tensor(torch.tensor(valid_acc_sen,
                                                            device=device)).item()
            if regression_metric_mode == 'corr':
                print(
                    'epoch: {}, train_loss: {}, train_corr: {}, train_r_square: {}, train_mae: {}, valid_loss: {}, valid_corr: {}, valid_r_square: {}, valid_mae: {}'
                    .format(
                        epoch + 1,
                        train_loss_avg,
                        round(train_metric_a_avg, 4),
                        round(train_metric_b_avg, 4),
                        round(train_metric_c_avg, 4),
                        valid_loss_avg,
                        round(valid_metric_a_avg, 4),
                        round(valid_metric_b_avg, 4),
                        round(valid_metric_c_avg, 4),
                    ))
            else:
                print(
                    'epoch: {}, train_loss: {}, train_msa: {}, train_f1: {}, train_mae: {}, valid_loss: {}, valid_msa: {}, valid_f1: {}, valid_mae: {}'
                    .format(
                        epoch + 1,
                        train_loss_avg,
                        round(train_metric_a_avg, 4),
                        round(train_metric_b_avg, 4),
                        round(train_metric_c_avg, 4),
                        valid_loss_avg,
                        round(valid_metric_a_avg, 4),
                        round(valid_metric_b_avg, 4),
                        round(valid_metric_c_avg, 4),
                    ))
            current_dev_target = valid_metric_a_avg
        elif args.classify == 'unsupervised':
            train_loss_adv_avg = reduce_tensor(torch.tensor(train_loss_adv,
                                                            device=device)).item()
            valid_loss_adv_avg = reduce_tensor(torch.tensor(valid_loss_adv,
                                                            device=device)).item()
            print(
                'epoch: {}, train_loss: {}, train_loss_adv: {}, valid_loss: {}, valid_loss_adv: {}'
                .format(
                    epoch + 1,
                    train_loss_avg,
                    round(train_loss_adv_avg, 4),
                    valid_loss_avg,
                    round(valid_loss_adv_avg, 4),
                ))
            current_dev_target = -valid_loss_avg
        else:
            train_f1_emo_avg = train_metric_main_avg
            valid_f1_emo_avg = valid_metric_main_avg
            valid_f1_sen_avg = valid_metric_aux_avg
            print(
                'epoch: {}, train_loss: {}, train_loss_adv: {}, train_acc_emo: {}, train_f1_emo: {}, valid_loss: {}, valid_loss_adv: {}, valid_acc_emo: {}, valid_f1_emo: {}'
                .format(
                    epoch + 1,
                    train_loss_avg,
                    train_loss_adv,
                    train_acc_emo,
                    train_f1_emo_avg,
                    valid_loss_avg,
                    valid_loss_adv,
                    valid_acc_emo,
                    valid_f1_emo_avg,
                ))
            if args.classify == 'emotion':
                current_dev_target = valid_f1_emo_avg
            else:
                current_dev_target = valid_f1_sen_avg

        improved = best_dev_metric is None or current_dev_target > best_dev_metric
        if improved:
            best_dev_metric = current_dev_target
            best_dev_epoch = epoch + 1
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if local_rank == 0:
            test_loss, test_loss_adv, test_label_emo, test_pred_emo, test_acc_emo, test_f1_emo, test_label_sen, test_pred_sen, test_acc_sen, test_f1_sen, test_acc_sft, test_f1_sft, _, _, test_extracted_feats = train_or_eval_model(
                model,
                loss_function_emo,
                loss_function_sen,
                loss_function_shift,
                test_loader,
                epoch,
                cuda,
                args.modals,
                None,
                False,
                args.dataset,
                args.loss_type,
                args.lambd,
                args.epochs,
                args.classify,
                args.shift_win,
                args.contrastive_weight,
                args.decouple_weight,
                args.warmup_epochs,
                args.supcon_temperature,
                args.label_smoothing,
                args.class_balance_beta,
                args.hard_negative_weight,
                args.pair_aux_weight_04,
                args.pair_aux_weight_25,
                args.proto_aux_weight,
                args.hier_aux_weight,
                args.hier_kl_weight,
                args.temporal_aux_weight,
                args.modality_adv_weight,
                args.pair04_sample_boost,
                args.pair04_margin,
                args.pair04_margin_weight,
                collect_features=True,
            )

            train_loss_hist.append(train_loss_avg)
            valid_loss_hist.append(valid_loss_avg)
            test_loss_hist.append(test_loss)
            if args.classify == 'regression':
                train_f1_emo_hist.append(train_metric_a_avg)
                valid_f1_emo_hist.append(valid_metric_a_avg)
                test_f1_emo_hist.append(test_acc_emo)
                if regression_metric_mode == 'corr':
                    print(
                        'test_loss: {}, test_corr: {}, test_r_square: {}, test_mae: {}'
                        .format(
                            test_loss,
                            test_acc_emo,
                            test_f1_emo,
                            test_acc_sen,
                        ))
                else:
                    print(
                        'test_loss: {}, test_msa: {}, test_f1: {}, test_mae: {}'
                        .format(
                            test_loss,
                            test_acc_emo,
                            test_f1_emo,
                            test_acc_sen,
                        ))
            elif args.classify == 'unsupervised':
                train_f1_emo_hist.append(-train_loss_avg)
                valid_f1_emo_hist.append(-valid_loss_avg)
                test_f1_emo_hist.append(-test_loss)
                print(
                    'test_loss: {}, test_loss_adv: {}'
                    .format(
                        test_loss,
                        test_loss_adv,
                    ))
            else:
                train_f1_emo_hist.append(train_f1_emo_avg)
                valid_f1_emo_hist.append(valid_f1_emo_avg)
                test_f1_emo_hist.append(test_f1_emo)
                print(
                    'test_loss: {}, test_loss_adv: {}, test_acc_emo: {}, test_f1_emo: {}, test_acc_sen: {}, test_f1_sen: {}, test_acc_sft: {}, test_f1_sft: {}'
                    .format(
                        test_loss,
                        test_loss_adv,
                        test_acc_emo,
                        test_f1_emo,
                        test_acc_sen,
                        test_f1_sen,
                        test_acc_sft,
                        test_f1_sft,
                    ))

            if improved:
                best_label_emo, best_pred_emo = test_label_emo, test_pred_emo
                best_label_sen, best_pred_sen = test_label_sen, test_pred_sen
                best_test_extracted_feats = test_extracted_feats
                if args.classify == 'regression':
                    if regression_metric_mode == 'corr':
                        best_test_metrics = {
                            'select_best_by': args.select_best_by,
                            'selected_target_corr': best_dev_metric,
                            'test_loss': test_loss,
                            'test_corr': test_acc_emo,
                            'test_r_square': test_f1_emo,
                            'test_mae': test_acc_sen,
                        }
                    else:
                        best_test_metrics = {
                            'select_best_by': args.select_best_by,
                            'selected_target_msa': best_dev_metric,
                            'test_loss': test_loss,
                            'test_msa': test_acc_emo,
                            'test_f1': test_f1_emo,
                            'test_mae': test_acc_sen,
                        }
                elif args.classify == 'unsupervised':
                    best_test_metrics = {
                        'select_best_by': 'dev_loss',
                        'selected_target_neg_loss': best_dev_metric,
                        'test_loss': test_loss,
                        'test_loss_adv': test_loss_adv,
                    }
                else:
                    best_test_metrics = {
                        'select_best_by': args.select_best_by,
                        'selected_target_f1': best_dev_metric,
                        'test_loss': test_loss,
                        'test_acc_emo': test_acc_emo,
                        'test_f1_emo': test_f1_emo,
                        'test_acc_sen': test_acc_sen,
                        'test_f1_sen': test_f1_sen,
                        'test_acc_sft': test_acc_sft,
                        'test_f1_sft': test_f1_sft,
                    }
                torch.save(
                    {
                        'epoch': best_dev_epoch,
                        'best_dev_metric': best_dev_metric,
                        'model_state_dict': get_checkpoint_state_dict(
                            deepcopy(model.module.state_dict())),
                    }, checkpoint_path)
                print(
                    'Best dev updated at epoch {} with target_metric={}'.format(
                        best_dev_epoch, round(best_dev_metric, 4)))

            print(
                'epoch time: {} sec, {}'.format(
                    round(time.time() - start_time, 2),
                    time.strftime('%Y-%m-%d %H:%M:%S',
                                  time.localtime(time.time())),
                ))
            print('-' * 100)

        dist.barrier()

        if early_stop_patience > 0 and no_improve_epochs >= early_stop_patience:
            if local_rank == 0:
                print(
                    'Early stop triggered at epoch {} after {} epochs without dev improvement.'
                    .format(epoch + 1, no_improve_epochs))
            break

        if args.tensorboard:
            if local_rank == 0:
                writer.add_scalar("train/loss", train_loss_avg, epoch)
                writer.add_scalar("dev/loss", valid_loss_avg, epoch)
                writer.add_scalar("test/loss", test_loss, epoch)
                writer.add_scalar("train/f1_emo", train_f1_emo_hist[-1], epoch)
                writer.add_scalar("dev/f1_emo", valid_f1_emo_hist[-1], epoch)
                writer.add_scalar("test/f1_emo", test_f1_emo_hist[-1], epoch)

        if epoch == 1:
            allocated_memory = torch.cuda.memory_allocated()
            reserved_memory = torch.cuda.memory_reserved()
            print(f"Allocated Memory: {allocated_memory / 1024**2:.2f} MB")
            print(f"Reserved Memory: {reserved_memory / 1024**2:.2f} MB")
            print(
                f"All Memory: {(allocated_memory + reserved_memory) / 1024**2:.2f} MB"
            )

    if args.tensorboard:
        writer.close()
    if local_rank == 0:
        epochs = list(range(1, len(train_loss_hist) + 1))
        draw_training_curves(
            plot_dir=plot_dir,
            epochs=epochs,
            train_loss_hist=train_loss_hist,
            valid_loss_hist=valid_loss_hist,
            train_f1_hist=train_f1_emo_hist,
            valid_f1_hist=valid_f1_emo_hist,
            best_epoch=best_dev_epoch,
            test_loss_hist=test_loss_hist,
            test_f1_hist=test_f1_emo_hist,
        )
        if not args.disable_tsne and args.classify not in ['regression', 'unsupervised']:
            print('Drawing t-SNE figures...')
            draw_tsne(
                plot_dir=plot_dir,
                features=best_test_extracted_feats,
                labels=best_label_emo,
                fig_name='tsne_emotion.png',
                fig_title='t-SNE on Test Features (Emotion Labels)',
                max_points=args.tsne_max_points,
            )
            draw_tsne(
                plot_dir=plot_dir,
                features=best_test_extracted_feats,
                labels=best_label_sen,
                fig_name='tsne_sentiment.png',
                fig_title='t-SNE on Test Features (Sentiment Labels)',
                max_points=args.tsne_max_points,
            )

        print("Test performance..")
        if args.classify == 'regression':
            if regression_metric_mode == 'corr':
                print("Corr: {}, R_squre: {}, MAE: {}".format(
                    best_test_metrics['test_corr'],
                    best_test_metrics['test_r_square'],
                    best_test_metrics['test_mae'],
                ))
            else:
                print("MSA: {}, F1: {}, MAE: {}".format(
                    best_test_metrics['test_msa'],
                    best_test_metrics['test_f1'],
                    best_test_metrics['test_mae'],
                ))
        elif args.classify == 'unsupervised':
            print("Unsupervised test_loss: {}, test_loss_adv: {}".format(
                best_test_metrics['test_loss'],
                best_test_metrics['test_loss_adv'],
            ))
        else:
            print("Acc: {}, F-Score: {}".format(best_test_metrics['test_acc_emo'],
                                                best_test_metrics['test_f1_emo']))
        print("Best dev epoch: {}".format(best_dev_epoch))
        print("Best test metrics: {}".format(best_test_metrics))
        print("Best model saved to: {}".format(checkpoint_path))
        if args.disable_tsne or args.classify in ['regression', 'unsupervised']:
            print("Curves saved to: {}".format(plot_dir))
        else:
            print("Curves and t-SNE figures saved to: {}".format(plot_dir))
        if not os.path.exists("results/record_{}_{}_{}.pk".format(
                today.year, today.month, today.day)):
            with open(
                    "results/record_{}_{}_{}.pk".format(
                        today.year, today.month, today.day),
                    "wb",
            ) as f:
                pk.dump({}, f)
        with open(
                "results/record_{}_{}_{}.pk".format(today.year, today.month,
                                                    today.day),
                "rb",
        ) as f:
            record = pk.load(f)
        key_ = name_
        if args.classify == 'regression':
            record_value = best_test_metrics['test_corr'] if regression_metric_mode == 'corr' else best_test_metrics['test_msa']
        elif args.classify == 'unsupervised':
            record_value = -best_test_metrics['test_loss']
        else:
            record_value = best_test_metrics['test_f1_emo']
        if record.get(key_, False):
            record[key_].append(record_value)
        else:
            record[key_] = [record_value]
        if args.classify not in ['regression', 'unsupervised']:
            if record.get(key_ + "record", False):
                record[key_ + "record"].append(
                    classification_report(best_label_emo,
                                          best_pred_emo,
                                          digits=4,
                                          zero_division=0))
            else:
                record[key_ + "record"] = [
                    classification_report(best_label_emo,
                                          best_pred_emo,
                                          digits=4,
                                          zero_division=0)
                ]
        with open(
                "results/record_{}_{}_{}.pk".format(today.year, today.month,
                                                    today.day),
                "wb",
        ) as f:
            pk.dump(record, f)

        if args.classify not in ['regression', 'unsupervised']:
            print(
                classification_report(best_label_emo,
                                      best_pred_emo,
                                      digits=4,
                                      zero_division=0))
            print(confusion_matrix(best_label_emo, best_pred_emo))
            print(
                classification_report(best_label_sen,
                                      best_pred_sen,
                                      digits=4,
                                      zero_division=0))
            print(confusion_matrix(best_label_sen, best_pred_sen))

    cleanup_distributed()


if __name__ == "__main__":
    print(args)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("not args.no_cuda:", not args.no_cuda)
    n_gpus = torch.cuda.device_count()
    print(f"Use {n_gpus} GPUs")
    mp.spawn(fn=main, args=(), nprocs=n_gpus)
