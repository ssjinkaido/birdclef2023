import numpy as np
import pandas as pd
import random
import glob
import os, shutil
from tqdm import tqdm
tqdm.pandas()
import copy
import joblib
from collections import defaultdict
import gc
import math
import cv2
import time
import ast
# PyTorch 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup
import sklearn
from audiomentations import Compose, AddGaussianNoise, AddGaussianSNR, Gain, GainTransition, TanhDistortion, SpecCompose, SpecFrequencyMask, AddBackgroundNoise, Shift, LowPassFilter
import torchaudio.transforms as T
# from torchlibrosa.augmentation import SpecAugmentation
import librosa as lb
import soundfile as sf
import colorednoise as cn
import timm
import sklearn
import json
from datetime import datetime
import albumentations as A
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
import optuna
from optuna.samplers import TPESampler
class CFG:
    seed = 1
    
    # Audio duration, sample rate, and length
    duration = 5# second
    sample_rate = 32000
    audio_len = duration*sample_rate
    
    # STFT parameters
    nfft = 768
    n_mels = 128
    fmin = 20
    fmax = 16000
#     model_name = "eca_nfnet_l0"
    model_name = "tf_efficientnetv2_s"
    # model_name = "tf_efficientnetv2_s_in21ft1k"
    train_bs = 128
    valid_bs = train_bs * 4
    num_fold = 5
    epoch_warm_up = 0
    total_epoch = 200
    learning_rate = 3e-4
    weight_decay = 0.01
    thr_upsample = 10
    # thr_downsample = 500
    mix_up = 0.8
    hop_length = 256
    train_with_mixup=True
    num_channels = 1
    use_spec_augment = False
    use_drop_path = True

    # Class Labels for BirdCLEF 23
    class_names = sorted(os.listdir('birdclef-2023/train_audio/'))
    num_classes = len(class_names)
    class_labels = list(range(num_classes))
    label2name = dict(zip(class_labels, class_names))
    name2label = {v:k for k,v in label2name.items()}
    
    # Class Labels for BirdCLEF 21 & 22
    # class_names2 = sorted(set(os.listdir('birdclef-2022/train_audio/')))
    class_names2 = sorted(set(os.listdir('birdclef-2021/train_short_audio/')
                       +os.listdir('birdclef-2022/train_audio/')
                       +os.listdir('birdclef-2020/train_audio/')))
    class_names2 = [s for s in class_names2 if s not in ('greegr', 'categr', 'yefcan')]
    num_classes2 = len(class_names2)
    class_labels2 = list(range(num_classes2))
    label2name2 = dict(zip(class_labels2, class_names2))
    name2label2 = {v:k for k,v in label2name2.items()}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(seed = 42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print('> SEEDING DONE')
set_seed(CFG.seed)  

def init_logger(log_file='train_10.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

LOGGER = init_logger()
now = datetime.now()
datetime_now = now.strftime("%m/%d/%Y, %H:%M:%S")
LOGGER.info(f"Date :{datetime_now}")
LOGGER.info(f"Duration: {CFG.duration}")
LOGGER.info(f"Sample rate: {CFG.sample_rate}")
LOGGER.info(f"nfft: {CFG.nfft}")
LOGGER.info(f"fmin: {CFG.fmin}")
LOGGER.info(f"nmels: {CFG.n_mels}")
LOGGER.info(f"fmax: {CFG.fmax}")
LOGGER.info(f"trainbs: {CFG.train_bs}")
LOGGER.info(f"validbs: {CFG.valid_bs}")
LOGGER.info(f"epochwarmup: {CFG.epoch_warm_up}")
LOGGER.info(f"totalepoch: {CFG.total_epoch}")
LOGGER.info(f"learningrate: {CFG.learning_rate}")
LOGGER.info(f"weightdecay: {CFG.weight_decay}")
LOGGER.info(f"thrupsample: {CFG.thr_upsample}")
LOGGER.info(f"model_name: {CFG.model_name}")
LOGGER.info(f"mix_up: {CFG.mix_up}")
LOGGER.info(f"hop_length: {CFG.hop_length}")
LOGGER.info(f"train_with_mixup: {CFG.train_with_mixup}")
LOGGER.info(f"num_channels: {CFG.num_channels}")
LOGGER.info(f"use_spec_augmenter: {CFG.use_spec_augment}")
LOGGER.info(f"use_drop_path: {CFG.use_drop_path}")
BASE_PATH = 'birdclef-2020'
BASE_PATH1 = 'birdclef-2021'
BASE_PATH2 = 'birdclef-2022'
BASE_PATH3 = 'birdclef-2023'

df_23 = pd.read_csv(f'{BASE_PATH3}/train_metadata.csv')
df_23['filepath'] = BASE_PATH3 + '/train_audio/' + df_23.filename
# df_23['filepath'] = BASE_PATH3 + df_23.filename.str.replace('.ogg', '.wav')
df_23['target'] = df_23.primary_label.map(CFG.name2label)
df_23['birdclef'] = '23'
df_23['filename'] = df_23.filepath.map(lambda x: x.split('/')[-1])
df_23['xc_id'] = df_23.filepath.map(lambda x: x.split('/')[-1].split('.')[0])

df_21 = pd.read_csv(f'{BASE_PATH1}/train_metadata.csv')
df_21['filepath'] = BASE_PATH1 + '/train_short_audio/' + df_21.primary_label + '/' + df_21.filename
df_21['target'] = df_21.primary_label.map(CFG.name2label2)
df_21['birdclef'] = '21'
corrupt_paths = ['birdclef-2021/train_short_audio/houwre/XC590621.ogg',
                 'birdclef-2021/train_short_audio/cogdov/XC579430.ogg']
df_21 = df_21[~df_21.filepath.isin(corrupt_paths)] # remove all zero audios

# BirdCLEF-2022
df_22 = pd.read_csv(f'{BASE_PATH2}/train_metadata.csv')
df_22['filepath'] = BASE_PATH2 + '/train_audio/' + df_22.filename
df_22['target'] = df_22.primary_label.map(CFG.name2label2)
df_22['birdclef'] = '22'


df_20 = pd.read_csv(f'{BASE_PATH}/train.csv')
df_20['primary_label'] = df_20['ebird_code']
df_20['filepath'] = BASE_PATH + '/train_audio/' + df_20.primary_label + '/' + df_20.filename
df_20['scientific_name'] = df_20['sci_name']
df_20['common_name'] = df_20['species']
# df_20['target'] = df_20.primary_label.map(CFG.name2label2)
df_20['birdclef'] = '20'

df_pre = pd.concat([df_20, df_21, df_22], axis=0, ignore_index=True)
df_pre['filename'] = df_pre.filepath.map(lambda x: x.split('/')[-1])
df_pre['xc_id'] = df_pre.filepath.map(lambda x: x.split('/')[-1].split('.')[0])
nodup_idx = df_pre[['xc_id','primary_label','author']].drop_duplicates().index
df_pre = df_pre.loc[nodup_idx].reset_index(drop=True)
corrupt_mp3s = ["birdclef-2020/train_audio/amepip/XC393498.mp3",
"birdclef-2020/train_audio/amepip/XC395154.mp3",
"birdclef-2020/train_audio/amepip/XC395155.mp3",
"birdclef-2020/train_audio/barswa/XC378710.mp3",
"birdclef-2020/train_audio/bawwar/XC177379.mp3",
"birdclef-2020/train_audio/bkhgro/XC233132.mp3",
"birdclef-2020/train_audio/bkhgro/XC233343.mp3",
"birdclef-2020/train_audio/btbwar/XC504005.mp3",
"birdclef-2020/train_audio/btbwar/XC504006.mp3",
"birdclef-2020/train_audio/cangoo/XC209702.mp3",
"birdclef-2020/train_audio/canwre/XC359436.mp3",
"birdclef-2020/train_audio/comgol/XC187568.mp3",
"birdclef-2020/train_audio/comgol/XC234890.mp3",
"birdclef-2020/train_audio/comgol/XC462425.mp3",
"birdclef-2020/train_audio/commer/XC375582.mp3",
"birdclef-2020/train_audio/comrav/XC147252.mp3",
"birdclef-2020/train_audio/comrav/XC199733.mp3",
"birdclef-2020/train_audio/comrav/XC262905.mp3",
"birdclef-2020/train_audio/comred/XC489918.mp3",
"birdclef-2020/train_audio/comyel/XC181588.mp3",
"birdclef-2020/train_audio/eargre/XC182452.mp3",
"birdclef-2020/train_audio/eucdov/XC167576.mp3",
"birdclef-2020/train_audio/eucdov/XC168488.mp3",
"birdclef-2020/train_audio/eucdov/XC505275.mp3",
"birdclef-2020/train_audio/fiscro/XC282291.mp3",
"birdclef-2020/train_audio/gadwal/XC182414.mp3",
"birdclef-2020/train_audio/gadwal/XC484631.mp3",
"birdclef-2020/train_audio/grnher/XC213291.mp3",
"birdclef-2020/train_audio/grtgra/XC395021.mp3",
"birdclef-2020/train_audio/houfin/XC233711.mp3",
"birdclef-2020/train_audio/leafly/XC180201.mp3",
"birdclef-2020/train_audio/lotduc/XC476155.mp3",
"birdclef-2020/train_audio/lotduc/XC476542.mp3",
"birdclef-2020/train_audio/merlin/XC183106.mp3",
"birdclef-2020/train_audio/pingro/XC341610.mp3",
"birdclef-2020/train_audio/pingro/XC504893.mp3",
"birdclef-2020/train_audio/pingro/XC505016.mp3",
"birdclef-2020/train_audio/rethaw/XC233997.mp3",
"birdclef-2020/train_audio/rewbla/XC321249.mp3",
"birdclef-2020/train_audio/rocpig/XC163851.mp3",
"birdclef-2020/train_audio/rocpig/XC266224.mp3",
"birdclef-2020/train_audio/rocpig/XC269762.mp3",
"birdclef-2020/train_audio/sheowl/XC489388.mp3",
"birdclef-2020/train_audio/snobun/XC152757.mp3",
"birdclef-2020/train_audio/tunswa/XC505006.mp3",
"birdclef-2020/train_audio/woothr/XC202293.mp3",
"birdclef-2020/train_audio/yerwar/XC173998.mp3",
"birdclef-2020/train_audio/yerwar/XC244402.mp3"]
df_pre = df_pre[~df_pre.xc_id.isin(df_23.xc_id)].reset_index(drop=True)
df_pre = df_pre[~df_pre.filepath.isin(corrupt_mp3s)]
df_pre['filepath'] =df_pre['filepath'].str.replace('.mp3', '.ogg')
df_pre = df_pre[df_pre['primary_label']!='greegr'].reset_index(drop=True)
df_pre['target'] = df_pre.primary_label.map(CFG.name2label2)
df_pre = df_pre[['filename','filepath','primary_label','secondary_labels',
                 'rating','author','xc_id','scientific_name',
                'common_name','target','birdclef']]
df_pre["fold"] = -1
skf1 = StratifiedKFold(n_splits=CFG.num_fold, shuffle=True, random_state=CFG.seed)
for fold, (train_idx, val_idx) in enumerate(skf1.split(df_pre, df_pre['primary_label'])):
    df_pre.loc[val_idx, 'fold'] = fold
LOGGER.info(len(df_pre))
skf2 = StratifiedKFold(n_splits=CFG.num_fold, shuffle=True, random_state=CFG.seed)
df_23["fold"] = -1
for fold, (train_idx, val_idx) in enumerate(skf2.split(df_23, df_23['primary_label'])):
    df_23.loc[val_idx, 'fold'] = fold
def upsample_data(df, thr=20):
    # get the class distribution
    class_dist = df['primary_label'].value_counts()

    # identify the classes that have less than the threshold number of samples
    down_classes = class_dist[class_dist < thr].index.tolist()

    # create an empty list to store the upsampled dataframes
    up_dfs = []

    # loop through the undersampled classes and upsample them
    for c in down_classes:
        # get the dataframe for the current class
        class_df = df.query("primary_label==@c")
        # find number of samples to add
        num_up = thr - class_df.shape[0]
        # upsample the dataframe
        class_df = class_df.sample(n=num_up, replace=True, random_state=CFG.seed)
        # append the upsampled dataframe to the list
        up_dfs.append(class_df)

    # concatenate the upsampled dataframes and the original dataframe
    up_df = pd.concat([df] + up_dfs, axis=0, ignore_index=True)
    
    return up_df
def downsample_data(df, thr=500):
    # get the class distribution
    class_dist = df['primary_label'].value_counts()
    
    # identify the classes that have less than the threshold number of samples
    up_classes = class_dist[class_dist > thr].index.tolist()

    # create an empty list to store the upsampled dataframes
    down_dfs = []

    # loop through the undersampled classes and upsample them
    for c in up_classes:
        # get the dataframe for the current class
        class_df = df.query("primary_label==@c")
        # Remove that class data
        df = df.query("primary_label!=@c")
        # upsample the dataframe
        class_df = class_df.sample(n=thr, replace=False, random_state=CFG.seed)
        # append the upsampled dataframe to the list
        down_dfs.append(class_df)

    # concatenate the upsampled dataframes and the original dataframe
    down_df = pd.concat([df] + down_dfs, axis=0, ignore_index=True)
    
    return down_df

def compute_melspec(y, sr, n_mels, fmin, fmax):
    """
    Computes a mel-spectrogram and puts it at decibel scale
    Arguments:
        y {np array} -- signal
        params {AudioParams} -- Parameters to use for the spectrogram. Expected to have the attributes sr, n_mels, f_min, f_max
    Returns:
        np array -- Mel-spectrogram
    """
    melspec = lb.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, fmin=fmin, fmax=fmax,n_fft = CFG.nfft
    )

    melspec = lb.power_to_db(melspec, ref=1.0).astype(np.float32)
    return melspec

def mono_to_color(X, eps=1e-6, mean=None, std=None):
    mean = mean or X.mean()
    std = std or X.std()
    X = (X - mean) / (std + eps)
    
    _min, _max = X.min(), X.max()

    if (_max - _min) > eps:
        V = np.clip(X, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
        
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(X, dtype=np.uint8)

    return V

def torch_mono_to_color(X, eps=1e-6, mean=None, std=None):
    mean = mean or X.mean()
    std = std or X.std()
    X = (X - mean) / (std + eps)
    
    _min, _max = X.min(), X.max()
    if (_max - _min) > eps:
        V = torch.clamp(X, _min, _max)
        V = 255*(V - _min) / (_max - _min)
        V = V.type(torch.int64)
        # V = V.to(torch.uint8)
    else:
        V = torch.zeros_like(X)
    # V = V.float()
    return V

def crop_or_pad(y, length, is_train=True, start=None):
    if len(y) < length:
        y = np.concatenate([y, np.zeros(length - len(y))])
        
        n_repeats = length // len(y)
        epsilon = length % len(y)
        
        y = np.concatenate([y]*n_repeats + [y[:epsilon]])
        
    elif len(y) > length:
        if not is_train:
            start = start or 0
        else:
            start = start or np.random.randint(len(y) - length)

        y = y[start:start + length]

    return y

class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray, sr):
        for trns in self.transforms:
            y = trns(y, sr)
        return y


class AudioTransform:
    def __init__(self, always_apply=False, p=0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, y: np.ndarray, sr):
        if self.always_apply:
            return self.apply(y, sr=sr)
        else:
            if np.random.rand() < self.p:
                return self.apply(y, sr=sr)
            else:
                return y

    def apply(self, y: np.ndarray, **params):
        raise NotImplementedError


class OneOf(Compose):
    # https://github.com/albumentations-team/albumentations/blob/master/albumentations/core/composition.py
    def __init__(self, transforms, p=0.5):
        super().__init__(transforms)
        self.p = p
        transforms_ps = [t.p for t in transforms]
        s = sum(transforms_ps)
        self.transforms_ps = [t / s for t in transforms_ps]

    def __call__(self, y: np.ndarray, sr):
        data = y
        if self.transforms_ps and (random.random() < self.p):
            random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))
            t = random_state.choice(self.transforms, p=self.transforms_ps)
            data = t(y, sr)
        return data


class Normalize(AudioTransform):
    def __init__(self, always_apply=False, p=1):
        super().__init__(always_apply, p)

    def apply(self, y: np.ndarray, **params):
        max_vol = np.abs(y).max()
        y_vol = y * 1 / max_vol
        return np.asfortranarray(y_vol)


class NewNormalize(AudioTransform):
    def __init__(self, always_apply=False, p=1):
        super().__init__(always_apply, p)

    def apply(self, y: np.ndarray, **params):
        y_mm = y - y.mean()
        return y_mm / y_mm.abs().max()


class NoiseInjection(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_noise_level=0.5):
        super().__init__(always_apply, p)

        self.noise_level = (0.0, max_noise_level)

    def apply(self, y: np.ndarray, **params):
        noise_level = np.random.uniform(*self.noise_level)
        noise = np.random.randn(len(y))
        augmented = (y + noise * noise_level).astype(y.dtype)
        return augmented


class GaussianNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5, max_snr=20):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        white_noise = np.random.randn(len(y))
        a_white = np.sqrt(white_noise ** 2).max()
        augmented = (y + white_noise * 1 / a_white * a_noise).astype(y.dtype)
        return augmented


class PinkNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5, max_snr=20):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        pink_noise = cn.powerlaw_psd_gaussian(1, len(y))
        a_pink = np.sqrt(pink_noise ** 2).max()
        augmented = (y + pink_noise * 1 / a_pink * a_noise).astype(y.dtype)
        return augmented

def _db2float(db: float, amplitude=True):
    if amplitude:
        return 10 ** (db / 20)
    else:
        return 10 ** (db / 10)


def volume_down(y: np.ndarray, db: float):
    """
    Low level API for decreasing the volume
    Parameters
    ----------
    y: numpy.ndarray
        stereo / monaural input audio
    db: float
        how much decibel to decrease
    Returns
    -------
    applied: numpy.ndarray
        audio with decreased volume
    """
    applied = y * _db2float(-db)
    return applied


def volume_up(y: np.ndarray, db: float):
    """
    Low level API for increasing the volume
    Parameters
    ----------
    y: numpy.ndarray
        stereo / monaural input audio
    db: float
        how much decibel to increase
    Returns
    -------
    applied: numpy.ndarray
        audio with increased volume
    """
    applied = y * _db2float(db)
    return applied


class RandomVolume(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, limit=10):
        super().__init__(always_apply, p)
        self.limit = limit

    def apply(self, y: np.ndarray, **params):
        db = np.random.uniform(-self.limit, self.limit)
        if db >= 0:
            return volume_up(y, db)
        else:
            return volume_down(y, db)


class CosineVolume(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, limit=10):
        super().__init__(always_apply, p)
        self.limit = limit

    def apply(self, y: np.ndarray, **params):
        db = np.random.uniform(-self.limit, self.limit)
        cosine = np.cos(np.arange(len(y)) / len(y) * np.pi * 2)
        dbs = _db2float(cosine * db)
        return y * dbs

wave_transforms = {
    "train":
        Compose([
            # NoiseInjection(max_noise_level = 0.04, p=0.5),
            # AddGaussianNoise(p=0.2),
            # AddGaussianSNR(p=0.2),
            # Gain(min_gain_in_db=-15,max_gain_in_db=15,p=0.3),
            # PinkNoise(p=0.5),
            # NoiseInjection(p=0.5),
            # GaussianNoise(p=0.5),
            # PinkNoise(p=0.5),
            # TanhDistortion(p=0.5),
            # Gain(min_gain_in_db=-15, max_gain_in_db=15, p=0.8),
            # GainTransition(min_gain_in_db=-15, max_gain_in_db=15, p=0.8),
            # Gain(min_gain_in_db=-12, max_gain_in_db=12, p=0.5), # DEFAULT
            # Shift(min_fraction=0.2, max_fraction=0.2, p=0.5),   # 15 + (15 * 0.2) = 18 input audio length 
            # LowPassFilter(min_cutoff_freq=100, max_cutoff_freq=10000, p=0.5), # possibly incorrect values
            # OneOf( [ NoiseInjection(p=0.5, max_noise_level=0.04), 
            #         GaussianNoise(p=0.5, min_snr=5, max_snr=20), 
            #         PinkNoise(p=0.5, min_snr=5, max_snr=20), 
            #         AddGaussianNoise(min_amplitude=0.0001, max_amplitude=0.03, p=0.5), 
            #         AddGaussianSNR(min_snr_in_db=5, max_snr_in_db=15, p=0.5)
            #         ],p=0.3), 
            OneOf([
                AddBackgroundNoise(sounds_path='birdclef2021-background-noise/aicrowd2020_noise_30sec/noise_30sec', min_snr_in_db=0, max_snr_in_db=2, p=0.5 ),
                AddBackgroundNoise(sounds_path='birdclef2021-background-noise/ff1010bird_nocall', min_snr_in_db=0, max_snr_in_db=2, p=0.5 ),
                AddBackgroundNoise(sounds_path='birdclef2021-background-noise/train_soundscapes', min_snr_in_db=0, max_snr_in_db=2, p=0.5 )], p=0.3)
                # Normalize(p=1),
            # NoiseInjection(p=1, max_noise_level=0.04),
            # GaussianNoise(p=1, min_snr=5, max_snr=20),
            # PinkNoise(p=1, min_snr=5, max_snr=20),
            # no random volume ict 4
            # RandomVolume(p=0.2, limit=4),
            # TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            # PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
            # Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
            # Normalize(p=1),
        ])
    ,
    "valid": Compose([
        # Normalize(p=1),
        ])
}
mean = (0.485) # R only for RGB
std = (0.229) # R only for RGB
img_transforms = {
    'train' : A.Compose([
            A.Normalize(mean, std),
    ], p=1.0),
    'valid' : A.Compose([
            A.Normalize(mean, std),
    ], p=1.0),
}

class AudioDataset(Dataset):
    def __init__(self, val,df, duration, sr, wave_transforms, img_transforms=None):
        self.df = df
        self.duration = duration
        self.sr = sr
        self.audio_length = self.duration*self.sr
        self.wave_transforms = wave_transforms
        self.img_transforms = img_transforms
        self.val = val
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        primary_label = row['primary_label']
        audio, orig_sr = sf.read(row.filepath, always_2d=True, dtype="float32")
        audio = np.mean(audio, 1)
        train_duration = 5
        if self.val==False:
            
            if len(audio) < CFG.sample_rate*train_duration:
                audio = audio[:CFG.sample_rate*train_duration]
                audio = crop_or_pad(audio , length=CFG.sample_rate*train_duration)

            else:
                max_start_index = len(audio) - CFG.sample_rate*train_duration
                start_index = np.random.randint(max_start_index)
                audio = audio[start_index:start_index+CFG.sample_rate*train_duration]
        else:
            # audio, orig_sr = sf.read(row.filepath, always_2d=True, dtype="float32")
            # audio = np.mean(audio, 1)
            # if len(audio) < CFG.sample_rate*5:
            audio = audio[:CFG.sample_rate*5]
            audio = crop_or_pad(audio , length=CFG.sample_rate*5)

            # else:
            #     max_start_index = len(audio) - CFG.sample_rate*5
            #     start_index = np.random.randint(max_start_index)
            #     audio = audio[start_index:start_index+CFG.sample_rate*5]
        # print("Execution time:", time.time()- start_time, "seconds")
        audio = self.wave_transforms(audio, sr=self.sr)
        image = crop_or_pad(audio, length = self.audio_length)
        waveform=torch.Tensor(image)
        torchaudio_melspec = T.MelSpectrogram(
            sample_rate=CFG.sample_rate,
            n_fft=CFG.nfft,
            win_length=None,
            hop_length=CFG.hop_length,
            center=True,
            pad_mode="constant",
            power=2.0,
            norm='slaney',
            mel_scale='slaney',
            n_mels=CFG.n_mels,
            f_min = CFG.fmin,
            f_max = CFG.fmax
        )(waveform)
        torchaudio_melspec = T.AmplitudeToDB(stype="power",top_db=80)(torchaudio_melspec)
        image = mono_to_color(torchaudio_melspec.numpy())
        image = image.astype(np.uint8)
        image = self.img_transforms(image=image)['image']
        image = torch.tensor(image).float()
        if CFG.num_channels==3:
            image = torch.stack([image, image, image])
        else:
            image = torch.stack([image])
        
        targets = np.zeros(CFG.num_classes, dtype=float)
        targets[CFG.name2label[primary_label]]=1.0
        labels = torch.tensor(row['target'])
        return image, targets, labels
    
    def __len__(self):
        return len(self.df)

    
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        ret = gem(x, p=self.p, eps=self.eps)
        return ret

    def __repr__(self):
        return (
                self.__class__.__name__
                + "("
                + "p="
                + "{:.4f}".format(self.p.data.tolist()[0])
                + ", "
                + "eps="
                + str(self.eps)
                + ")"
        )
    
class TimmClassifier(nn.Module):
    def __init__(self, base_model_name, pretrained=True, num_classes=CFG.num_classes, in_channels=CFG.num_channels):
        super().__init__()

        self.bn0 = nn.BatchNorm2d(CFG.n_mels)
        if CFG.use_drop_path:
            self.base_model = timm.create_model(
                base_model_name, pretrained=pretrained, in_chans=in_channels, drop_rate=0.5, drop_path_rate = 0.2)
        else:
            self.base_model = timm.create_model(
                base_model_name, pretrained=pretrained, in_chans=in_channels)
        self.gem = GeM(p=3, eps=1e-6)
        if 'efficientnet' in base_model_name:
            in_features = self.base_model.classifier.in_features
        elif 'nfnet' in base_model_name:
            in_features = 2304
        elif 'convnext_base' in base_model_name:
            in_features = 1024
        elif 'convnext_tiny' in base_model_name:
            in_features = 768
        self.head1 = nn.Linear(in_features, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.base_model.forward_features(x)
        x = self.gem(x)
        x = x[:, :, 0, 0]
        logit = self.head1(x)

        output_dict = {
            'logit': logit,
        }

        return output_dict
class FineTuneTimmClassifier(nn.Module):
    def __init__(self, base_model_name = CFG.model_name, num_classes=CFG.num_classes):
        super().__init__()
        self.backbone = TimmClassifier(base_model_name=base_model_name, num_classes = 572, pretrained=False)
        # checkpoint = torch.load('pretrained_tf_efficientnet_b1_ns_fold_0_model_epoch_96_f1_0.7765.pth')
        # self.backbone.load_state_dict(checkpoint['state_dict'])
        if 'v2_b0' in base_model_name or 'v2_b1' in base_model_name or 'v2_s' in base_model_name:
            in_features = 1280
        elif 'v2_b2' in base_model_name:
            in_features = 1408
        elif 'nfnet' in base_model_name:
            in_features = 2304
        self.backbone.head1 = nn.Linear(in_features, num_classes)

    
    def forward(self, x):
        output_dict = self.backbone(x)
        return output_dict
    
class BCEBirdLossCalculator(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, outputs, targets):
        input = outputs["logit"]
        target = targets.float()
        loss = self.loss(input, target)
        return loss.sum(dim=1).mean()

class BCEFocal2WayLoss(nn.Module):
    def __init__(self, weights=[1, 1], class_weights=None):
        super().__init__()

        # self.focal = BCEFocalLoss()
        self.ce = nn.CrossEntropyLoss()
        self.weights = weights

    def forward(self, input, target):
        input_ = input["logit"]
        target = target.float()

        framewise_output = input["framewise_logit"]
        clipwise_output_with_max, _ = framewise_output.max(dim=1)
        loss = self.ce(input_, target)
        
        aux_loss = self.ce(clipwise_output_with_max, target)
        return self.weights[0] * loss + self.weights[1] * aux_loss

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))

def valid_fn(val_dataloader, model, criterion, device):
    losses = AverageMeter()
    model.eval()
    truth = []
    valid_predictions = []
    valid_labels = []
    valid_targets = []
    start = end = time.time()
    pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc='Val')
    for step, (images, targets, labels) in pbar:
        images = images.to(device, dtype=torch.float)
        targets = targets.to(device)
        # labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.no_grad():
            outputs = model(images)
            loss=criterion(outputs, targets)
        # valid_labels.append(labels.cpu().numpy())
        valid_targets.append(targets.cpu().numpy())
        losses.update(loss.item(), batch_size)
#         print(outputs)
        valid_predictions.append(F.softmax(outputs['logit'], dim=1).to('cpu').numpy())
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        pbar.set_postfix(eval_loss=f'{losses.avg:0.4f}',
                        gpu_mem=f'{mem:0.2f} GB')
        torch.cuda.empty_cache()
        gc.collect()
    # predictions = np.concatenate(preds)
    # valid_labels = np.concatenate(valid_labels)
    # valid_targets = np.concatenate(valid_targets)
    gc.collect()
    return losses.avg, valid_predictions, valid_targets

 
def padded_cmap(solution, submission, padding_factor=5):
    solution = solution#.drop(['row_id'], axis=1, errors='ignore')
    submission = submission#.drop(['row_id'], axis=1, errors='ignore')
    new_rows = []
    for i in range(padding_factor):
        new_rows.append([1 for i in range(len(solution.columns))])
    new_rows = pd.DataFrame(new_rows)
    new_rows.columns = solution.columns
    padded_solution = pd.concat([solution, new_rows]).reset_index(drop=True).copy()
    padded_submission = pd.concat([submission, new_rows]).reset_index(drop=True).copy()
    score = sklearn.metrics.average_precision_score(
        padded_solution.values,
        padded_submission.values,
        average='macro',
    )
    return score

class TestDataset(Dataset):
    def __init__(self, 
                 df: pd.DataFrame, 
                 clip: np.ndarray
                ):
        
        self.df = df
        self.clip = clip
        

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):

        sample = self.df.loc[idx, :]
        row_id = sample.row_id

        end_seconds = int(sample.seconds)
        start_seconds = int(end_seconds - 5)
        
        y = self.clip[CFG.sample_rate * start_seconds : CFG.sample_rate  * end_seconds].astype(np.float32)
        waveform=torch.Tensor(y)
        torchaudio_melspec = T.MelSpectrogram(
            sample_rate=CFG.sample_rate,
            n_fft=CFG.nfft,
            win_length=None,
            hop_length=CFG.hop_length,
            center=True,
            pad_mode="constant",
            power=2.0,
            norm='slaney',
            mel_scale='slaney',
            n_mels=CFG.n_mels,
            f_min = CFG.fmin,
            f_max = CFG.fmax
        )(waveform)
        torchaudio_melspec = T.AmplitudeToDB(stype="power",top_db=80.00)(torchaudio_melspec)
        image = mono_to_color(torchaudio_melspec.numpy())
        image = image.astype(np.uint8)
        image = img_transforms['valid'](image=image)['image']
        image = np.stack([image])
        image = torch.tensor(image).float()
            
        return {
            "image": image,
            "row_id": row_id,
        }
    
def prediction_for_clip(audio_path, target, val_predictions, val_targets, model):
    # set_seed(CFG.seed)
    model.eval()
    clip, sr = lb.load(audio_path, sr=None, duration=120)
    # clip, sr = lb.load(audio_path, sr=32000)
    duration = int(lb.get_duration(clip, sr))
    name_ = "_".join(audio_path.split(".")[:-1])
    seconds = [i for i in range(5, duration, 5)]
    row_ids = [name_+f"_{second}" for second in seconds]
    test_df = pd.DataFrame({
        "row_id": row_ids,
        "seconds": seconds
    })
    dataset = TestDataset(
        df=test_df, 
        clip=clip,
    )
    loader = DataLoader(
        dataset,
        batch_size=128, 
        num_workers=0,
        drop_last=False,
        shuffle=False,
        pin_memory=True
    )
    targets = np.zeros((len(seconds),CFG.num_classes), dtype=float)
    targets[:,target]=1.0
    val_targets.append(targets)
    for data in loader:           
        
        image = data['image'].to(CFG.device)
        
        with torch.no_grad():
            output = model(image)
            output = F.softmax(output['logit'], dim=1).to('cpu').numpy()
            val_predictions.append(output)

def padded_cmap(solution, submission, padding_factor=5):
    solution = solution#.drop(['row_id'], axis=1, errors='ignore')
    submission = submission#.drop(['row_id'], axis=1, errors='ignore')
    new_rows = []
    for i in range(padding_factor):
        new_rows.append([1 for i in range(len(solution.columns))])
    new_rows = pd.DataFrame(new_rows)
    new_rows.columns = solution.columns
    padded_solution = pd.concat([solution, new_rows]).reset_index(drop=True).copy()
    padded_submission = pd.concat([submission, new_rows]).reset_index(drop=True).copy()
    score = sklearn.metrics.average_precision_score(
        padded_solution.values,
        padded_submission.values,
        average='macro',
    )
    return score

out_file = 'swa_ensemble_v2s_fold1_5.pth' 
iteration = [
    # 'finaltrainv2b1/finetune_tf_efficientnetv2_b1_fold_1_model_epoch_176_0.818317.pth',
    # 'finaltrainv2b1/finetune_tf_efficientnetv2_b1_fold_1_model_epoch_158_0.817900.pth',
    # 'finaltrainv2b1/finetune_tf_efficientnetv2_b1_fold_1_model_epoch_111_0.815124.pth',
    # 'finaltrainv2b1/finetune_tf_efficientnetv2_b1_fold_1_model_epoch_160_0.817094.pth',
    # 'finaltrainv2b1/finetune_tf_efficientnetv2_b1_fold_1_model_epoch_150_0.815493.pth',
    # 'finaltrainv2b1/finetune_tf_efficientnetv2_b1_fold_1_model_epoch_126_0.816729.pth'
    # 'trainv2b1/finetune_tf_efficientnetv2_b1_fold_1_model_epoch_166_0.8142.pth',
    # 'trainv2b1/finetune_tf_efficientnetv2_b1_fold_1_model_epoch_144_0.8139.pth',
    # 'finaltrainv2b2/finetune_tf_efficientnetv2_b2_fold_1_model_epoch_176_0.818862.pth',
    # 'finaltrainv2b2/finetune_tf_efficientnetv2_b2_fold_1_model_epoch_158_0.817907.pth',
    # 'finaltrainv2b2/finetune_tf_efficientnetv2_b2_fold_1_model_epoch_182_0.816055.pth',
    # 'finaltrainv2b2/finetune_tf_efficientnetv2_b2_fold_1_model_epoch_133_0.817837.pth'
    'finaltrainv2s/finetune_tf_efficientnetv2_s_fold_1_model_epoch_154_0.813581.pth',
    'finaltrainv2s/finetune_tf_efficientnetv2_s_fold_1_model_epoch_171_0.813307.pth'
]

best_metric = 0
def objective(trial):
    
    a1 = trial.suggest_uniform('a1', 0.4, 0.99)
    a2 = 1-a1
    # a2 = trial.suggest_uniform('a2', 0.0009, 1-a1-0.001)
    # a3 = trial.suggest_uniform('a3', 0.0009, 1-a1-a2-0.001)
    # a3 = 1-a1-a2
    # a4 = trial.suggest_loguniform('a4', 0.0009, 1-a1-a2-a3-0.001)
    # a4 = 1-a1-a2-a3
    # a5 = 1-a1-a2-a3-a4
    # a5 = trial.suggest_loguniform('a5', 0.00009, 1-a1-a2-a3-a4-0.001)
    # a6 = 1-a1-a2-a3-a4-a5
    # a3 = trial.suggest_loguniform('a3', 0.0009, 1-a1-a2-0.001)
    # a4 = 1-a1-a2-a3
    a1 = 0.5
    a2 = 0.5
    state_dict = None
    for i in iteration:
        f = i
        f = torch.load(f, map_location=lambda storage, loc: storage)
        if state_dict is None:
            LOGGER.info(f"none: {i}")
            LOGGER.info(f"a1: {a1}")
            state_dict = f['state_dict']
            key = list(f['state_dict'].keys())
            for k in key:
                state_dict[k] = f['state_dict'][k]*a1
        elif i=='finaltrainv2s/finetune_tf_efficientnetv2_s_fold_1_model_epoch_171_0.813307.pth':
            LOGGER.info(f"none: {i}")
            LOGGER.info(f"a2: {a2}")  
            key = list(f['state_dict'].keys())
            for k in key:
                state_dict[k] = state_dict[k] + a2*f['state_dict'][k]
        # elif i=='finaltrainv2s/finetune_tf_efficientnetv2_s_fold_1_model_epoch_171_0.813307.pth':
        #     LOGGER.info(f"none: {i}")
        #     LOGGER.info(f"a3: {a3}")     
        #     key = list(f['state_dict'].keys())
        #     for k in key:
        #         state_dict[k] = state_dict[k] + a3 *f['state_dict'][k]
        # elif i=='finaltrainv2b2/finetune_tf_efficientnetv2_b2_fold_1_model_epoch_133_0.817837.pth':
        #     LOGGER.info(f"none: {i}")
        #     LOGGER.info(f"a4: {a4}") 
        #     key = list(f['state_dict'].keys())
        #     for k in key:
        #         state_dict[k] = state_dict[k] + a4 *f['state_dict'][k]
        # elif i=='finaltrainv2b1/finetune_tf_efficientnetv2_b1_fold_1_model_epoch_150_0.815493.pth':
        #     LOGGER.info(f"none: {i}")
        #     LOGGER.info(f"a5: {a5}") 
        #     key = list(f['state_dict'].keys())
        #     for k in key:
        #         state_dict[k] = state_dict[k] + a5 *f['state_dict'][k]
        # elif i=='finaltrainv2b1/finetune_tf_efficientnetv2_b1_fold_1_model_epoch_111_0.815124.pth':
        #     LOGGER.info(f"none: {i}")
        #     LOGGER.info(f"a6: {a6}") 
        #     key = list(f['state_dict'].keys())
        #     for k in key:
        #         state_dict[k] = state_dict[k] + a6 *f['state_dict'][k]
    
        
                 
    
    
    
    # print(out_file)
    torch.save({'state_dict': state_dict}, out_file)

    model = FineTuneTimmClassifier(base_model_name=CFG.model_name).to(CFG.device)
    checkpoint = torch.load('swa_ensemble_v2s_fold1_5.pth')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    valid_df = df_23[df_23['fold']==1].reset_index(drop=True)
    all_audios = valid_df['filepath'].tolist()
    all_targets = valid_df['target'].tolist()
    val_predictions = []    
    val_targets = []
    start = time.time()
    
    for (audio_path, target) in zip(all_audios, all_targets):
        prediction_for_clip(audio_path, target, val_predictions, val_targets, model)
    LOGGER.info(f"time: {(time.time()-start):.4f}")
    val_predictions = np.vstack(val_predictions)
    val_targets = np.vstack(val_targets)
    val_cmap = padded_cmap(pd.DataFrame(val_targets), pd.DataFrame(val_predictions))
    
    if val_cmap>0.816:
        state = {'state_dict': model.state_dict()}
        path = f'swa_{CFG.model_name}_fold_{fold}_model_{val_cmap:.4f}.pth'
        torch.save(state, path)
        # best_metric = val_cmap
    
    LOGGER.info(f"Val cmap: {val_cmap:.7f}")
    # LOGGER.info(f"Best metric: {best_metric:.7f}")
    return val_cmap

if __name__ == '__main__':

    set_seed(CFG.seed)
    gc.collect()
    torch.cuda.empty_cache()
    for fold in [1]:
        LOGGER.info(f"Fold: {fold}")
        study = optuna.create_study(direction='maximize', sampler = TPESampler())
        study.optimize(func=objective, n_trials=200)
        LOGGER.info(study.best_params)
