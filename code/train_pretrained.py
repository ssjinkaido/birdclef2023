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
    model_name = "tf_efficientnetv2_b0"
    # model_name = "tf_efficientnetv2_s_in21ft1k"
    train_bs = 32
    valid_bs = train_bs * 6
    num_fold = 5
    epoch_warm_up = 0
    total_epoch = 60
    learning_rate = 1e-3
    weight_decay = 0.01
    thr_upsample = 50
    mix_up = 0.2
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

def init_logger(log_file='train_2023_10.log'):
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
            OneOf( [ NoiseInjection(p=0.5, max_noise_level=0.04), 
                    GaussianNoise(p=0.5, min_snr=5, max_snr=20), 
                    PinkNoise(p=0.5, min_snr=5, max_snr=20), 
                    AddGaussianNoise(min_amplitude=0.0001, max_amplitude=0.03, p=0.5), 
                    AddGaussianSNR(min_snr_in_db=5, max_snr_in_db=15, p=0.5)
                    ],p=0.3), 
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
            
            if len(audio) <= CFG.sample_rate*train_duration:
                audio = audio[:CFG.sample_rate*train_duration]
                audio = crop_or_pad(audio , length=CFG.sample_rate*train_duration)

            else:
                max_start_index = len(audio) - CFG.sample_rate*train_duration
                start_index = np.random.randint(max_start_index)
                audio = audio[start_index:start_index+CFG.sample_rate*train_duration]
        else:
            
            if len(audio) <= self.audio_length:
                audio = audio[:self.audio_length]
                audio = crop_or_pad(audio , length=self.audio_length)

            else:
                max_start_index = len(audio) - self.audio_length
                start_index = np.random.randint(max_start_index)
                audio = audio[start_index:start_index+self.audio_length]
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
            f_max = CFG.fmax,
            # normalized=True
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
        
        targets = np.zeros(CFG.num_classes2, dtype=float)
        targets[CFG.name2label2[primary_label]]=1.0
        labels = torch.tensor(row['target'])
        return image, targets, labels
    
    def __len__(self):
        return len(self.df)
    
def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.0)


def init_weights(model):
    classname = model.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.xavier_uniform_(model.weight, gain=np.sqrt(2))
        model.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for weight in model.parameters():
            if len(weight.size()) > 1:
                nn.init.orghogonal_(weight.data)
    elif classname.find("Linear") != -1:
        model.weight.data.normal_(0, 0.01)
        model.bias.data.zero_()


def interpolate(x: torch.Tensor, ratio: int):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    output = F.interpolate(
        framewise_output.unsqueeze(1),
        size=(frames_num, framewise_output.size(2)),
        align_corners=True,
        mode="bilinear").squeeze(1)

    return output


class AttBlockV2(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation="linear"):
        super().__init__()

        self.activation = activation
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)



class TimmSED(nn.Module):
    def __init__(self, base_model_name, pretrained=True, num_classes=506, in_channels=CFG.num_channels):
        super().__init__()

        self.bn0 = nn.BatchNorm2d(CFG.n_mels)
        if CFG.use_drop_path:
            self.base_model = timm.create_model(
                base_model_name, pretrained=pretrained, in_chans=in_channels, drop_rate=0.5, drop_path_rate = 0.2)
        else:
            self.base_model = timm.create_model(
                base_model_name, pretrained=pretrained, in_chans=in_channels)
        # self.spec_augmenter = SpecAugmentation(time_drop_width=64//2, time_stripes_num=2,
        #                                        freq_drop_width=8//2, freq_stripes_num=2)
        if 'efficientnet' in base_model_name:
            in_features = self.base_model.classifier.in_features
        elif 'nfnet' in base_model_name:
            in_features = 2304

        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.att_block = AttBlockV2(
            in_features, num_classes, activation="sigmoid")

        self.init_weight()
        self.dropout = nn.Dropout(p=0.5)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        

    def forward(self, x):
        x = x.transpose(2, 3)
        frames_num = x.shape[2]
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        # if CFG.use_spec_augment:
        #     x = self.spec_augmenter(x)
        x = x.transpose(2, 3)
        x = self.base_model.forward_features(x)
        # Aggregate in frequency axis
        x = torch.mean(x, dim=2)
        # print("after mean shape: ", x.shape)
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = self.dropout(x)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = self.dropout(x)

        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        logit = torch.sum(norm_att * self.att_block.cla(x), dim=2)
        segmentwise_logit = self.att_block.cla(x).transpose(1, 2)
        segmentwise_output = segmentwise_output.transpose(1, 2)

        interpolate_ratio = frames_num // segmentwise_output.size(1)

        # Get framewise output
        framewise_output = interpolate(segmentwise_output,
                                       interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        framewise_logit = interpolate(segmentwise_logit, interpolate_ratio)
        framewise_logit = pad_framewise_output(framewise_logit, frames_num)

        output_dict = {
            'framewise_output': framewise_output,
            'clipwise_output': clipwise_output,
            'logit': logit,
            'framewise_logit': framewise_logit,
        }

        return output_dict
    
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
    def __init__(self, base_model_name, pretrained=True, num_classes=CFG.num_classes2, in_channels=CFG.num_channels):
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
        self.head1 = nn.Linear(in_features, num_classes)
        

    def forward(self, x):
        x = self.base_model.forward_features(x)
        x = self.gem(x)
        x = x[:, :, 0, 0]
        logit = self.head1(x)

        output_dict = {
            'logit': logit,
        }

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

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(data, targets, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    new_targets = [targets, shuffled_targets, lam]
    return data, new_targets

def mixup(data, targets, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    new_data = data * lam + shuffled_data * (1 - lam)
    new_targets = [targets, shuffled_targets, lam]
    return new_data, new_targets


def cutmix_criterion(preds, new_targets, criterion):
    targets1, targets2, lam = new_targets[0], new_targets[1], new_targets[2]
    return lam * criterion(preds, targets1) + (1 - lam) * criterion(preds, targets2)

def mixup_criterion(preds, new_targets, criterion):
    targets1, targets2, lam = new_targets[0], new_targets[1], new_targets[2]
    return lam * criterion(preds, targets1) + (1 - lam) * criterion(preds, targets2)

def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device):
    torch.cuda.empty_cache()
    gc.collect()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    model.train()
    start = end = time.time()
    truth = []
    pred = []
    global_step = 0
    scaler = GradScaler()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc='Train')
    for step, (images, targets, labels) in pbar:
        optimizer.zero_grad()
        data_time.update(time.time() - end)
        images = images.to(device, dtype=torch.float)
        targets = targets.to(device)
        # labels = labels.to(device)
        batch_size = labels.size(0)
        with autocast():
            outputs = model(images)
            loss=criterion(outputs, targets)
            # loss = criterion(outputs, labels)
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        global_step += 1
        scheduler.step()
            # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        torch.cuda.empty_cache()
        gc.collect()
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(train_loss=f'{losses.avg:0.4f}',
                        lr=f'{current_lr:0.7f}',
                        gpu_mem=f'{mem:0.2f} GB')

    return losses.avg


def train_mixup_cutmix_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device):
    torch.cuda.empty_cache()
    gc.collect()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    model.train()
    start = end = time.time()
    truth = []
    pred = []
    global_step = 0
    scaler = GradScaler()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc='Train')
    for step, (images, targets, labels) in pbar:
        optimizer.zero_grad()
        data_time.update(time.time() - end)
        images = images.to(device)
        targets = targets.to(device)
        # labels = labels.to(device)
        batch_size = labels.size(0)
        if np.random.rand()<0.5:
            images, new_targets = mixup(images, targets, CFG.mix_up)
            with autocast(enabled=True):
                outputs = model(images)
                loss = mixup_criterion(outputs, new_targets, criterion) 
        else:
            images, new_targets = cutmix(images, targets, CFG.mix_up)
            with autocast(enabled=True):
                outputs = model(images)
                loss = cutmix_criterion(outputs, new_targets, criterion)
                
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        global_step += 1
        scheduler.step()
            # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        torch.cuda.empty_cache()
        gc.collect()
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(train_loss=f'{losses.avg:0.4f}',
                        lr=f'{current_lr:0.7f}',
                        gpu_mem=f'{mem:0.2f} GB')

    return losses.avg

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

if __name__ == '__main__':

    set_seed(CFG.seed)
    gc.collect()
    torch.cuda.empty_cache()
    for fold in [0]:
        LOGGER.info(f"Fold: {fold}")
        # model = Model().to(device)
        train_df = df_pre[df_pre['fold']!=fold].reset_index(drop=True)
        valid_df = df_pre[df_pre['fold']==fold].reset_index(drop=True)
        train_df = upsample_data(train_df, thr=CFG.thr_upsample)
        LOGGER.info(len(train_df))
        train_dataset = AudioDataset(val=False, df = train_df, duration=CFG.duration, sr = CFG.sample_rate, wave_transforms = wave_transforms['train'], img_transforms=img_transforms['train'])

        train_loader = DataLoader(train_dataset, batch_size = CFG.train_bs,
                                    num_workers=20, shuffle=True, pin_memory=True, drop_last=True)
        
        valid_dataset = AudioDataset(val=True, df = valid_df, duration=CFG.duration, sr = CFG.sample_rate, wave_transforms = wave_transforms['valid'], img_transforms=img_transforms['valid'])

        valid_loader = DataLoader(valid_dataset, batch_size = CFG.valid_bs, 
                                    num_workers=20, shuffle=False, pin_memory=True, drop_last=False)
        
        LEN_DL_TRAIN = len(train_loader)
        best_metric = 0
        
        # model = FineTuneTimmSED().to(CFG.device)
        # model = TimmSED(base_model_name=CFG.model_name).to(CFG.device)
        model = TimmClassifier(base_model_name=CFG.model_name).to(CFG.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.learning_rate, weight_decay = CFG.weight_decay)  
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = LEN_DL_TRAIN*20)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = LEN_DL_TRAIN* CFG.epoch_warm_up, num_training_steps = LEN_DL_TRAIN*CFG.total_epoch)
        # criterion = nn.CrossEntropyLoss().to(CFG.device)
        criterion = BCEBirdLossCalculator().to(CFG.device)
        # criterion = BCEFocal2WayLoss().to(CFG.device)
        
        LOGGER.info(optimizer)
        for epoch in range(CFG.total_epoch):
            LOGGER.info(f"Epoch: {epoch+1}/{CFG.total_epoch}")
            # if epoch < 10:
            if CFG.train_with_mixup:
                loss_train = train_mixup_cutmix_fn(train_loader, model, criterion, optimizer, epoch, scheduler, CFG.device)
            else:
                loss_train = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, CFG.device)
            loss_valid, valid_predictions, valid_targets = valid_fn(valid_loader, model, criterion, CFG.device)
            LOGGER.info(f"Train loss: {loss_train:.4f}, val loss: {loss_valid:.4f}")
            TARGETS = np.vstack(valid_targets)
            PREDS = np.vstack(valid_predictions)
        #     print(LABELS.shape, PREDS.shape)
        #     print(LABELS, PREDS)
            val_cmap = padded_cmap(pd.DataFrame(TARGETS), pd.DataFrame(PREDS))
            LOGGER.info(val_cmap)
            if val_cmap>best_metric and val_cmap>0.83:
                LOGGER.info(f"Model improve: {best_metric:.4f} -> {val_cmap:.4f}")
                best_metric = val_cmap
                state = {'epoch': epoch+1, 'state_dict': model.state_dict()}
                path = f'trainv2b0/{CFG.model_name}_fold_{fold}_model_epoch_{epoch+1}_{val_cmap:.4f}.pth'
                torch.save(state, path)
            if val_cmap>best_metric:
                LOGGER.info(f"Model improve: {best_metric:.4f} -> {val_cmap:.4f}")
                best_metric = val_cmap