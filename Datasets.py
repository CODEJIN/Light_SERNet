from re import S
import torch
import math
import numpy as np
import pickle, os, logging, librosa
from typing import Dict, List, Tuple
from random import choice
from multipledispatch import dispatch
import matplotlib.pyplot as plt
import time

from meldataset import mel_spectrogram

@dispatch(tuple)
def Feature_Stack(features: Tuple[np.array]):
    max_feature_length = max([feature.shape[0] for feature in features])
    features = np.stack(
        [
            np.pad(
                (feature + 11.5129) / (2.0957 + 11.5129) * 2.0 - 1.0,
                [[0, max_feature_length - feature.shape[0]], [0, 0]]
                )
            for feature in features
            ],
        axis= 0
        )
    return features

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        emotion_info_dict: Dict[str, int],
        pattern_path: str,
        metadata_file: str,
        feature_length_min: int,
        feature_length_max: int,
        accumulated_dataset_epoch: int= 1,
        augmentation_ratio: float= 0.0
        ):
        super().__init__()
        self.emotion_info_dict = emotion_info_dict
        self.pattern_path = pattern_path
        
        metadata_dict = pickle.load(open(
            os.path.join(pattern_path, metadata_file).replace('\\', '/'), 'rb'
            ))
        
        self.patterns = []
        max_pattern_by_speaker = max([
            len(patterns)
            for patterns in metadata_dict['File_List_by_Speaker_Dict'].values()
            ])
        for patterns in metadata_dict['File_List_by_Speaker_Dict'].values():
            ratio = float(len(patterns)) / float(max_pattern_by_speaker)
            if ratio < augmentation_ratio:
                patterns *= int(np.ceil(augmentation_ratio / ratio))
            self.patterns.extend(patterns)

        # This is temporal to remove silence patterns.
        self.patterns = [
            pattern for pattern in self.patterns
            if all([
                not 'LMY04282' in pattern,
                not 'LMY07365' in pattern,
                metadata_dict['Mel_Length_Dict'][pattern] >= feature_length_min,
                metadata_dict['Mel_Length_Dict'][pattern] <= feature_length_max,
                ])
            ]
        self.patterns *= accumulated_dataset_epoch

    def __getitem__(self, idx):
        path = os.path.join(self.pattern_path, self.patterns[idx]).replace('\\', '/')
        pattern_dict = pickle.load(open(path, 'rb'))
        feature = pattern_dict['Mel']
        emotion = pattern_dict['Emotion']
        
        return feature, self.emotion_info_dict[emotion]

    def __len__(self):
        return len(self.patterns)

class Inference_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        n_fft: int,
        num_features: int,
        sample_rate: int,
        frame_length: int,
        frame_shift: int,
        fmin: int,
        fmax: int,
        source_paths: List[str],
        ):
        super().__init__()
        self.n_fft = n_fft
        self.num_features = num_features
        self.sample_rate = sample_rate
        self.window_size = frame_length
        self.hop_size = frame_shift
        self.fmin = fmin
        self.fmax = fmax

        self.patterns = []
        for index, source_path in enumerate(source_paths):
            if not os.path.exists(source_path):
                logging.warn('The path of line {} in \'{}\' is incorrect. This line is ignoired.'.format(index + 1, source_path))
                continue

            self.patterns.append((source_path))

    def __getitem__(self, idx):
        path = self.patterns[idx]
        audio, _ = librosa.load(path, sr= self.sample_rate)
        audio = librosa.effects.trim(audio, frame_length= 512, hop_length= 256)[0]
        audio = librosa.util.normalize(audio) * 0.95
        audio = torch.from_numpy(audio).unsqueeze(0)

        feature = mel_spectrogram(
            y= audio,
            n_fft= self.n_fft,
            num_mels= self.num_features,
            sampling_rate= self.sample_rate,
            hop_size= self.hop_size,
            win_size= self.window_size,
            fmin= self.fmin,
            fmax= self.fmax
            ).squeeze(0).permute(1, 0)

        return feature, path

    def __len__(self):
        return len(self.patterns)

class Collater:
    def __call__(self, batch):
        features, emotions = zip(*batch)
        
        features = Feature_Stack(features)
        emotions = np.array(emotions)

        features = torch.FloatTensor(features).permute(0, 2, 1)  # [Batch, Feature_d, Feature_t]
        emotions = torch.LongTensor(emotions)  # [Batch]
        
        return features, emotions
        
class Inference_Collater:
    def __call__(self, batch):
        features, paths = zip(*batch)
        
        features = Feature_Stack(features)
        
        features = torch.FloatTensor(features).permute(0, 2, 1)   # [Batch, Time]
        
        return features, paths