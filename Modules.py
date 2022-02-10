from argparse import Namespace
import enum
import math
from unicodedata import bidirectional
from matplotlib.style import use
import torch
from typing import List

from Layer import Conv1d, Conv2d, Lambda, Linear, AdaIN2d
from librosa.filters import mel as librosa_mel_fn

class Light_SERNet(torch.nn.Module):
    def __init__(self, hyper_parameters: Namespace):
        super().__init__()
        self.hp = hyper_parameters

        self.parallel_paths = torch.nn.ModuleList([
            Block(
                in_channels= 1,
                out_channels= channels,
                kernel_size= kernel_size,
                pool_size= pool_size
                )
            for channels, kernel_size, pool_size in zip(
                self.hp.Light_SERNet.Parallel_Path.Channels,
                self.hp.Light_SERNet.Parallel_Path.Kernel_Size,
                self.hp.Light_SERNet.Parallel_Path.Pool_Size,
                )
            ])

        previous_channels = sum(self.hp.Light_SERNet.Parallel_Path.Channels)
        self.lflb = torch.nn.Sequential()
        for index, (channels, kernel_size, pool_size) in enumerate(zip(
            self.hp.Light_SERNet.LFLB.Channels,
            self.hp.Light_SERNet.LFLB.Kernel_Size,
            self.hp.Light_SERNet.LFLB.Pool_Size,
            )):
            self.lflb.add_module('Block_{}'.format(index), Block(
                in_channels= previous_channels,
                out_channels= channels,
                kernel_size= kernel_size,
                pool_size= pool_size
                ))
            previous_channels = channels

        self.classifier = torch.nn.Sequential(
            Block(
                in_channels= previous_channels,
                out_channels= self.hp.Light_SERNet.Postnet.Channels,
                kernel_size= [1, 1],
                pool_size= None
                ),
            Lambda(lambda x: x.mean(dim= (2, 3))),
            torch.nn.Dropout(p= 0.3),
            torch.nn.Linear(
                in_features= self.hp.Light_SERNet.Postnet.Channels,
                out_features= self.hp.Emotions
                )
            )

    def forward(self, features: torch.Tensor):
        x = features.unsqueeze(1)   # [Batch, 1, Feature_d, Feature_t]
        x = torch.cat(
            [path(x) for path in self.parallel_paths],
            dim= 1
            )   # [Batch, Path_d * 3, Feature_d / 2, Feature_t / 2]
        x = self.lflb(x)
        x = self.classifier(x)

        return x


class Block(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: List[int],
        pool_size: List[int] = None
        ):
        super().__init__()

        self.add_module('Conv', Conv2d(
            in_channels= in_channels,
            out_channels= out_channels,
            kernel_size= kernel_size,
            padding= [
                (kernel_size[0] - 1) // 2,
                (kernel_size[1] - 1) // 2,
                ],
            w_init_gain= 'relu'
            ))
        self.add_module('BatchNorm', torch.nn.BatchNorm2d(
            num_features= out_channels
            ))
        self.add_module('ReLU', torch.nn.ReLU())

        if not pool_size is None:
            self.add_module('AvgPool', torch.nn.AvgPool2d(
                kernel_size= pool_size
                ))
        
    def forward(self, x: torch.Tensor):
        return super().forward(x)

class Mel_Generator(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        num_mels: int,
        window_size: int,
        hop_size: int,
        fmin: int,
        fmax: int
        ):
        super().__init__()
        self.mel_basis = torch.from_numpy(librosa_mel_fn(sample_rate, n_fft, num_mels, fmin, fmax))        
        self.hann_window = torch.hann_window(window_size)

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.window_size = window_size
        self.hop_size = hop_size
        self.fmin = fmin
        self.fmax = fmax

    def forward(self, audios):
        if audios.ndim == 1:
            audios = audios.unsqueeze(0)

        audios = torch.nn.functional.pad(
            audios.unsqueeze(1),
            [int((self.n_fft - self.hop_size) / 2), int((self.n_fft - self.hop_size) / 2)],
            mode='reflect'
            ).squeeze(1)
        spectrograms = torch.stft(
            audios,
            n_fft= self.n_fft,
            hop_length= self.hop_size,
            win_length= self.window_size,
            window= self.hann_window,
            center= False,
            pad_mode= 'reflect',
            normalized= False,
            onesided= True
            )
        spectrograms = torch.sqrt(spectrograms.pow(2).sum(-1)+(1e-9))

        mels = torch.matmul(self.mel_basis, spectrograms)
        mels = torch.log(torch.clamp(mels, min=1e-5))
        mels = (mels + 11.5129) / (2.0957 + 11.5129) * 2.0 - 1.0

        return mels


def Mask_Generate(lengths: torch.Tensor, max_lengths: int= None):
    '''
    lengths: [Batch]
    max_lengths: an int value. If None, max_lengths == max(lengths)
    '''
    max_lengths = max_lengths or torch.max(lengths)
    sequence = torch.arange(max_lengths)[None, :].to(lengths.device)
    return sequence >= lengths[:, None]    # [Batch, Time]
