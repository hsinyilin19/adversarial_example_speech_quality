import os
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa
import soundfile as sf
from utils import *


INPUT_LENGTH = 9.01

class AudioData(Dataset):
    def __init__(self, input_files, noisy_paths, clean_paths, max_audio_length, N=None, test=False):
        super(AudioData, self).__init__()
        self.noisy_paths, self.clean_paths = noisy_paths, clean_paths
        self.input_files = input_files[:N] if N else input_files
        self.max_audio_length = max_audio_length
        self.test = test

    def load_wav(self, file):
        wav, sr = torchaudio.load(file)

        # for CNN training, need to crop to same length for batch
        if not self.test:
            if wav.shape[-1] <= self.max_audio_length:
                cycle = self.max_audio_length // wav.shape[-1] + 1
                wav = wav.repeat(1, cycle)

            random_start = torch.randint(0, (wav.shape[1] - self.max_audio_length), (1,))
            wav = wav[..., random_start: (random_start + self.max_audio_length)]

        return wav, sr

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        # load noisy audio
        noisy_file = os.path.join(self.noisy_paths[idx], self.input_files[idx])
        noisy_wav, sr = self.load_wav(noisy_file)

        # load clean audio
        clean_file = os.path.join(self.clean_paths[idx], self.input_files[idx])
        clean_wav, sr = self.load_wav(clean_file)

        noisy_wav = noisy_wav.squeeze(dim=0)
        clean_wav = clean_wav.squeeze(dim=0)

        return noisy_wav, clean_wav, self.input_files[idx]



class Perturbation_Data(Dataset):
    def __init__(self, input_files, audio_paths, perturb_path, max_audio_length, eval_metric, N=None, test=False, normalize_input=False):
        super(Perturbation_Data, self).__init__()
        self.audio_paths = audio_paths
        self.perturb_path = perturb_path
        self.input_files = input_files[:N] if N else input_files
        self.max_audio_length = max_audio_length
        self.eval_metric = eval_metric.cpu()   # put network on cpu to match data device
        self.test = test
        self.normalize_input=normalize_input

    def load_wav(self, file, δ):
        wav, sr = torchaudio.load(file)
        if self.normalize_input:
            wav = wav / torch.max(torch.abs(wav))

        wav_attacked = wav + δ  # X + δ
        aud_len = wav.shape[-1]

        # for CNN training, need to crop to same length for batch
        if aud_len <= self.max_audio_length:
            cycle = self.max_audio_length // aud_len + 1
            wav = wav.repeat(1, cycle)
            wav_attacked = wav_attacked.repeat(1, cycle)

        random_start = torch.randint(0, (wav.shape[-1] - self.max_audio_length), (1,)) if not self.test else 0
        wav = wav[..., random_start: (random_start + self.max_audio_length)]
        wav_attacked = wav_attacked[..., random_start: (random_start + self.max_audio_length)]

        return wav, wav_attacked, sr

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        # load audio & corresponding perturbation
        audio_file = os.path.join(self.audio_paths[idx], self.input_files[idx])
        perturb_file = os.path.join(self.perturb_path, f'δ_{os.path.splitext(self.input_files[idx])[0]}.pt')

        # load data
        δ = torch.load(perturb_file)   # perturbation
        X, X_attacked, sr = self.load_wav(audio_file, δ)

        self.eval_metric.eval()
        with torch.no_grad():
            y = self.eval_metric(X)  # scores

        return X.squeeze(), X_attacked.squeeze(), y.squeeze(), self.input_files[idx]
