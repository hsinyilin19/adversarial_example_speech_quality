import os
import numpy as np
import torch
import librosa
import scipy
from tqdm import tqdm
import torchaudio
from torchaudio import functional as taF
from scipy import io
from pathlib import Path

class AverageMeter(object):
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


n_fft      = 512
win_length = 512
hop_length = 256
n_mels     = 40
n_mfcc     = 40
epsilon    = 1e-6
top_DB     = 80


def log1p(Y):
    pos = (Y > 0.)
    neg = (Y < 0.)
    Y[pos] = torch.log1p(Y[pos])
    Y[neg] = - torch.log1p(-Y[neg])
    return Y


def inv_log1p(Y):
    pos = (Y > 0.)
    neg = (Y < 0.)
    Y[pos] = torch.expm1(Y[pos])
    Y[neg] = - torch.expm1(-Y[neg])
    return Y

class get_feature():
    def __init__(self, sr=16000):
        super().__init__()
        self.sr = sr
        self.torchmscale = torchaudio.transforms.MelScale(n_stft=n_fft // 2 + 1, n_mels=n_mels, sample_rate=sr)
        self.dct_mat = taF.create_dct(n_mfcc, n_mels, norm='ortho')
        self.amplitude_to_DB = torchaudio.transforms.AmplitudeToDB('power', top_DB)

    def __call__(self, wav, sr=16000, ftype='log1p', log=False, log1p_scaling=False):
        self.torchmscale = self.torchmscale.to(wav.device)
        self.dct_mat = self.dct_mat.to(wav.device)
        self.amplitude_to_DB = self.amplitude_to_DB.to(wav.device)

        if len(wav.shape) > 2:
            multi_channel = True
            B, C, L = wav.shape
            wav = wav.reshape(B * C, L)
        else:
            multi_channel = False

        length = wav.shape[-1]
        phase_list = ['spec', 'log1p']
        x_stft = torch.stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=True,
                            normalized=False, onesided=True, pad_mode='reflect', return_complex=False,
                            window=torch.hamming_window(win_length).to(wav.device))

        if ftype == 'complex':
            X = x_stft
        elif ftype == 'spec' or ftype == 'log1p':
            X = torch.norm(x_stft, dim=-1, p=2) + epsilon
            phase = x_stft / X.unsqueeze(-1)
            if ftype == 'log1p':
                X = torch.log1p(X)
        elif ftype == 'mel_spec' or ftype == 'mfcc':
            X = self.torchmscale(torch.view_as_complex(x_stft).abs().pow(2))
            if ftype == 'mfcc':
                X = self.amplitude_to_DB(X)
                X = torch.matmul(X.transpose(-2, -1), self.dct_mat).transpose(-2, -1)

        if multi_channel:
            _, D, T = X.shape[:3]  # D: feature dim
            if ftype == 'complex':
                X = X.reshape(B, C, D, T, 2).permute(0, 1, 4, 2, 3).reshape(B, -1, D, T)  # e.STOI_net. torch.Size([128, nMics, 2, 201, 202]), "2"= real + imaginary
            else:
                X = X.reshape(B, C, D, T)

        if log: X = X.log()

        if log1p_scaling: X = log1p(X)

        return X


def get_filepaths(directory, ftype='.wav', sort=True):
    file_paths = []
    file_names = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            if filename.endswith(ftype):
                # filepath = os.path.join(root, filename)
                file_names.append(filename)
                file_paths.append(root)

    file_paths = sorted(file_paths) if sort else file_paths
    file_names = sorted(file_names) if sort else file_names
    return file_paths, file_names


def make_spectrum(filename=None, y=None, feature_type='logmag'):
    if filename:
        y, sr = librosa.load(filename, sr=16000)
        if sr != 16000:
            raise ValueError('Sampling rate is expected to be 16kHz!')
        if y.dtype == 'int16':
            y = np.float32(y/32767.)
        elif y.dtype != 'float32':
            y = np.float32(y)

    D = librosa.stft(y, center=False, n_fft=512, hop_length=256, win_length=512, window=scipy.signal.hamming)
    phase = np.exp(1j * np.angle(D))
    D = np.abs(D)

    # convert feature type
    if feature_type == 'logmag':
        Sxx = np.log1p(D)
    else:
        Sxx = D
    return Sxx, phase, len(y)


def recons_spec_phase(Sxx_r, phase, length_wav, feature_type='logmag'):
    if feature_type == 'logmag':
        Sxx_r = np.expm1(Sxx_r)
        Sxx_r = np.clip(Sxx_r, a_min=0., a_max=None)

    R = np.multiply(Sxx_r, phase)
    result = librosa.istft(R, center=False, hop_length=256, win_length=512, window=scipy.signal.hamming, length=length_wav)
    return result


def check_path(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def check_folder(path):
    path_n = '/'.join(path.split('/')[:-1])
    check_path(path_n)




def normalize_volume(y):
    ''' normalize audio volume for EACH wav clip'''
    if len(y.shape) > 1:  # if batch dim exists
        N = y.shape[0]

        # for each audio sample
        for i in range(N):
            M = torch.abs(y[i]).max()
            y[i] = y[i] / M
    return y


def record_audio(input, filenames, scores, score_types, output_path, fs=16000):
    X = normalize_volume(input).squeeze().detach().cpu().numpy()
    record_type = list(score_types.keys())[0]

    if len(X.shape) > 1:  # if batch dim exists
        N = X.shape[0]

        with tqdm(total=N, desc=f'writing audio', unit='step') as t:
            for _ in range(N):
                # write scores into filenames
                write_score = f'{record_type}'
                for i in range(len(score_types[record_type])):
                    write_score += f'_{score_types[record_type][i]}_{scores[_, i]:.2f}'

                # write into wavs
                output_file = os.path.join(output_path, f'{write_score}_{filenames[_]}')
                scipy.io.wavfile.write(output_file, fs, X[_])

                # display progress
                t.set_description(output_file)
                t.update(1)


    else:
        # write scores into filenames
        write_score = f'{record_type}'
        for i in range(len(score_types[record_type])):
            write_score += f'_{score_types[record_type][i]}_{scores[0,i]:.2f}'

        # write into wavs
        output_file = os.path.join(output_path, f'{write_score}_{filenames[0]}')
        scipy.io.wavfile.write(output_file, fs, X)



