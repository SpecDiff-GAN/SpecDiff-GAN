# Adapted from https://github.com/jik876/hifi-gan under the MIT license.

import glob
import os
import matplotlib
import torch
import torchaudio
import numpy as np
from torch.nn.utils import weight_norm
matplotlib.use("Agg")
import matplotlib.pylab as plt


def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]

def ConcatNegativeFrequency(tensor):
  return torch.concat((tensor[..., :-1], tensor[..., 1:].flip(dims = (-1,))), -1)


def MinimumPhaseFilter(amplitude):
  rank = amplitude.ndim
  num_bins = amplitude.shape[-1]
  amplitude = ConcatNegativeFrequency(amplitude)

  fftsize = (num_bins - 1) * 2
  m0 = torch.zeros((fftsize // 2 - 1,), dtype=torch.complex64)
  m1 = torch.ones((1,), dtype=torch.complex64)
  m2 = torch.ones((fftsize // 2 - 1,), dtype=torch.complex64) * 2.0
  minimum_phase_window = torch.concat([m1, m2, m1, m0], axis=0)

  if rank > 1:
    new_shape = [1] * (rank - 1) + [fftsize]
    minimum_phase_window = torch.reshape(minimum_phase_window, new_shape)

  cepstrum = torch.fft.ifft(torch.log(amplitude).to(torch.complex64))
  windowed_cepstrum = cepstrum * minimum_phase_window
  imag_phase = torch.imag(torch.fft.fft(windowed_cepstrum))
  phase = torch.exp(torch.complex(imag_phase * 0.0, imag_phase))
  minimum_phase = amplitude.to(torch.complex64) * phase
  return minimum_phase[..., :num_bins]

def istft_M_stft(audio, M, n_fft, hop_size, win_size):
  Spec = torchaudio.transforms.Spectrogram(n_fft=n_fft, win_length=win_size, hop_length=hop_size, power=None, normalized=False, pad_mode='reflect',
                                           center = False, onesided=True).to(audio.device)
  iSpec = torchaudio.transforms.InverseSpectrogram(n_fft=n_fft, win_length=win_size, hop_length=hop_size, normalized=False, pad_mode='reflect',
                                                   center=True, onesided=True).to(audio.device)
  audio1 = torch.nn.functional.pad(audio.squeeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
  audio_new = M * Spec(audio1)
  audio_new = iSpec(audio_new, audio.shape[-1]).unsqueeze(1)
  return audio_new
