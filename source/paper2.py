import librosa
import numpy as np
import os
import sys
import io
import librosa
import librosa.display
import IPython.display
from librosa.feature import melspectrogram

from scipy.fftpack import fft
import matplotlib.pylab as plt

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')

y,sr=librosa.load('C:/project1/train/audio/ZOOM0021 .wav')
IPython.display.Audio(data=y,rate=sr)

D = librosa.amplitude_to_db(librosa.stft(y[:1024]),ref = np.max)
plt.plot(D.flatten())
plt.show()


S = librosa.feature.melspectrogram(y,sr=sr,n_mels=128)

log_S = librosa.power_to_db(S, ref=np.max)


plt.figure(figsize=(12,4))
librosa.display.specshow(log_S,sr=sr,x_axis='time',y_axis='mel')
plt.title('mel power spectrogram')
plt.colorbar(format='%+02.0f dB')
plt.tight_layout()
plt.show()
