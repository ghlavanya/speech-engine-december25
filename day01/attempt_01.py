print("script started")

import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

audio_path = r"C:\Users\lavan\OneDrive\Desktop\dev\spectograms\0_jackson_21.wav"

print("cheching file exists at:",audio_path)
print("exists?:",os.path.exists(audio_path))

#load audio file
signal, sr = librosa.load(audio_path, sr=None)

print("Signal shape:",signal.shape)
print("Sample rate:",sr)

#------------plot waveform ------------
plt.figure(figsize=(10, 4))
librosa.display.waveshow(signal, sr=sr)
plt.title("Waveform")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()

#------------plot spectrogram ------------

#step1: stft- cut signal into frames and perform FFT on each
n_fft = 1024  #window size (no. of sameples/ frame)
hop_length = 512  #how muchwe move the window each step

stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
spectrogram = np.abs(stft)

#step2: convert amplitude to dB
spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title("Spectrogram")
plt.tight_layout()
plt.show()