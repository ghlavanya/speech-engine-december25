import os

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

audio_path = r"C:\Users\lavan\OneDrive\Desktop\dev\spectograms\0_jackson_21.wav"

#load audio file
signal, sr = librosa.load(audio_path, sr=None)

print(f"Signal shape: {signal.shape}, Sample rate: {sr}")

# ---------- 3. Linear-frequency spectrogram (baseline) ----------
n_fft = 1024
hop_length = 512

stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
stft_mag = np.abs(stft)
stft_db = librosa.amplitude_to_db(stft_mag, ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(
    stft_db,
    sr=sr,
    hop_length=hop_length,
    x_axis="time",
    y_axis="log",
)
plt.colorbar(format="%+2.0f dB")
plt.title("Linear-frequency Spectrogram (log-freq, dB)")
plt.tight_layout()
plt.show()

# ---------- 4. Mel spectrogram in db----------

mel_spec = librosa.feature.melspectrogram(
    y=signal,
    sr=sr,
    n_fft=n_fft,
    hop_length=hop_length,
    n_mels=128,
)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(
    mel_spec_db,
    sr=sr,
    hop_length=hop_length,
    x_axis="time",
    y_axis="mel",
)
plt.colorbar(format="%+2.0f dB")
plt.title("Mel Spectrogram (dB)")
plt.tight_layout()
plt.show()