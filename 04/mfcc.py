import sklearn
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

AUDIO_FILE = r"C:\Users\lavan\Downloads\archive\UrbanSound8K\UrbanSound8K\audio\fold1\7061-6-0-0.wav"

samples, sample_rate = librosa.load(AUDIO_FILE, sr=None)

mfcc = librosa.feature.mfcc(y=samples, sr=sample_rate, n_mfcc=20)
mfcc = sklearn.preprocessing.scale(mfcc, axis=1)

plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc, sr=sample_rate, x_axis='time')
plt.colorbar()
plt.title("MFCC")
plt.show()

print(f"MFCC type: {type(mfcc)}, shape: {mfcc.shape}")
