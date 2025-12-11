import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

audio_file = r"C:\Users\lavan\Downloads\archive\UrbanSound8K\UrbanSound8K\audio\fold1\7061-6-0-0.wav"

# Load audio
samples, sr = librosa.load(audio_file)

# Create Mel Spectrogram
mel_sgram = librosa.feature.melspectrogram(
    y=samples,
    sr=sr,
    n_fft=2048,
    hop_length=512,
    n_mels=128
)

# Convert to decibels
mel_sgram = librosa.power_to_db(mel_sgram, ref=np.max)

# Print type + shape
print(type(mel_sgram), mel_sgram.shape)

plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_sgram, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar()
plt.title("Mel Spectrogram")
plt.show()
