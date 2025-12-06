print("script started")

import os
import librosa
import librosa.display
import matplotlib.pyplot as plt

audio_path = r"C:\Users\lavan\OneDrive\Desktop\dev\spectograms\0_jackson_21.wav"

print("cheching file exists at:",audio_path)
print("exists?:",os.path.exists(audio_path))

#load audio file
signal, sr = librosa.load(audio_path, sr=None)

print("Signal shape:",signal.shape)
print("Sample rate:",sr)

#plot waveform
plt.figure(figsize=(10, 4))
librosa.display.waveshow(signal, sr=sr)
plt.title("Waveform")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()