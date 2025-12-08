from scipy.io import wavfile
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

audio_path = r"C:\Users\lavan\OneDrive\Desktop\dev\spectograms\0_jackson_21.wav"

sr, samples = wavfile.read(audio_path)

print("SciPy sample rate:", sr)
print("Raw samples dtype:", samples.dtype)
print("First 10 samples:", samples[:10])

# Normalize if needed
if samples.dtype != np.float32:
    samples = samples.astype(np.float32) / np.max(np.abs(samples))
    print("Normalized samples dtype:", samples.dtype)


#from playsound import playsound
#playsound(audio_path)

import sounddevice as sd

sd.play(samples, sr)
sd.wait()


print ('Example shape ', samples.shape, 'Sample rate ', sr, 'Data type', type(samples))
print (samples[22400:22420])

sgram = librosa.stft(samples)
librosa.display.specshow(sgram)

sgram_mag, _ = librosa.magphase(sgram)

mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sr)

mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
librosa.display.specshow(mel_sgram, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
