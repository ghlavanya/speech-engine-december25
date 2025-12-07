from scipy.io import wavfile
import numpy as np

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
