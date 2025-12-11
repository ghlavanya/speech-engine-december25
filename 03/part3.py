

import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

audio_path = r"C:\Users\lavan\OneDrive\Desktop\dev\spectograms\0_jackson_21.wav"

# Load audio file
try:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    signal, sr = librosa.load(audio_path, sr=None)
    
    print("Loaded:", audio_path)
    print("Sample rate:", sr, " | Length:", len(signal))
except FileNotFoundError as e:
    print(f"ERROR: {e}")
    print("Please check that the audio file exists at the specified path.")
    exit(1)
except Exception as e:
    print(f"ERROR loading audio file: {type(e).__name__}: {e}")
    exit(1)

# --------------------------
# 1. ADD GAUSSIAN NOISE
# --------------------------

def add_noise(audio, noise_factor=0.02):
    noise = np.random.randn(len(audio))
    augmented = audio + noise_factor * noise
    return augmented

try:
    noisy = add_noise(signal, noise_factor=0.02)
    
    plt.figure(figsize=(12, 3))
    plt.title("Noisy Audio Waveform")
    plt.plot(noisy)
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"ERROR in noise addition: {type(e).__name__}: {e}")

# --------------------------
# 2. TIME STRETCHING
# --------------------------

def apply_time_stretch(audio, rate=0.8):
    return librosa.effects.time_stretch(audio, rate)

try:
    stretched = apply_time_stretch(signal, rate=0.8)
    
    plt.figure(figsize=(12, 3))
    plt.title("Time-Stretched Audio (rate=0.8)")
    plt.plot(stretched)
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"ERROR in time stretching: {type(e).__name__}: {e}")

# --------------------------
# 3. PITCH SHIFTING
# --------------------------

def pitch_shift(audio, sr, n_steps=2):
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

try:
    shifted = pitch_shift(signal, sr, n_steps=2)
    
    plt.figure(figsize=(12, 3))
    plt.title("Pitch Shifted Audio (+2 steps)")
    plt.plot(shifted)
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"ERROR in pitch shifting: {type(e).__name__}: {e}")
