import pandas as pd
from pathlib import Path

data_path = Path(r"C:\Users\lavan\OneDrive\Desktop\dev\kaggle\recordings")

# List all audio files in the folder
audio_files = list(data_path.glob("*.wav"))

# Make a dataframe
df = pd.DataFrame({
    "relative_path": [str(f) for f in audio_files],
    "classID": None  # You must fill this manually or from another mapping
})

# Save CSV (optional)
df.to_csv(data_path/"metadata.csv", index=False)

print(df.head())

import math, random
import torch
import torchaudio
from torchaudio import transforms
from IPython.display import Audio

class AudioUtil():
    # ----------------------------
    # Load an audio file and return signal + sample rate
    # ----------------------------
    @staticmethod
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return (sig, sr)

audio_path = df.iloc[0]['relative_path']
sig, sr = AudioUtil.open(audio_path)