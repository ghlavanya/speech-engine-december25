import pandas as pd
from pathlib import Path

#download_path = Path.cwd()/'UrbanSound8K'

download_path = Path(r"C:\Users\lavan\Downloads\archive\UrbanSound8K\UrbanSound8K")

# Read metadata file
metadata_file = download_path/"metadata"/"UrbanSound8K.csv"

df = pd.read_csv(metadata_file)
print(df.head())

# Construct REAL file paths (full absolute paths)
df['relative_path'] = df.apply(
    lambda row: str(download_path / 'audio' / f"fold{row['fold']}" / row['slice_file_name']),
    axis=1
)

# Keep only needed columns
df = df[['relative_path', 'classID']]

print(df.head())

import math, random
import torch
import torchaudio
from torchaudio import transforms
from IPython.display import Audio

class AudioUtil():
  # ----------------------------
  # Load an audio file. Return the signal as a tensor and the sample rate
  # ----------------------------
    @staticmethod
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return (sig, sr)

 # ----------------------------
  # Convert the given audio to the desired number of channels
  # ----------------------------
    @staticmethod
    def rechannel(aud, new_channel):
        sig, sr = aud

        if (sig.shape[0] == new_channel):
      # Nothing to do
            return aud

        if (new_channel == 1):
      # Convert from stereo to mono by selecting only the first channel
            resig = sig[:1, :]
        else:
      # Convert from mono to stereo by duplicating the first channel
            resig = torch.cat([sig, sig])

        return ((resig, sr))

# Test rechannel

#test_file = df.iloc[0]['relative_path']
#aud = AudioUtil.open(test_file)

#mono = AudioUtil.rechannel(aud, 1)
#stereo = AudioUtil.rechannel(aud, 2)

#print("Original channels:", aud[0].shape[0])
#print("Mono shape:", mono[0].shape)
#print("Stereo shape:", stereo[0].shape)

# ----------------------------
  # Since Resample applies to a single channel, we resample one channel at a time
  # ----------------------------
    @staticmethod
    def resample(aud, newsr):
        sig, sr = aud

        if (sr == newsr):
      # Nothing to do
            return aud

        num_channels = sig.shape[0]
    # Resample first channel
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1,:])
        if (num_channels > 1):
      # Resample the second channel and merge both channels
            retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:,:])
            resig = torch.cat([resig, retwo])

        return ((resig, newsr))

    # ----------------------------
    # resize to same length 
  # Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds
  # ----------------------------
    @staticmethod
    def pad_trunc(aud, max_ms):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr//1000 * max_ms

        if (sig_len > max_len):
      # Trim the signal to the given length
            sig = sig[:,:max_len]

        elif (sig_len < max_len):
      # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

      # Pad with 0s
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((pad_begin, sig, pad_end), 1)
      
        return (sig, sr)

     # ----------------------------
  # Shifts the signal to the left or right by some percent. Values at the end
  # are 'wrapped around' to the start of the transformed signal.
  # ----------------------------
    @staticmethod
    def time_shift(aud, shift_limit):
        sig,sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len) # no of samples to shift
        return (sig.roll(shift_amt), sr)

# ----------------------------
  # Generate a Spectrogram
  # ----------------------------
    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
        sig,sr = aud
        top_db = 80

    # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

    # Convert to decibels
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return (spec)

 # ----------------------------
  # Augment the Spectrogram by masking out some sections of it in both the frequency
  # dimension (ie. horizontal bars) and the time dimension (vertical bars) to prevent
  # overfitting and to help the model generalise better. The masked sections are
  # replaced with the mean value.
  # ----------------------------
    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = spec.shape
        aug_spec = spec

        freq_mask_param = int(max_mask_pct * n_mels)
        for _ in range(n_freq_masks):
            freq_mask = transforms.FrequencyMasking(freq_mask_param, iid_masks=False)
            aug_spec = freq_mask(aug_spec)

        time_mask_param = int(max_mask_pct * n_steps)
        for _ in range(n_time_masks):
            time_mask = transforms.TimeMasking(time_mask_param, iid_masks=False)
            aug_spec = time_mask(aug_spec)

        return aug_spec
    
from torch.utils.data import DataLoader, Dataset, random_split

class SoundDS(Dataset):
    # Sound Dataset
    def __init__(self, df, data_path):
        self.df = df
        self.data_path = str(data_path)
        self.duration = 4000   # milliseconds
        self.sr = 44100        # target sample rate
        self.channel = 2       # target channels (stereo)
        self.shift_pct = 0.4   # up to 40% time shift
    
    # Number of items in dataset
    def __len__(self):
        return len(self.df)

    # Get i'th item in dataset
    def __getitem__(self, idx):
        audio_file = self.df.loc[idx, 'relative_path']
        class_id = self.df.loc[idx, 'classID']

        aud = AudioUtil.open(audio_file)

        reaud = AudioUtil.resample(aud, self.sr)
        rechan = AudioUtil.rechannel(reaud, self.channel)

        dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
        shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)

        sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
        aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

        return aug_sgram, class_id

# Split our data for training and validation

myds = SoundDS(df, download_path)
#myds = SoundDS(df, None)
print("Total items in dataset:", len(myds))

# Random split of 80:20 between training and validation
num_items = len(myds)
num_train = round(num_items * 0.8)
num_val = num_items - num_train
train_ds, val_ds = random_split(myds, [num_train, num_val])

train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=16, shuffle=False)

# Test: fetch one batch
print("\nFetching one batch from train_dl...")
xb, yb = next(iter(train_dl))
print("Batch X shape:", xb.shape)
print("Batch Y shape:", yb.shape)
print("First 5 labels:", yb[:5])


import torch.nn as nn           
import torch.nn.functional as F
from torch.nn import init
# ----------------------------
# Audio Classification Model
# ----------------------------
class AudioClassifier (nn.Module):

    def __init__(self):
        super().__init__()
        conv_layers = []

    # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

    # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

    # Third Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

    # Fourth Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

    #wrapping all convolutional blocks
        self.conv = nn.Sequential(*conv_layers)

        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
    # Run the convolutional blocks
        x = self.conv(x)
        x = self.ap(x)
        x = x.view(x.shape[0], -1)
        x = self.lin(x)
        return x

myModel = AudioClassifier()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(device)
#next(myModel.parameters()).device
print(next(myModel.parameters()).device)

# ----------------------------
# Training Loop
# ----------------------------
def training(model, train_dl, num_epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        steps_per_epoch=int(len(train_dl)),
        epochs=num_epochs,
        anneal_strategy='linear'
    )

    for epoch in range(num_epochs):
        model.train()  # <-- good practice: set model in training mode

        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0

        for i, data in enumerate(train_dl):
            inputs, labels = data[0].to(device), data[1].to(device)

            inputs_m, inputs_s = inputs.mean(), inputs.std()
            # Avoid division by zero
            inputs_s = torch.clamp(inputs_s, min=1e-8)
            inputs = (inputs - inputs_m) / inputs_s

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            _, prediction = torch.max(outputs, 1)
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        acc = correct_prediction / total_prediction
        print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')

    print('Finished Training')

# ----------------------------
# Actually run training
# ----------------------------
num_epochs = 2   # you can increase later
training(myModel, train_dl, num_epochs, device)

# Save the trained model
torch.save(myModel.state_dict(), 'audio_classifier.pth')
print("Model saved to 'audio_classifier.pth'")

# ----------------------------
# Inference
# ----------------------------
def inference(model, val_dl, device):
    model.eval()  # Set model to evaluation mode
    correct_prediction = 0
    total_prediction = 0

    # Disable gradient updates
    with torch.no_grad():
        for data in val_dl:
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            # Avoid division by zero
            inputs_s = torch.clamp(inputs_s, min=1e-8)
            inputs = (inputs - inputs_m) / inputs_s

            # Get predictions
            outputs = model(inputs)

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs, 1)

            # Count correct predictions
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += labels.shape[0]

    acc = correct_prediction / total_prediction
    print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')

# Run inference on trained model
inference(myModel, val_dl, device)
