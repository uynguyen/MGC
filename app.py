# NOTE: THIS IS FOR FLASK

# from flask import Flask, render_template, request

# app = Flask(__name__)

# @app.route("/")
# def home():
#     return render_template('homepage.html')

# @app.route("/predict", methods=["POST"])
# def prediction():
#     audio = request.form['myfile']

#     # if audio.endswith(".mp3"):
#     #     import subprocess
#     #     import os.path
#     #     extension = os.path.splitext(audio)[0]
#     #     subprocess.call(['ffmpeg', '-i', audio, extension+'.wav'])

#     #     t1 = 60 * 1000  # Works in milliseconds
#     #     t2 = 90 * 1000
#     #     newAudio = AudioSegment.from_wav(extension+'.wav')
#     #     newAudio = newAudio[t1:t2]
#     #     newAudio.export(extension+'.wav', format="wav")  # Exports to a wav file in the current path.
#     #     audio=extension+'.wav'

#     return render_template('prediction.html', title="Prediction",
#                            prediction="",probability="")

# if __name__ == '__main__':
#     app.run(debug=True)

# NOTE: THIS IS FOR STREAMLIT
import torchaudio
import streamlit as st
import keras
import numpy as np
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class AudioUtil():
    @staticmethod
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return (sig, sr)
    @staticmethod
    def rechannel(aud, new_channel):
        sig, sr = aud
        if (sig.shape[0] == new_channel):
            return aud
        if (new_channel == 1):
            resig = sig[:1, :]
        else:
            resig = torch.cat([sig, sig])
            
        return ((resig, sr))
    @staticmethod
    def resample(aud, newsr):
        sig, sr = aud
        if (sr == newsr):
            return aud
        
        num_channels = sig.shape[0]
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1,:])
        if (num_channels > 1):
            retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:,:])
            resig = torch.cat([resig, retwo])
        return ((resig, newsr))
    @staticmethod
    def pad_trunc(aud, max_ms):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr//1000 * max_ms

        if (sig_len > max_len):
            # Truncate the signal to the given length
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
    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
        sig, sr = aud
        top_db = 80
        
        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        
        return (spec)

class SpectrogramCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SpectrogramCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.25)
        self.dropout_fc = nn.Dropout(0.5)

    def forward(self, x):
        # Convolutional layers with MaxPool
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        
        # Adaptive pooling to ensure fixed output dimensions
        x = self.adaptive_pool(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
model = SpectrogramCNN()

model.load_state_dict(torch.load('MGC_Model.pth'))
model.to(device)
model.eval()

# Title and project details
st.title("Music Genre Classification")

# Additional project info
st.markdown("""
**Name**: Nguyen Long Uy  
**Student ID**: MITIU24206  
**Description**: This project is a Music Genre Classification application that leverages a Convolutional Neural Network (CNN) to classify music into 10 different genres. It provides a user-friendly interface where users can upload audio files and receive predictions about the genre based on model analysis.
""")

# Audio file upload
audio_file = st.file_uploader("Upload an audio file (.wav or .mp3)", type=["wav", "mp3"])

# Handle audio file and make predictions
if audio_file:
    # Convert MP3 to WAV and extract segment if necessary
    if audio_file.name.endswith(".mp3"):
        with open("temp.mp3", "wb") as f:
            f.write(audio_file.read())

        waveform, sample_rate = torchaudio.load("temp.mp3")
        # Define the segment start and end points (in seconds)
        start_time = 60  # 1 minute
        end_time = 90    # 1.5 minutes

        # Calculate sample positions
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)

        # Extract the segment
        segment_waveform = waveform[:, start_sample:end_sample]
        torchaudio.save("temp.wav", segment_waveform, sample_rate)
        audio_path = "temp.wav"
    else:
        with open("temp.wav", "wb") as f:
            f.write(audio_file.read())
        audio_path = "temp.wav"

    audio, sr = AudioUtil.open(audio_path)
    audio = AudioUtil.rechannel((audio, sr), 1)
    audio = AudioUtil.resample(audio, 44100)
    audio = AudioUtil.pad_trunc(audio, 5500)
    spec = AudioUtil.spectro_gram(audio)
    spec = spec.unsqueeze(0)
    
    # Model prediction
    with torch.no_grad():
        output = model(spec.to(device))
        
        # Get probabilities using softmax
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]  # Get softmax probabilities for the first item in batch
        
        # Get top 3 predictions with probabilities
        top_probs, top_idxs = torch.topk(probabilities, 3)
        top_probs = top_probs.cpu().numpy() * 100  # Convert to percentage
        top_idxs = top_idxs.cpu().numpy()


    # Display predictions in columns
    st.subheader("Prediction Results")

    # Create columns for each prediction
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🏆 Top Prediction")
        st.markdown(f"**Genre**: {genres[int(top_idxs[0])]}")
        st.markdown(f"**Confidence**: {top_probs[0]:.2f}%")
        st.progress(int(top_probs[0]))

    with col2:
        st.markdown("### 🥈 Second Prediction")
        st.markdown(f"**Genre**: {genres[int(top_idxs[1])]}")
        st.markdown(f"**Confidence**: {top_probs[1]:.2f}%")
        st.progress(int(top_probs[1]))

    with col3:
        st.markdown("### 🥉 Third Prediction")
        st.markdown(f"**Genre**: {genres[int(top_idxs[2])]}")
        st.markdown(f"**Confidence**: {top_probs[2]:.2f}%")
        st.progress(int(top_probs[2]))
        
    with open(audio_path, "rb") as audio_file:
        st.audio(audio_file.read(), format="audio/wav")

    # Clean up temporary files
    import os
    if os.path.exists("temp.mp3"):
        os.remove("temp.mp3")
    if os.path.exists("temp.wav"):
        os.remove("temp.wav")
else:
    st.warning("Please upload an audio file to get predictions.")