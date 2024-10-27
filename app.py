# from flask import Flask, render_template, request
# from pydub import AudioSegment

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

import streamlit as st
import keras
import numpy as np
import math
import os
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F

class MusicGenreCNN(nn.Module):
    def __init__(self):
        super(MusicGenreCNN, self).__init__()
        # Define layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)  # Assuming input spectrograms of size 128x128
        self.fc2 = nn.Linear(256, 10)  # Assuming 10 music genres

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Apply Conv + Relu + Pooling
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)  # Flatten before feeding to fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

genre_dict = {0: "disco", 1: "pop", 2: "classical", 3: "metal", 4: "rock", 5: "blues", 6: "hiphop", 7: "reggae",
                  8: "country", 9: "jazz"}

model = keras.models.load_model("MusicGenre_CNN_79.73.h5")


cnn_model = MusicGenreCNN()

# Load the state_dict from the .pth file
cnn_model.load_state_dict(torch.load('./model.pth', map_location=torch.device('cpu')))  # Adjust path as necessary

# Set the model to evaluation mode if you’re using it for inference
cnn_model.eval()

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

# Function to process audio
def process_input(audio_path, track_duration):
    SAMPLE_RATE = 22050
    NUM_MFCC = 13
    N_FTT = 2048
    HOP_LENGTH = 512
    TRACK_DURATION = track_duration  # measured in seconds
    SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
    NUM_SEGMENTS = 10

    samples_per_segment = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / HOP_LENGTH)

    signal, sample_rate = librosa.load(audio_path, sr=SAMPLE_RATE)

    for d in range(NUM_SEGMENTS):
        # Calculate start and finish sample for current segment
        start = samples_per_segment * d
        finish = start + samples_per_segment

        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate, n_mfcc=NUM_MFCC, n_fft=N_FTT, hop_length=HOP_LENGTH)
        mfcc = mfcc.T
        return mfcc

# Handle audio file and make predictions
if audio_file:
    # Convert MP3 to WAV and extract segment if necessary
    if audio_file.name.endswith(".mp3"):
        with open("temp.mp3", "wb") as f:
            f.write(audio_file.read())
        audio_segment = AudioSegment.from_mp3("temp.mp3")
        segment = audio_segment[60 * 1000: 90 * 1000]  # 1 to 1.5 minutes
        segment.export("temp.wav", format="wav")
        audio_path = "temp.wav"
    else:
        with open("temp.wav", "wb") as f:
            f.write(audio_file.read())
        audio_path = "temp.wav"

    # Process audio and make prediction
    audio_data = process_input(audio_path, 30)
    X_to_predict = audio_data[np.newaxis, ..., np.newaxis]

    # Model prediction
    
    # prediction = model.predict(X_to_predict)
    # pred = np.argmax(prediction)
    # proba = np.max(prediction) * 100

    # # Calculate second and third most likely genres
    # sorted_indices = np.argsort(prediction[0])
    # second_pred = genre_dict[sorted_indices[-2]]
    # second_prob = prediction[0][sorted_indices[-2]] * 100
    # third_pred = genre_dict[sorted_indices[-3]]
    # third_prob = prediction[0][sorted_indices[-3]] * 100
    
    
    # Perform inference
    with torch.no_grad():  # Disable gradient computation for inference
        output = cnn_model(X_to_predict)
        print(output)

    # Display predictions in columns
    st.subheader("Prediction Results")

    # Create columns for each prediction
    # col1, col2, col3 = st.columns(3)

    # with col1:
    #     st.markdown("### 🏆 Top Prediction")
    #     st.markdown(f"**Genre**: {genre_dict[int(pred)]}")
    #     st.markdown(f"**Confidence**: {proba:.2f}%")
    #     st.progress(int(proba))

    # with col2:
    #     st.markdown("### 🥈 Second Prediction")
    #     st.markdown(f"**Genre**: {second_pred}")
    #     st.markdown(f"**Confidence**: {second_prob:.2f}%")
    #     st.progress(int(second_prob))

    # with col3:
    #     st.markdown("### 🥉 Third Prediction")
    #     st.markdown(f"**Genre**: {third_pred}")
    #     st.markdown(f"**Confidence**: {third_prob:.2f}%")
    #     st.progress(int(third_prob))


    # Clean up temporary files
    import os
    if os.path.exists("temp.mp3"):
        os.remove("temp.mp3")
    if os.path.exists("temp.wav"):
        os.remove("temp.wav")
else:
    st.warning("Please upload an audio file to get predictions.")