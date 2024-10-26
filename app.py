from flask import Flask, render_template, request
from pydub import AudioSegment

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('homepage.html')

@app.route("/predict", methods=["POST"])
def prediction():
    audio = request.form['myfile']

    # if audio.endswith(".mp3"):
    #     import subprocess
    #     import os.path
    #     extension = os.path.splitext(audio)[0]
    #     subprocess.call(['ffmpeg', '-i', audio, extension+'.wav'])

    #     t1 = 60 * 1000  # Works in milliseconds
    #     t2 = 90 * 1000
    #     newAudio = AudioSegment.from_wav(extension+'.wav')
    #     newAudio = newAudio[t1:t2]
    #     newAudio.export(extension+'.wav', format="wav")  # Exports to a wav file in the current path.
    #     audio=extension+'.wav'

    return render_template('prediction.html', title="Prediction",
                           prediction="",probability="")

# if __name__ == '__main__':
#     app.run(debug=True)
