from flask import Flask, request, render_template,redirect, url_for, flash
import numpy as np
from xgboost import XGBClassifier
import os
import torchaudio

app = Flask(__name__)
model = XGBClassifier()
model.load_model("Audio_XBG_model.pth")

app.secret_key = "hello"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/nofile')
def nofile():
    return f'<h3><center>You have entered/uploaded invalid file<br>Enter a VALID FILE</center></h3>'

@app.route('/predict',methods=['GET', 'POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    if request.method == 'POST':
        if request.files:
            audio_file = request.files['file']

            if audio_file.filename == "":
                #print("No filename")
                return redirect(url_for('nofile'))

            audio_file.save(audio_file.filename)
            """
            print("Audio File: ",audio_file)

            audio, sample_rate = librosa.load(audio_file.filename, res_type='kaiser_fast')
            def mfcc_extractor(file_, sample_rate):
                mfccs_features = librosa.feature.mfcc(y=file_, sr=sample_rate, n_mfcc=64)
                mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
                return mfccs_scaled_features
            test_audio = mfcc_extractor(audio, sample_rate)
            test_audio.shape = -1,64
            """
            signal, sr = torchaudio.load(audio_file.filename)
            
            SAMPLE_RATE = 22050
            NFFT = 1024
            HLEN = 512
            NMEL = 64
            
            mel_spectogram = torchaudio.transforms.MelSpectrogram(
                sample_rate= SAMPLE_RATE,
                n_fft= NFFT,
                hop_length= HLEN,
                n_mels= NMEL)(signal)
            
            if sr != SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, 22050)
                mel_spectogram_resampled = resampler(signal)
            
            if mel_spectogram_resampled.shape[0] > 1:
                mel_signal = np.mean(mel_spectogram_resampled, dim=0, keepdim=True)
            
            if mel_signal.shape[1] > NMEL:
                mel_signal = mel_signal[:, :NMEL]

            
            class_mapping = [
            "Star Wars",
            "Harry Potter"
            ]

            #prediction_index = model.predict(np.array(mel_signal.tolist()))
            prediction_index = model.predict(mel_signal.tolist())
            prediction = class_mapping[prediction_index[0]]

            # Remove audio file
            os.remove(audio_file.filename)
            return render_template('index.html', prediction_text='Prediction: {}'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)