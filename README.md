# hijalearn-ml
Machine learning project that evaluate hijaiyah pronunciation

## Hijaiyah Directory
List of Hijaiyah Directory <br>
https://docs.google.com/spreadsheets/d/1no5iTYEPH5Qz63lp0rJFNWCPSn-TJBe50PpQ8uNChYE/edit?usp=sharing


## Models and Labels Link
https://drive.google.com/drive/folders/14YmzV3eUagTSMGXVJ_qW6ETnQfW8sai1?usp=sharing

## How to predict audio
To make predictions using our model, follow these step-by-step instructions <br>
1. Import packages
   ```
   import librosa
   import numpy as np
   import matplotlib.pyplot as plt
   import tensorflow as tf
   from scipy.signal import spectrogram
   import tensorflow_io as tfio
   import tensorflow_hub as hub
   import csv
   ```
1. Load model
   ```
    from keras.models import load_model
    model_path = '/kaggle/working/model_polos_inception_89.h5' #CHANGE THIS
    loaded_model = load_model(model_path)
   ```
1. Load labels from csv
   ```
   def load_labels_from_csv(csv_file_path):
    labels = []

   with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # Skip the header row if present

        for row in csv_reader:
            labels.append(row[0])

   return labels
   ```
   ```
   csv_file_path = '/kaggle/working/polos_labels.csv' #CHANGE THIS
   loaded_labels = load_labels_from_csv(csv_file_path)
   print(loaded_labels)
   ```
1. Process audio to spectrogram
   ```
   def process_audio_to_spectrogram(file_path, target_length=12000):
    
        if file_path.endswith(".wav"):
                        wav, sr = librosa.load(file_path, sr = None)
                        wav = tf.convert_to_tensor(wav, dtype=tf.float32)
                        sr = tf.convert_to_tensor(sr, dtype=tf.int32)
                        sample_rate = tf.cast(sr, dtype=tf.int64)
                        wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    
                        # Adjust the length of the audio sequence
                        if len(wav) < target_length:
                            # Zero-pad if the sequence is shorter than the target length
                            pad_size = target_length - len(wav)
                            wav = tf.pad(wav, paddings=[[0, pad_size]])
                        elif len(wav) > target_length:
                            # Trim if the sequence is longer than the target length
                             wav = wav[:target_length]
                        wav = np.array(wav)
                        
                        sr = float(sample_rate)
                        # Size of the Fast Fourier Transform (FFT), which will also be used as the window length
                        n_fft=1024

                        # Step or stride between windows. If the step is smaller than the window length, the windows will overlap
                        hop_length=160
                        sr = float(sr)
                        window_type ='hann'
                        mel_bins = 128
                        fmin = 0
                        fmax = None
                        Mel_spectrogram = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=window_type, n_mels=mel_bins,                           power=2.0)

                        mel_spectrogram_db = librosa.power_to_db(Mel_spectrogram, ref=np.max)
                        
                        #change to rgb image
                        cmap = plt.get_cmap('jet')  # You can choose other colormaps
                        mel_spectrogram_db = cmap(mel_spectrogram_db / np.min(mel_spectrogram_db))[:, :, :3]
                        
                        #Resize Image
                        mel_spectrogram_db = tf.image.resize(mel_spectrogram_db, size=(128, 75)).numpy()
                        

        return mel_spectrogram_db
   ```
1. Predict
   ```
   def predict_audio(audio_file, true_label, label_list):
    spectrogram = process_audio_to_spectrogram(audio_file, target_length=12000)
    spectrogram = np.expand_dims(spectrogram, axis=0)
    predicted = model.predict(spectrogram)
    top_predict_index = np.argmax(predicted, axis=1)
    label_name = label_list[int(top_predict)]
    probability = predicted[0,top_predict_index]
    
    return predicted, top_predict_index, label_name, probability
   ```
   ```
   audio_file = "/kaggle/input/dataset-hijaiyah/Dataset Hijaiyah/Hijaiyah polos/25_Nun/2_4_NT_Noon_T10001.wav" #CHANGE THIS
   true_label = '25_nun' #CHANGE THIS
   predicted, top_predict_index, label_name, probability = predict_audio(audio_file, true_label, loaded_labels)

   print("Predict result : ", predicted)
   print("\nLabel index\t\t: ", top_predict_index)
   print("Predicted label name\t: ", label_name)
   print("True Label name\t\t: ", true_label)
   print("Probability\t\t: ",probability)
   ```
   Output
   ```
   Predict result :  [[2.6527500e-09 1.8013134e-12 6.5256778e-10 1.3944617e-11 2.3759122e-15
   1.1514542e-06 8.5240730e-09 1.5967119e-09 3.6847848e-07 7.1299451e-07
   5.5577043e-08 8.0551532e-14 1.8430099e-09 8.5623647e-13 5.8353800e-07
   9.3332565e-05 9.9983811e-01 4.4479260e-05 2.9057171e-13 1.3719097e-05
   7.2057701e-06 1.3950576e-11 2.0036668e-15 1.1864861e-11 3.4788332e-07
   2.4487580e-18 1.9933662e-16 1.1143029e-11 1.5962605e-10]]
    
   Label index		:  [16]
   Predicted label name	:  25_Nun
   True Label name		:  25_nun
   Probability		:  [0.9998381]
   ```
