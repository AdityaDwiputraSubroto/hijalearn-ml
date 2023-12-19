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
   def process_audio_to_spectrogram(file_path, target_length=18000):
    
        if file_path.endswith(".wav"):
                        wav, sr = librosa.load(file_path, sr = None)
                              
                        # Set a custom threshold for trimming (adjust as needed)
                        custom_top_db = 20
                        # Trim leading and trailing silence with a custom threshold
                        wav, _ = librosa.effects.trim(wav, top_db=custom_top_db)
   
                        wav = tf.convert_to_tensor(wav, dtype=tf.float32)
                        sr = tf.convert_to_tensor(sr, dtype=tf.int64)
                        wav = tfio.audio.resample(wav, rate_in=sr, rate_out=16000)
    
                        # Adjust the length of the audio sequence
                        if len(wav) < target_length:
                            # Zero-pad if the sequence is shorter than the target length
                            pad_size = target_length - len(wav)
                            wav = tf.pad(wav, paddings=[[0, pad_size]])
                        elif len(wav) > target_length:
                            # Trim if the sequence is longer than the target length
                             wav = wav[:target_length]
                        wav = np.array(wav)
                        
                        sr = float(sr)
                        # Size of the Fast Fourier Transform (FFT), which will also be used as the window length
                        n_fft=1024

                        # Step or stride between windows. If the step is smaller than the window length, the windows will overlap
                        hop_length=320
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
    spectrogram = process_audio_to_spectrogram(audio_file, target_length=18000)
    spectrogram = np.expand_dims(spectrogram, axis=0)
    predicted = model.predict(spectrogram)
    top_predict_index = np.argmax(predicted, axis=1)
    label_name = label_list[int(top_predict_index)]
    probability = predicted[0,top_predict_index]
    
    if true_label == label_name and probability>0.6:
            correct = True
    else:
            correct = False
            
    return correct, predicted, top_predict_index, label_name, probability
   ```
   ```
   audio_file = "/kaggle/input/hijaiyah-all/Dataset Hijaiyah/Hijaiyah polos/11_Za/1_0_NT_Za_T10 (2).wav"
   true_label = '11_Za'
   correct, predicted, top_predict_index, label_name, probability = predict_audio(audio_file, true_label, loaded_labels)
   print("Predict result : ", predicted)
   print("\nLabel index\t\t: ", top_predict_index)
   print("Predicted label name\t: ", label_name)
   print("True Label name\t\t: ", true_label)
   print("Probability\t\t: ",probability)
   print("Correct?\t\t: ",correct)
   ```
   Output
   ```
   Predict result :  [[5.9871570e-12 9.9999952e-01 5.0053152e-14 1.0658123e-19 1.0835376e-13
     9.6104669e-12 1.6341000e-15 3.8584639e-07 5.7546735e-20 7.3074901e-13
     1.4545040e-13 2.2054159e-11 1.7658119e-15 9.1710222e-15 2.6104492e-14
     7.3043607e-15 1.5159074e-17 2.9278770e-11 1.1090896e-12 1.9755935e-09
     4.0145016e-11 4.3219130e-09 7.9129232e-13 6.4669340e-16 5.1729721e-15
     2.3864520e-15 3.3022500e-12 2.1758172e-15 5.6094798e-08]]
   
   Label index		:  [1]
   Predicted label name	:  11_Za
   True Label name		:  11_Za
   Probability		:  [0.9999995]
   Correct?		:  True
   ```
