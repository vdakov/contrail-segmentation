import librosa 
import os 
import numpy as np
from tqdm import tqdm
from data_util.AudioMNIST.extract_features import preprocess_wav_into_features

def load_audio_data(folder_path, max_samples=3000):
    X, y, wavs = [], [], []
    for file in tqdm(os.listdir(folder_path)[:max_samples]):
        x = librosa.load(os.path.join(folder_path, file), sr=44100)[0]
        wavs.append(x)
        x = np.array(x)
        X.append(preprocess_wav_into_features(x))
        y.append(file[:1])
    X = np.array(X)
    y = np.array(y)
    
    return X, y, wavs

def load_audio_data_noisy(folder_path, max_samples=3000):
    X, y, wavs = [], [], []
    for file in tqdm(os.listdir(folder_path)[:max_samples]):
        x = librosa.load(os.path.join(folder_path, file), sr=44100)[0]
        X.append(preprocess_wav_into_features(np.array(x)))
        x += np.random.normal(0, 0.001, x.shape) 
        
        wavs.append(x)
        x = np.array(x)
        X.append(preprocess_wav_into_features(x))
        y.append(file[:1])
        y.append(file[:1])
    X = np.array(X)
    y = np.array(y)
    
    return X, y, wavs