import librosa
import numpy as np

def preprocess_wav_into_features(x, sample_rate=44100):
    mfccs = librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=13)
    delta = librosa.feature.delta(mfccs)
    delta2 = librosa.feature.delta(mfccs, order=2)


    feat = np.concatenate([
        mfccs.mean(axis=1),
        delta.mean(axis=1),
        delta2.mean(axis=1),
    ])
    return feat  
