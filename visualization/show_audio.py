from IPython.display import Audio, display
import matplotlib.pyplot as plt
import librosa.display
import numpy as np

def visualize_waveform_and_play_audio(y, wavs, sample_rate=44100):
    for i in range(len(y)):
        #Plot waveform and spectrogram
        
        fig, ax = plt.subplots(1, 2, figsize=(20, 5))
        ax[0].set_title(f'Waveform of Audio Sample {i} - Label: {y[i]}')
        librosa.display.waveshow(wavs[i], sr=sample_rate, ax=ax[0])
        ax[0].set_xlabel('Time (s)')
        ax[0].set_ylabel('Amplitude')
        
        S = librosa.feature.melspectrogram(y=wavs[i], sr=sample_rate, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, sr=sample_rate, x_axis='time', y_axis='mel', ax=ax[1], cmap='coolwarm')
        ax[1].set_title(f'Spectrogram of Audio Sample {i} - Label: {y[i]}')
        plt.colorbar(img, ax=ax[1], format='%+2.0f dB')
        plt.suptitle(f'Audio Sample {i} - Label: {y[i]}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        
        # plt.figure(figsize=(10, 3))
        # librosa.display.waveshow(y[i], sr=sample_rate)
        # plt.figure(figsize=(10, 3))
        # librosa.display.waveshow(wavs[i], sr=sample_rate)
        # plt.title(f'Audio Sample {i} - Label: {y[i]}')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Amplitude')
        # plt.tight_layout()
        # plt.show()
        
        # Play audio
        display(Audio(wavs[i], rate=sample_rate))