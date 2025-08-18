from IPython.display import Audio, display
import matplotlib.pyplot as plt
import librosa.display

def visualize_waveform_and_play_audio(y, wavs, sample_rate=44100):
    for i in range(len(y)):
        plt.figure(figsize=(10, 3))
        librosa.display.waveshow(wavs[i], sr=sample_rate)
        plt.title(f'Audio Sample {i} - Label: {y[i]}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.tight_layout()
        plt.show()
        
        # Play audio
        display(Audio(wavs[i], rate=sample_rate))