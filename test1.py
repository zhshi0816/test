from moviepy.editor import *
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import librosa
import librosa.display as Dp
from test2 import *
import resampy


file_path = "/Users/shi/Desktop/1.mp4"
# audio = wavio.read(file_path)

video = VideoFileClip(file_path)
audio = video.audio
print(audio.duration)
print(audio.fps)
audioArray = audio.to_soundarray()
leftChannel = audioArray[:42998,0]
plt.plot(leftChannel)
plt.show()

# leftChannel = resampy.resample(leftChannel, 44100, 16000)

log_mel = log_mel_spectrogram(leftChannel,
                        audio_sample_rate=44100,
                        log_offset=0.01,
                        window_length_secs=0.025,
                        hop_length_secs=0.010,
                        num_mel_bins=64,
                        lower_edge_hertz=125.0,
                        upper_edge_hertz=7500.0
                        )
print(log_mel.shape)
log_mel = stats.zscore(log_mel)
Dp.specshow(log_mel)
plt.show()