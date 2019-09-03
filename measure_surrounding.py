# This program is for estimating the magnitude of surrounding voice

import pyaudio

import time
import numpy as np
import os
import sys
import IPython
import matplotlib.pyplot as plt

# parameters
T_x = 5511
T_y = 1375
n_freq = 101

fs = 44100
chunk_duration = 0.5 # each read window
chunk_samples = int(fs * chunk_duration)

audio_data = None
arr_data = []

# callback function
def callback(in_data, frame_count, time_info, status):
    global audio_data
    audio_data = np.frombuffer(in_data, dtype='int16')
    ave = np.abs(audio_data).mean()
    print(int(ave))
    arr_data.append(ave)
    return (in_data, pyaudio.paContinue)

# start recording
print()
print("Starting to estimate surrounding volume...")
time.sleep(2)
print()

# set up stream
stream = pyaudio.PyAudio().open(
    format=pyaudio.paInt16,
    channels=1,
    rate=fs,
    input=True,
    frames_per_buffer=chunk_samples,
    input_device_index=0,
    stream_callback=callback)

stream.start_stream()
time.sleep(5.1)
stream.stop_stream()
stream.close()

print("total average: ", sum(arr_data)/len(arr_data))