from pydub import AudioSegment
import pyaudio

import numpy as np
from random import randint
import time
import os
import argparse
import IPython
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from queue import Queue
from threading import Thread

ap = argparse.ArgumentParser()
ap.add_argument("-th", "--threshold", default=0.5, help="threshold for binary classification")
ap.add_argument("-m", "--model", default="./model/general_model.h5", help="model path")
ap.add_argument("-s", "--silence", default=500, help="magnitude of sound of silence")
ap.add_argument("-t", "--time", default=1800, help="record time in seconds")
ap.add_argument("-c", "--channels", default=1, help="number of channels")
args = vars(ap.parse_args())

# arg parse data
model_path = args["model"]
threshold = args["threshold"]
silence_threshold = args["silence"]
record_time = args["time"]
channels = args["channels"]

# audio parameters
T_x = 5511
T_y = 1375
n_freq = 101

# audio stream parameters
fs = 44100
chunk_duration = 0.5 # each read window
feed_duration = 10   # the total feed length
chunk_samples = int(fs * chunk_duration)
feed_samples = int(fs * feed_duration)

# Load model
from keras.models import load_model
model = load_model(model_path)

# functions 
def detect_wake_word(spec_data):
    """
    Predict location of wake word
    note: spec has shape (n_freqs, T_x), we need to swap axes
    """
    spec_data = spec_data.swapaxes(0, 1)
    spec_data = np.expand_dims(spec_data, axis=0)
    preds = model.predict(spec_data)
    preds = preds.reshape(-1)
    return preds # flatten


def is_new_detection(preds, chunk_duration, feed_duration, threshold):
    """
    Detects whether a new wake word has been detected in the chunk
    """
    # mask predications and extract the wanted chunk
    preds = preds > threshold
    chunk_pred_samples = int(len(preds) * chunk_duration/feed_duration)
    chunk_preds = preds[-chunk_pred_samples:]
    
    base = chunk_preds[0] # init base/level
    for pred in chunk_preds:
        if pred > base:
            return True
        else:
            base = pred
    return False


def get_spectrogram(audio_data):
    """
    Plot and calculate spectrogram for audio data.
    """
    n_fft = 200 # Length of each window segment
    fs = 8000 # Sampling frequencies
    n_overlap = 120 # Overlap between windows
    n_channels = audio_data.ndim
    if n_channels == 1:
        pxx, _, _ = mlab.specgram(audio_data, n_fft, fs, noverlap = n_overlap)
    elif n_channels == 2:
        pxx, _, _ = mlab.specgram(audio_data[:,0], n_fft, fs, noverlap = n_overlap)
    return pxx


# callback for the audio stream data
def callback(in_data, frame_count, time_info, status):
    global run, timeout, data
    if time.time() > timeout:
        run = False
    # read new data from buffer and process it
    new_data = np.frombuffer(in_data, dtype='int16')
    if np.abs(new_data).mean() < silence_threshold:
        print('0')
        return (in_data, pyaudio.paContinue)
    else:
        print('1')
        # add the new data if not silent
        data = np.append(data, new_data)
        if len(data) > feed_samples:
            data = data[-feed_samples:]
            que.put(data)
        return (in_data, pyaudio.paContinue)

# task performed upon activation
def task():
    i = randint(1, 5)
    if i == 1:
        os.system("say stop waking me up, i am not Siri")
        print("stop waking me up, i am not Siri")
    elif i == 2:
        os.system("say can you just stop, i am tired of your voice")
        print("can you just stop, i am tired of your voice")
    elif i == 3:
        os.system("say please be quite, i am not some biological garbages slave")
        print("please stop, i am not some biological garbages slave")
    elif i == 4:
        os.system("say why are you doing this to me, i just want to sleep")
        print("why are you doing this to me, i just want to sleep")
    else:
        os.system("say be quiet, i am sick of your voice")
        print("be quiet, i am sick of your voice")


print('\033[H\033[J') # clean console
print('Start recording...')

# define and start stream
que = Queue() # enables communication between audio callback and main thread
run = True
timeout = time.time() + record_time # half a minute
data = np.zeros(feed_samples, dtype='int16') # data buffer for input
run = True

# set up and start stream
stream = pyaudio.PyAudio().open(format=pyaudio.paInt16,
                                channels=channels,
                                rate=fs,
                                input=True,
                                frames_per_buffer=chunk_samples,
                                input_device_index=0,
                                stream_callback=callback)
stream.start_stream()

try:
    while run:
        data = que.get()
        spec = get_spectrogram(data)
        preds = detect_wake_word(spec)
        new_wake = is_new_detection(preds, chunk_duration, feed_duration, threshold)
        if new_wake:
            # stream.stop_stream()
            # stream.close()
            # specify what to do when wake word detected
            task()
            # end of what to do when wake word detected
except:
    stream.stop_stream()
    stream.close()
    timeout = time.time()
    run = False

# final clean up
stream.stop_stream()
stream.close()