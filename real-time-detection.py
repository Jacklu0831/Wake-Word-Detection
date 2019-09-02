from pydub import AudioSegment
import pyaudio

import numpy as np
import time
import os
import IPython
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from queue import Queue
from threading import Thread

# initialize parameters
model_path = "./model/final_model.h5"
threshold = 0.8
silence_threshold = 100
record_time = 60
channels = 1

T_x = 5511
T_y = 1375
n_freq = 101

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
    note: spec has shape (n_freqs, T_x)
    """
    spec_data = spec_data.swapaxes(0, 1)
    spec_data = np.expand_dims(spec_data, axis=0)
    preds = model.predict(spec_data)
    preds = preds.reshape(-1)
    return preds # flatten

def is_new_detection(preds, chunk_duration, feed_duration, threshold=0.5):
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




def callback(in_data, frame_count, time_info, status):
    global run, timeout, data, threshold
    if time.time() > timeout:
        run = False
    # read new data from buffer and process it
    new_data = np.frombuffer(in_data, dtype='int16')
    if np.abs(new_data).mean() < threshold:
        print('0', end='')
        return (in_data, pyaudio.paContinue)
    else:
        print('1', end='')
        # add the new data if not silent
        data = np.append(data, new_data)
        if len(data) > feed_samples:
            data = data[-feed_samples:]
            que.put(data)
        return (in_data, pyaudio.paContinue)




# define and start stream
que = Queue() # enables communication between audio callback and main thread
run = True
timeout = time.time() + record_time # half a minute
data = np.zeros(feed_samples, dtype='int16') # data buffer for input

run = True

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
		new_wake = is_new_detection(preds, chunk_duration, feed_duration)
		if new_wake:
			print('█', end='')
			os.system("say hello world")
			stream.stop_stream()
			stream.close()
except (KeyboardInterrupt, SystemExit):
    stream.stop_stream()
    stream.close()
    timeout = time.time()
    run = False

stream.stop_stream()
stream.close()