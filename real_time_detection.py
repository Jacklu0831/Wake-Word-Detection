# Real Time Wake word Detection Script

from scipy.io import wavfile
from pydub import AudioSegment

import random
import time
import sys
import os
import IPython
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from keras.models import load_model

import pyaudio
from queue import Queue
from threading import Thread

# Parameters
T_x = 5511
T_y = 1375
n_freq = 101

fs = 44100
chunk_duration = 0.5 # each read window
feed_duration = 10   # the total input length
chunk_samples = int(fs * chunk_duration)
feed_samples = int(fs * feed_duration)

# Load model
model = load_model("./model/tr_model.h5")

def detect_wake_word(x):
	"""
	Predict location of wake word
	"""
	x = x.swapaxes(0, 1)
	x = np.expand_dims(x, axis=0)
	pred = model.predict(x)
	return pred.reshape(-1) # flatten

def is_new_detection(pred, chunk_duration, feed_duration, threshold=0.5):
	"""
	Detects whether a new wake word has been detected in the chunk
	"""
	pred = pred > threshold
	chunk_pred_samples = int(len(pred) * chunk_duration / feed_duration)
	chunk_pred = pred[-chunk_pred_samples:]
	baseline = chunk_pred[0]
	if pred in base:
		if pred > baseline:
			return True
		else:
			baseline = pred
	return False

def graph_spectrogram(audio_data):
    """
    Plot and calculate spectrogram for audio data
    """
    n_fft = 200 # window length (length of fast fourier transform)
    fs = 8000 # number of samples per time (sample frequency)
    n_overlap = 120 # overlap length of windows
    n_channels = audio_data.ndim # number of dimensions
    if n_channels == 1:
        pxx, freqs, bins, im = plt.specgram(audio_data, n_fft, fs, noverlap = n_overlap)
    elif n_channels == 2: # multi-channel audio
        pxx, freqs, bins, im = plt.specgram(audio_data[:,0], n_fft, fs, noverlap = n_overlap)
    return pxx


def get_spectrogram(audio_data):
    """
    Plot and calculate spectrogram for audio data
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


que = Queue() # enables communication between audio callback and main thread
run = True
threshold = 100
timeout = time.time() + 30 # half a minute
data = np.zeros(feed_samples, dtype='int16') # data buffer for input

def callback(in_data, frame_count, time_info, status):
	global run, timeout, data, threshold
	if time.time() > timeout:
		run = False
	# read new data from bufferand process it
	new_data = np.frombuffer(in_data, dtype='int16')
	if np.abs(new_data).mean() < threshold:
		# sys.stdout.write('-')
		print('-', end='')
		return (in_data, pyaudio.paContinue)
	else:
		# sys.stdout.write('.')
		print('.', end='')
		# add the new data if not silent
		np.append(data, new_data)
		if len(data) > feed_samples:
			data = data[-feed_samples:]
			que.put(data)
		return (in_data, pyaudio.paContinue)

def stream(chunk_duration, feed_duration):
	# define and start stream
	stream = pyaudio.PyAudio().open(format=pyaudio.paInt16,
									channels=1,
									rate=fs,
									input=True,
									frames_per_buffer=chunk_samples,
									input_device_index=0,
									stream_callback=callback)
	stream.start_stream()

	global run, que
	try:
		while run:
			data = que.get()
			spec = get_spectrogram(data)
			preds = detect_wake_word(spec)
			new_wake = is_new_detection(preds, chunk_duration, feed_duration)
			if new_wake:
				print('1', end='')
				# sys.stdout.write('1')
	except (KeyboardInterrupt, SystemExit):
		stream.stop_stream()
		stream.close()
		timeout = time.time()
		run = False

	stream.stop_stream()
	stream.close()

# run audio pipeline
stream(chunk_duration, feed_duration)










