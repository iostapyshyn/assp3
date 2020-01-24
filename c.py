#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
plt.style.use('ggplot')

from a import *

FILENAME = 'audio/aeiou_16000.wav'
DURATION = 100

# Computes CEPSTRUM for whole signal.
# The FFT size depends on the supplied window size
# window - analysis window
# hop - hop size
# count - the size of the matrix column
def stcep(y, window, hop, count):
    N = len(window)
    hops = len(y)//hop

    kmat = np.zeros((hops, count))
    for i in range(hops):
        buf = y[i*hop:(i*hop)+N]
        buf *= window[:len(buf)]

        fftbuf = np.fft.fft(buf)
        cepbuf = cepstrum(fftbuf)

        kmat[i] = cepbuf[0:count]

    return kmat

if __name__ == '__main__':
    # Read and normalize the audio file
    fs, y = scipy.io.wavfile.read(FILENAME)
    y = y * (0.99 / max(abs(y))) # Normalize

    # Calculate the number of samples for analysis
    sample_duration_ms = 1/fs * 1000 # duration of one sample in ms
    samples = int(DURATION / sample_duration_ms)

    kmat = stcep(y, np.hanning(samples), 512, 200)

    fig = plt.figure()
    sub = fig.add_subplot(111)
    for i in kmat:
        plot_cepstrum(sub, i, fs)
        plt.pause(DURATION/1000)
        plt.cla()

    plt.show()
