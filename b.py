#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile

from a import *

plt.style.use('ggplot')
DURATION = 100 # ms
FILENAME = 'audio/aeiou_16000.wav'
FILTER = 50
PLOT_CAP = 300

TIMESTAMPS = {
    'a': 0.85,
    'e': 1.5,
    'i': 2.3,
    'o': 3.3,
    'u': 3.8,
}

if __name__ == '__main__':
    # Read and normalize the audio file
    fs, y = scipy.io.wavfile.read(FILENAME)
    y = y * (0.99 / max(abs(y))) # Normalize

    # Calculate the number of samples for analysis
    sample_duration_ms = 1/fs * 1000 # duration of one sample in ms
    samples = int(DURATION / sample_duration_ms)

    fig = plt.figure()
    lowcep = fig.add_subplot(211)
    hicep = fig.add_subplot(212)
    for (i, color) in [('a', 'r'), ('e', 'g'), ('i', 'b'), ('o', 'c'), ('u', 'm')]:
        timestamp = TIMESTAMPS[i]

        starting_sample = int(timestamp / (1/fs))
        segment = y[starting_sample:starting_sample+samples+1]

        # Analyse
        Yfft = fft_hanning(segment)
        Ycep = cepstrum(Yfft)

        plot_cepstrum(lowcep, Ycep[1:FILTER+1], fs, color=color, label=i, linewidth=3)
        lowcep.set_xlabel('')
        plot_cepstrum(hicep, Ycep[FILTER:PLOT_CAP], fs, color=color, label=i, linewidth=1)

        # Find maximum
        Ycep_half = np.abs(Ycep[1:len(Ycep)//2+1])
        Ycep_half[0:FILTER] = 0
        Ycep_max = np.amax(Ycep_half)
        index = np.where(Ycep_half == Ycep_max)[0][0]

        # Mark the max
        hicep.scatter(1/fs*(index-FILTER+1), Ycep_max, marker='x', color=color, zorder=5)
        hicep.text(1/fs*(index-FILTER+1), Ycep_max, '{:0.1f} Hz'.format(fs/index))

    fig.suptitle('Summary for vowels.')

    lowcep.set_title("High frequency content (>{:0.0f} Hz)".format(fs/FILTER))
    lowcep.legend()

    hicep.set_title("Low frequency content (<{:0.0f} Hz)".format(fs/FILTER))
    hicep.legend()

    plt.show()
