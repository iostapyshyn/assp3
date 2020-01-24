#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
plt.style.use('ggplot')

FILENAME="audio/a.wav"
DURATION=100 # ms

def fft_hanning(y):
    # Apply Hanning window
    y *= np.hanning(len(y))
    Y = np.fft.fft(y)
    return Y

# Compute cepstrum coefficients
def cepstrum(Y):
    logY = np.log(np.abs(Y))
    return np.fft.ifft(logY)

# Plot the positive half of fft, with proper frequency scale
def plot_fft(fftfig, Yfft, fs, **kwargs):
    # Compute sample frequencies and amplitude spectrum
    freq = np.fft.fftfreq(len(Yfft), d=1/fs)[:len(Yfft)//2]
    ampl = abs(Yfft)[:len(Yfft)//2] ** 2

    fftfig.set_ylabel('Amplitude spectrum')
    fftfig.set_xlabel('Frequency [Hz]')
    fftfig.xaxis.set_label_position('top')
    fftfig.yaxis.set_label_position('right')
    fftfig.loglog(freq, ampl, linewidth=3, **kwargs)

# Plot the positive part of CEPSTRUM with sample time (quefrency) as x axis
def plot_cepstrum(cepfig, Ycep, fs, **kwargs):
    quefrency = np.arange(len(Ycep))/fs

    cepfig.set_ylabel('CEPSTRUM coefficients')
    cepfig.set_xlabel('Quefrency [s]')
    cepfig.yaxis.set_label_position('right')
    cepfig.plot(quefrency[1:len(quefrency)//2+1], abs(Ycep)[1:len(Ycep)//2+1], **kwargs) # Skipping the first element

if __name__ == '__main__':
    ### A.1 Computation of the CEPSTRUM coefficients
    # Read and normalize the audio file
    fs, y = scipy.io.wavfile.read(FILENAME)
    y = y * (0.99 / max(abs(y))) # Normalize

    # Calculate the number of samples for analysis
    sample_duration_ms = 1/fs * 1000 # duration of one sample in ms
    samples = int(DURATION / sample_duration_ms)

    # Plot the signal
    time = np.arange((1/fs)*512, (1/fs)*(512+samples), (1/fs)) * 1000
    plt.title("Part of the signal being analysed")
    plt.xlabel("Time [ms]")
    plt.ylabel("Amplitude")
    plt.plot(time, y[512:512+samples])
    plt.show()

    # Compute the FFT
    Yfft = fft_hanning(y[512:512+samples])
    # Compute the CEPSTRUM coefficients
    Ycep = cepstrum(Yfft)

    # Show the plot of the linear magnitude
    fig = plt.figure()
    fig.suptitle("The whole spectrum.")
    fftfig = fig.add_subplot(211)
    cepfig = fig.add_subplot(212)

    plot_fft(fftfig, Yfft, fs)
    plot_cepstrum(cepfig, Ycep, fs)

    plt.show()

    ### A.3 Liftering
    for liftering in [20, 40, 5]:
        cep_filtered = Ycep.copy()
        cep_filtered[liftering+1:len(cep_filtered)-liftering-1] = 0

        fft_filtered = fft_hanning(cep_filtered)

        # Plots
        fig = plt.figure()

        fig.suptitle(str(liftering) + " samples liftering.")
        fftfig = fig.add_subplot(211)
        cepfig = fig.add_subplot(212)

        plot_fft(fftfig, fft_filtered, fs)
        plot_cepstrum(cepfig, cep_filtered[1:liftering+1], fs)
        plt.show()

    # Back to the whole spectrum
    # Set first 19 coefficients to 0
    Ycep_half = np.abs(Ycep[1:len(Ycep)//2+1])
    Ycep_half[0:20] = 0

    # Find maximum
    Ycep_max = np.amax(Ycep_half)
    index = np.where(Ycep_half == Ycep_max)[0][0]

    print("Max coefficient: {} at quefrency {} [s] or frequency {} [Hz]."
          .format(Ycep_max, 1/fs*index, fs/index))

    fig = plt.figure()
    fig.suptitle("The whole spectrum again.")
    fftfig = fig.add_subplot(211)
    cepfig = fig.add_subplot(212)

    plot_fft(fftfig, Yfft, fs)
    plot_cepstrum(cepfig, Ycep, fs)

    # Mark the max
    cepfig.scatter(1/fs*index, Ycep_max, marker='x', color='b', zorder=5)

    plt.show()
