import tensorflow as tf
import os.path

from scipy import signal
from scipy.signal import butter, lfilter

from audio.audio_utils import load_wav_resample_to_16k_mono, load_wav_native_mono


def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=True)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y




def notch_filter(data, notch_freq=100, quality_factor=30, fs=16000):
    # Create/view notch filter
    samp_freq = fs  # 1000  # Sample frequency (Hz)
    b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, samp_freq)
    freq, h = signal.freqz(b_notch, a_notch, fs=samp_freq)
    plt.figure('filter')
    plt.plot(freq, 20 * np.log10(abs(h)))

    # Create/view signal that is a mixture of two frequencies
    f1 = 17
    f2 = 60
    t = np.linspace(0.0, 1, 1_000)
    y_pure = np.sin(f1 * 2.0 * np.pi * t) + np.sin(f2 * 2.0 * np.pi * t)
    plt.figure('result')
    plt.subplot(211)
    plt.plot(t, y_pure, color='r')

    # apply notch filter to signal
    y_notched = signal.filtfilt(b_notch, a_notch, y_pure)

    # plot notch-filtered version of signal
    plt.subplot(212)
    plt.plot(t, y_notched, color='r')

    return y_notched


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import freqz

    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 16000.0
    fs = 44100.0
    lowcut = 500.0
    highcut = 3000.0

    # Load a roast sound
    audio_path = "/Users/neil/Documents/Coffee/data-series"
    fqn_to_roast = os.path.join(audio_path, "first_crack/first_crack_roast_1.wav")
    wav, sample_rate = load_wav_native_mono(fqn_to_roast)

    # So we can hear it at 16Khz, save

    # convert tensor to nmpy array
    wav = wav.numpy()
    print(f"have numpy source: {wav}, type: {type(wav)}, shape: {wav.shape}, data type: {wav.dtype}")
    # convert it back to a tensor
    # wav = tf.convert_to_tensor(wav)
    # print(f"have: {wav}, type: {type(wav)}, shape: {wav.shape}")

    # Filter a noisy signal.
    x = wav

    # Plot the waveform
    plt.figure(1)
    plt.title('Waveform')
    plt.plot(x)
    plt.xlabel('time (samples)')
    plt.ylabel('amplitude')
    plt.grid(True)

    # y = butter_highpass_filter(x, 4000, fs, order=6)
    y = butter_bandpass_filter(x, lowcut, highcut, fs, order=6)
    # y = notch_filter(x, 100, 30, fs)

    # Plot Y, the filtered signal
    plt.subplot(1, 1, 1)
    plt.title('Filtered Signal')
    plt.plot(y)
    plt.xlabel('time (samples)')
    plt.ylabel('amplitude')

    plt.show()

    # Save filtered as wave file
    # print type, shape and data type of y
    # convert from float64 to float32
    y = y.astype(np.float32)
    print(f"have filtered: {y}, type: {type(y)}, shape: {y.shape}, data type: {y.dtype}")

    # convert y (audio samples) to a 2D tensor, with the audio data, and 1 channel
    y = np.expand_dims(y, axis=1)

    print(f"have filtered: {y}, type: {type(y)}, shape: {y.shape}, data type: {y.dtype}")
    filtered_wav = tf.convert_to_tensor(y)
    normalize_wav = filtered_wav  # normalize(filtered_wav)

    data_string = tf.audio.encode_wav(normalize_wav, int(fs))
    with open("first_crack_roast_1_filtered.wav", "wb") as f:
        f.write(data_string.numpy())
