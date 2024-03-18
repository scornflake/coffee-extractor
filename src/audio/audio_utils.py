import librosa
import tensorflow as tf
import tensorflow_io as tfio


@tf.function
def load_wav_resample_to_16k_mono(filename):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    wav, sample_rate = load_wav_native_mono(filename)
    if sample_rate != 16000:
        wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav


@tf.function
def load_wav_native_mono(filename) -> (tf.Tensor, tf.Tensor):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
        file_contents,
        desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    return wav, sample_rate


@tf.function
def load_wav_filter_then_reduce_to_16khz(filename):
    wav, sample_rate = extract_percussive(filename, margin=4)
    if sample_rate != 16000:
        wav = librosa.resample(wav, orig_sr=sample_rate, target_sr=16000)
    # convert to tensor
    wav = tf.convert_to_tensor(wav)
    return wav


def save_audio_data_to_wav(filename, data, sample_rate):
    import soundfile as sf
    sf.write(filename, data, sample_rate)


def extract_percussive(filename, margin=16):
    y, sr = librosa.load(filename)
    y = librosa.to_mono(y)
    # Compute the short-time Fourier transform of y
    D = librosa.stft(y)
    # Decompose D into harmonic and percussive components
    d_harmonic, d_percussive = librosa.decompose.hpss(D, margin=margin)
    y_percussive = librosa.istft(d_percussive, length=len(y))
    return y_percussive, sr
