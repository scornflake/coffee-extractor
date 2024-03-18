import os

from audio.audio_utils import load_wav_resample_to_16k_mono, load_wav_filter_then_reduce_to_16khz

audio_path = "/Users/neil/Documents/Coffee/data-series"
fqn_to_roast = os.path.join(audio_path, "first_crack/first_crack_roast_1.wav")


tf_audio = load_wav_resample_to_16k_mono(fqn_to_roast)
print(f"tf_audio: {tf_audio}, type: {type(tf_audio)}, shape: {tf_audio.shape}, data type: {tf_audio.dtype}")

librosa_audio = load_wav_filter_then_reduce_to_16khz(fqn_to_roast)
print(f"librosa_audio: {librosa_audio}, type: {type(librosa_audio)}, shape: {librosa_audio.shape}, data type: {librosa_audio.dtype}")
