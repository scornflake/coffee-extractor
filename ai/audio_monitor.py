import argparse
import asyncio
import os
import sys

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import sounddevice as sd
import tensorflow_io as tfio

from ai.audio_trainer import ReduceMeanLayer, Trainer


async def inputstream_generator(input_device: int | str, channels=1, **kwargs):
    """Generator that yields blocks of input data as NumPy arrays."""
    q_in = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def callback(indata, frame_count, time_info, status):
        loop.call_soon_threadsafe(q_in.put_nowait, (indata.copy(), status))

    stream = sd.InputStream(device=input_device, callback=callback, channels=channels, **kwargs)
    with stream:
        while True:
            indata, status = await q_in.get()
            yield indata, status


class AudioModel:
    def __init__(self, model_filename, input_device: str = None):
        self.sample_rate = None
        self.input_device = None

        self.model = tf.keras.models.load_model(model_filename, custom_objects={'KerasLayer': hub.KerasLayer,
                                                                                'ReduceMeanLayer': ReduceMeanLayer})
        self.model.summary()

    def monitor_audio(self, input_device: int | str = None):
        # Enumerate all devices
        if input_device is None:
            print(sd.query_devices())
            sys.exit(-1)

        self.input_device = input_device

        # Query sd and get the sample rate for this device
        query_devices = sd.query_devices(self.input_device, 'input')
        if not query_devices:
            print(f"Device {self.input_device} not found")
            sys.exit(-1)
        self.sample_rate = query_devices['default_samplerate']
        print("Sample rate is: ", self.sample_rate)

        # Monitor the audio device, and continually try to guess what we have
        # If we have a guess, print it out
        try:
            asyncio.run(self.get_input_audio(blocksize=1024))
        except KeyboardInterrupt:
            sys.exit('\nInterrupted by user')

    async def get_input_audio(self, blocksize=1024):
        try:
            await asyncio.wait_for(self.classify_raw_audio_from_device(), timeout=500)
        except asyncio.TimeoutError:
            print("Right. That's enough...")
            sys.exit(1)

    def classify_audio_via_model(self, audio):
        # Given a piece of audio, classify it
        result = self.model(audio)
        class_probabilities = tf.nn.softmax(result, axis=-1)
        return class_probabilities

    async def classify_raw_audio_from_device(self, **kwargs):
        rate = tf.cast(self.sample_rate, tf.int64)
        temporary_q = np.array([], dtype=np.float32)
        async for indata, status in inputstream_generator(input_device=self.input_device, **kwargs):
            if status:
                print("Status: ", status)
            else:
                # Add to a separate q until we have about 1s of audio
                temporary_q = np.append(temporary_q, indata)
                if len(temporary_q) > self.sample_rate:
                    # this is raw audio. Need to make sure it's 1ch 16Khz
                    # the audio should be float32
                    audio_data = tf.convert_to_tensor(temporary_q)
                    temporary_q = np.array([], dtype=np.float32)
                    wav = tfio.audio.resample(audio_data, rate_in=rate, rate_out=16000)

                    # audio_data = tf.io.decode_raw(indata, out_type=tf.float32)
                    result = self.classify_audio_via_model(wav)

                    # The result is a tensor, that is the probability that this audio is a certain class
                    inferred_class, top_score = self.get_most_likely(result)
                    print(f"Primary sound is: {inferred_class} with a probability of {top_score}. Results: {result}")

                # else:
                # print(f"Temporary q is {len(temporary_q)} long")

    def get_most_likely(self, result) -> (str, float):
        top_class = tf.argmax(result)
        inferred_class = Trainer.audio_labels()[top_class]
        top_score = result[top_class]
        return inferred_class, top_score

    def classify(self, filename):
        # Check it exists
        if not os.path.exists(filename):
            print(f"File {filename} does not exist")
            return

        # Load the file
        audio_data = tf.io.read_file(filename)
        wav, sample_rate = tf.audio.decode_wav(audio_data, desired_channels=1)
        wav = tf.squeeze(wav, axis=-1)
        sample_rate = tf.cast(sample_rate, dtype=tf.int64)
        wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)

        result = self.classify_audio_via_model(wav)
        top_class = tf.argmax(result)
        inferred_class = Trainer.audio_labels()[top_class]
        top_score = result[top_class]

        # print(f"Result: {result}")
        print(f"Primary sound is: {inferred_class} with a probability of {top_score}")


def process_args():
    parser = argparse.ArgumentParser(description='Run training')
    parser.add_argument('data', type=str, help='Data series folder, where the training data is')
    args = parser.parse_args()

    # Check data series folder exists on disk
    if not os.path.exists(args.data):
        print(f"Data series folder {args.data} does not exist")
        exit(1)

    return args


if __name__ == "__main__":
    args = process_args()
    data_series_folder = args.data

    monitor = AudioModel("audio_model.keras")
    background_noise_file = os.path.join(data_series_folder, "background_noise", "background_noise_roast_1.wav")
    first_crack_file = os.path.join(data_series_folder, "first_crack", "first_crack_roast_1.wav")
    monitor.classify(background_noise_file)
    monitor.classify(first_crack_file)

    monitor.monitor_audio(input_device="iMac Microphone")
