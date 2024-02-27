import asyncio
import sys

import tensorflow as tf
import tensorflow_hub as hub
import sounddevice as sd
import tensorflow_io as tfio

from ai.audio_trainer import ReduceMeanLayer


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
        self.keep_running = False

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

        self.model = tf.keras.models.load_model(model_filename, custom_objects={'KerasLayer': hub.KerasLayer,
                                                                                'ReduceMeanLayer': ReduceMeanLayer})
        self.model.summary()

    def monitor_audio(self, device):
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
        return self.model(audio)

    async def classify_raw_audio_from_device(self, **kwargs):
        async for indata, status in inputstream_generator(input_device=self.input_device, **kwargs):
            if status:
                print("Status: ", status)
            else:
                # this is raw audio. Need to make sure it's 1ch 16Khz
                # the audio should be float32
                audio_data = tf.convert_to_tensor(indata)
                rate = tf.cast(self.sample_rate, tf.int64)
                wav = tfio.audio.resample(audio_data, rate_in=rate, rate_out=16000)

                # audio_data = tf.io.decode_raw(indata, out_type=tf.float32)
                result = self.classify_audio_via_model(wav)
                # The result is a tensor, that is the probability that this audio is a certain class
                print(f"Result: {result}")


if __name__ == "__main__":
    monitor = AudioModel("audio_model.keras", input_device="iMac Microphone")
    monitor.monitor_audio("default")
