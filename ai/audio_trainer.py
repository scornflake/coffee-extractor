import os

import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio

neils_data_folder = "/Users/neil/development/ai/data series"


@tf.keras.saving.register_keras_serializable()
class ReduceMeanLayer(tf.keras.layers.Layer):
    def __init__(self, axis=0, **kwargs):
        super(ReduceMeanLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, input):
        return tf.math.reduce_mean(input, axis=self.axis)

    def get_config(self):
        return {"axis": self.axis}


# Goal: transfer train a model on recognition of background noise, first_crack, second_crack
class Trainer:
    @classmethod
    def audio_mapping(cls):
        return {'background_noise': 0, 'first_crack': 1, 'second_crack': 2}

    @classmethod
    def audio_target_for_label(cls, label: str) -> int:
        # if the key does not exist in self.audio_mapping, raise an error
        if label not in cls.audio_labels():
            raise ValueError(f"Label {label} is not in the audio mapping")

        return cls.audio_mapping()[label]

    @classmethod
    def audio_labels(cls):
        return list(cls.audio_mapping().keys())

    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.model = None
        self.yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
        self.yamnet_model = hub.load(self.yamnet_model_handle)
        class_map_path = self.yamnet_model.class_map_path().numpy().decode('utf-8')
        self.class_names = list(pd.read_csv(class_map_path)['display_name'])

    def train_audio_from_data_folder(self):
        # Get the filenames, labels
        dataset = self.create_audio_files_dataset()

        @tf.function
        def from_tuple_to_wav_label(filename, target):
            wav_mono = self.load_wav_16k_mono(filename)
            return wav_mono, target

        # TODO: Normalize the wave data?  Maybe do this after getting it going, to measure the impact

        # Convert to wave data
        dumpdataset(dataset, "initial", ["file", "label"])
        dataset = dataset.map(from_tuple_to_wav_label)
        # dumpdataset(dataset, "waves", ["wave", "label_index"])

        # Convert to embeddings
        all_data = self.from_wave_to_embeddings(dataset)

        # Now train
        self.perform_training(all_data)

    def create_audio_files_dataset(self):
        # Begin by reading the filenames from data_folder, where there are subfolders for each label
        # We want to get a list of tuples, (filename, label)
        filenames = []
        target_indexes = []
        for label in self.audio_labels():
            path_for_labelled_data = os.path.join(self.data_folder, label)
            target_index = self.audio_target_for_label(label)
            print(f"Use: {label}, target: {target_index}")
            for file in os.listdir(path_for_labelled_data):
                full_path = os.path.join(path_for_labelled_data, file)
                filenames.append(full_path)
                target_indexes.append(target_index)

        dataset = tf.data.Dataset.from_tensor_slices((filenames, target_indexes))
        # dumpdataset(dataset, "initial", ["wave", "target"])
        return dataset

    # Utility functions for loading audio files and making sure the sample rate is correct.
    @tf.function
    def load_wav_16k_mono(self, filename):
        """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
        file_contents = tf.io.read_file(filename)
        wav, sample_rate = tf.audio.decode_wav(
            file_contents,
            desired_channels=1)
        wav = tf.squeeze(wav, axis=-1)
        sample_rate = tf.cast(sample_rate, dtype=tf.int64)
        wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
        return wav

    def perform_training(self, dataset_with_audio_embeddings):
        dataset_size = len(list(dataset_with_audio_embeddings))

        # We need a tf dataset, of form: (extraction, label)
        # We then take 20% of the dataset as a test set, 20% as validation, and remainder as a training set
        test_size = int(dataset_size * 0.2)
        val_size = int(dataset_size * 0.2)
        training_size = dataset_size - test_size - val_size

        print(f"Data set has total size: {dataset_size}")
        print(f"Training set will have size: {training_size}")
        print(f"Test set will have size: {test_size}")
        print(f"Validation set will have size: {val_size}")

        cached = dataset_with_audio_embeddings.cache().shuffle(1000)

        # dumpdataset(cached_without_target, "Cached before split", ["embeddings", "target"])

        # We want a mix of training, test and validation data
        test_data = cached.take(test_size).batch(32).prefetch(tf.data.AUTOTUNE)
        val_ds = cached.skip(test_size).take(val_size).batch(32).prefetch(tf.data.AUTOTUNE)
        training_data = cached.skip(test_size + val_size).batch(32).prefetch(tf.data.AUTOTUNE)

        # dumpdataset(training_data, "Training", ["embeddings", "target"])

        # Model. 1024 input, 512 hidden, and the number of classes as the output
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(1024), dtype=tf.float32, name='input_embedding'),
            # A rectified linear unit (ReLU) is an activation function that introduces the property of nonlinearity to a deep learning model and solves the vanishing gradients issue
            tf.keras.layers.Dense(512, activation='relu', name='hidden_layer'),
            tf.keras.layers.Dense(len(self.audio_labels()), name='output')
        ], name='audio_model')

        self.model.summary()

        self.model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           optimizer="adam",
                           metrics=['accuracy'])
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                    patience=3,
                                                    restore_best_weights=True)

        history = self.model.fit(training_data,
                                 epochs=20,
                                 validation_data=val_ds,
                                 callbacks=callback)

        print("Training complete. Testing on test_data set...")
        loss, accuracy = self.model.evaluate(test_data)

        print("Loss: ", loss)
        print("Accuracy: ", accuracy)

    def test_against_wav_file(self, wav_file_name):
        wav = self.load_wav_16k_mono(wav_file_name)
        embeddings = self._extract_embedding(wav, 0)
        result = self.model(embeddings)
        print(f"Result for file: {wav_file_name}\nResult: {result}")

    def save_model(self):
        """Gives a model that can take audio directly, and classify it.  Saves it to a file."""
        # Check we have a model, if not throw
        if self.model is None:
            raise ValueError("No model to save")

        model_filename = "audio_model.keras"
        print(f"Saving model to: {model_filename}")

        input_segment = tf.keras.layers.Input(shape=(), dtype=tf.float32, name='audio')
        embedding_extraction_layer = hub.KerasLayer(self.yamnet_model_handle,
                                                    trainable=False, name='yamnet')
        _, embeddings_output, _ = embedding_extraction_layer(input_segment)
        serving_outputs = self.model(embeddings_output)
        serving_outputs = ReduceMeanLayer(axis=0, name='classifier')(serving_outputs)
        serving_model = tf.keras.Model(input_segment, serving_outputs)
        serving_model.save(model_filename)

    def from_wave_to_embeddings(self, dataset):
        def conversion_func(wave, label):
            return self._extract_embedding(wave, label)

        return dataset.map(conversion_func).unbatch()

    # applies the embedding extraction model to a wav data
    # The purpose of these operations in the context of the extract_embedding function is to associate each embedding extracted from the
    # wav_data with the corresponding label.
    def _extract_embedding(self, wav_data, label):
        # run YAMNet to extract embedding from the wav data
        scores, embeddings, spectrogram = self.yamnet_model(wav_data)
        num_embeddings = tf.shape(embeddings)[0]
        print(f"Generated {num_embeddings} embeddings for: {label}")
        return (embeddings,
                tf.repeat(label, num_embeddings))


def dumpdataset(ds, label: str, labels: [str]):
    # Print the contents of the data set
    print(f"Dataset {label}, spec: {ds.element_spec}")
    if len(labels) == 3:
        for x, y, z in ds:
            print(f"{labels[0]}: {x}, {labels[1]}: {y}, {labels[2]}: {z}")
    elif len(labels) == 2:
        for x, y in ds:
            print(f"{labels[0]}: {x}, {labels[1]}: {y}")


if __name__ == "__main__":
    trainer = Trainer(neils_data_folder)
    trainer.train_audio_from_data_folder()
    trainer.save_model()

    print("Finished")
