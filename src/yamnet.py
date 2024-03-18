import os

from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio

from audio.audio_utils import load_wav_resample_to_16k_mono

# Get the model
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

# Load up the .wav
testing_wav_file_name = tf.keras.utils.get_file('miaow_16k.wav',
                                                'https://storage.googleapis.com/audioset/miaow_16k.wav',
                                                cache_dir='../',
                                                cache_subdir='test_data')

# Convert to 1ch 16kHz
testing_wav_data = load_wav_resample_to_16k_mono(testing_wav_file_name)

# Load the classes able to be recognized from the model
class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
class_names = list(pd.read_csv(class_map_path)['display_name'])

for name in class_names[:20]:
    print(name)
print('...')

# Run inference on the test wav file
# scores, embeddings, spectrogram = yamnet_model(testing_wav_data)
# class_scores = tf.reduce_mean(scores, axis=0)
#
# # Print the scores, ordered by score (higher should be first)
# top_n = 10
# top_class_indices = tf.argsort(class_scores, direction='DESCENDING')
# for i in range(top_n):
#     print(f'{class_names[top_class_indices[i]]:<30}: {class_scores[top_class_indices[i]]}')
#
# # Get the top class by score
# top_class = tf.math.argmax(class_scores)
# inferred_class = class_names[top_class]
# print(f'The main sound is: {inferred_class}')
# print(f'The embeddings shape: {embeddings.shape}')

# Make better
# A whole bunch of sampled audio
# Think it's already normalized, 16khz?
_ = tf.keras.utils.get_file('esc-50.zip',
                            'https://github.com/karoldvl/ESC-50/archive/master.zip',
                            cache_dir='../',
                            cache_subdir='datasets',
                            extract=True)

# This part grabs just cat/dog from the list of all audio
esc50_csv = './datasets/ESC-50-master/meta/esc50.csv'
base_data_path = '../data/datasets/ESC-50-master/audio/'

pd_data = pd.read_csv(esc50_csv)
pd_data.head()

my_classes = ['dog', 'cat']
map_class_to_id = {'dog': 0, 'cat': 1}

filtered_pd = pd_data[pd_data.category.isin(my_classes)]

class_id = filtered_pd['category'].apply(lambda name: map_class_to_id[name])
filtered_pd = filtered_pd.assign(target=class_id)

full_path = filtered_pd['filename'].apply(lambda row: os.path.join(base_data_path, row))
filtered_pd = filtered_pd.assign(filename=full_path)

print("Filtered dataset (just cats/digs from esc50):")
filtered_pd.head(10)

filenames = filtered_pd['filename']
targets = filtered_pd['target']  # aka, category
folds = filtered_pd['fold']


def dumpdataset(ds, label: str, labels: [str]):
    # Print the contents of the data set
    print(f"Dataset {label}, spec: {ds.element_spec}")
    if len(labels) == 3:
        for x, y, z in ds:
            print(f"{labels[0]}: {x}, {labels[1]}: {y}, {labels[2]}: {z}")
    elif len(labels) == 2:
        for x, y in ds:
            print(f"{labels[0]}: {x}, {labels[1]}: {y}")


# Folds: Instead of dividing your dataset into just two parts, a training set and a testing set, it involves dividing the dataset into K subsets or "folds."
# K represents the number of partitions you want to create. The most common choice for K is 5 or 10
# https://www.theknowledgeacademy.com/blog/k-fold-cross-validation-in-machine-learning/#:~:text=Instead%20of%20dividing%20your%20dataset,dataset's%20size%20and%20specific%20needs.


# Print the top 10 again, this time with filename, target and fold
# print("Top 10 again with filename, target and fold:")
# for i in range(10):
#     print(f'{filenames[i]}: {targets[i]}: {folds[i]}')

# filename: what it says
# target: some number?  or is target, the label?
# fold: like a group. all sounds in the same fold, are the same sound (different recordings of it)

main_ds = tf.data.Dataset.from_tensor_slices((filenames, targets, folds))
main_ds.element_spec


def load_wav_for_map(filename, label, fold):
    return load_wav_resample_to_16k_mono(filename), label, fold


# Map the dataset. Convert from filenames, to the actual data. First column is now the .wav data, followed by label, and fold
dumpdataset(main_ds, "initial construction", ["filename", "target", "fold"])
main_ds = main_ds.map(load_wav_for_map)
main_ds.element_spec


# print(f"Wave DataSet: {main_ds}")
# dumpdataset(main_ds, "Wave Data", ["wav", "target", "fold"])

# applies the embedding extraction model to a wav data
# The purpose of these operations in the context of the extract_embedding function is to associate each embedding extracted from the
# wav_data with the corresponding label and fold. This is done because the YAMNet model might produce multiple embeddings for a single audio file (or wav_data),
# and we want to keep track of the label and fold associated with each of these embeddings.
def extract_embedding(wav_data, label, fold):
    ''' run YAMNet to extract embedding from the wav data '''
    scores, embeddings, spectrogram = yamnet_model(wav_data)
    num_embeddings = tf.shape(embeddings)[0]
    return (embeddings,
            tf.repeat(label, num_embeddings),
            tf.repeat(fold, num_embeddings))


# extract embeddings for each of the wave files.
# because we get multiple embeddings for a single wav file, we copy the label/fold across each of the embeddings

main_ds = main_ds.map(extract_embedding).unbatch()
main_ds.element_spec
dumpdataset(main_ds, "After Embedding", ["Embeddings", "label", "fold"])
# seems like (array, int int)

# Now we train.
# Take the first 80% of the data for training
# Take the next 10% for validation
# Take the last 10% for testing

cached_ds = main_ds.cache()
train_ds = cached_ds.filter(lambda embedding, label, fold: fold < 4)
val_ds = cached_ds.filter(lambda embedding, label, fold: fold == 4)
test_ds = cached_ds.filter(lambda embedding, label, fold: fold == 5)

# remove the folds column now that it's not needed anymore
remove_fold_column = lambda embedding, label, fold: (embedding, label)

# we're left with the embedding for the audio and the label
train_ds = train_ds.map(remove_fold_column)
val_ds = val_ds.map(remove_fold_column)
test_ds = test_ds.map(remove_fold_column)
dumpdataset(train_ds, "Training After Remove Fold", ["embeddings", "labels"])
print(f"after remove fold: {train_ds}")

# Train in batches of 32 (32 sounds at a time)
train_ds = train_ds.cache().shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)

dumpdataset(train_ds, "Batched Training", ["embeddings", "labels"])

# Model. 1024 input, 512 hidden, and the number of classes as the output
my_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1024), dtype=tf.float32,
                          name='input_embedding'),
    # A rectified linear unit (ReLU) is an activation function that introduces the property of nonlinearity to a deep learning model and solves the vanishing gradients issue
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(len(my_classes))
], name='my_model')

my_model.summary()

my_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 optimizer="adam",
                 metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                            patience=3,
                                            restore_best_weights=True)

history = my_model.fit(train_ds,
                       epochs=20,
                       validation_data=val_ds,
                       callbacks=callback)

loss, accuracy = my_model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)
