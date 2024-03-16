import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers

# read from an excel file (xlsx)
training_data = pd.read_excel("./sample-coffee-data.xlsx",
                              names=["RoastNumber", "FirstCrackStart", "FirstCrackEnd", "SecondCrackStart",
                                     "SecondCrackEnd", "Rating"])

# dump the first few rows of the dataframe
print(training_data.head())

# display the type of columns in the dataframe
print(training_data.dtypes)

normalize = layers.Normalization()

features = training_data.copy()

# remove the header from the data table
features = features[1:]

# remove the first column from features
labels = features.pop('RoastNumber')

features = np.array(features)
normalize.adapt(features)

final_model = tf.keras.Sequential([
    normalize,
    layers.Dense(64),
    layers.Dense(1)
])

final_model.compile(loss=tf.keras.losses.MeanSquaredError(),
                    optimizer=tf.keras.optimizers.Adam())

final_model.fit(features, labels, epochs=20)

# now test the model -
