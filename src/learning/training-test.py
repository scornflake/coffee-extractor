import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

(xtrain , ytrain) , (xtest , ytest) = mnist.load_data()
xtrain.shape
xtest.shape

num_labels = len(np.unique(ytrain))
num_labels
