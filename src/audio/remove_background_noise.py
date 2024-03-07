import argparse
import os

from maad.util import plot2d, power2dB, dB2power
from maad.sound import (load, spectrogram, write,
                        remove_background, median_equalizer,
                        remove_background_morpho,
                        remove_background_along_axis, sharpness)
import numpy as np

from timeit import default_timer as timer

import matplotlib.pyplot as plt

from scipy.io import wavfile
import noisereduce as nr

def process_args():
    parser = argparse.ArgumentParser(description='Run training')
    parser.add_argument('data', type=str, help='Data series folder, where the training data is')
    args = parser.parse_args()

    # Check data series folder exists on disk
    if not os.path.exists(args.data):
        print(f"Data series folder {args.data} does not exist")
        exit(1)

    return args


args = process_args()
data_series_folder = args.data
first_crack_file = os.path.join(data_series_folder, "first_crack", "first_crack_roast_1.wav")

rate, data = wavfile.read(first_crack_file)
reduced_noise = nr.reduce_noise(y=data, sr=rate)
wavfile.write('first_crack_roast_1_no_background.wav', rate, reduced_noise)

# samples, sample_rate = load(first_crack_file)
# Sxx, tn, fn, extents = spectrogram(samples, sample_rate, fcrop=[0, 20000], tcrop=[0, 10])
# Sxx_db = power2dB(Sxx, db_range=96) + 96
# plot2d(Sxx_db, title='Spectrogram', extent=extents, vmin=np.median(Sxx_db), vmax=np.max(Sxx_db))
# plt.show()
#
# # Remove background noise
# Sxx_db_no_noise, noise_profile1, _ = remove_background(Sxx_db)
# plot2d(Sxx_db_no_noise, title='Spectrogram without background noise', extent=extents, vmin=np.median(Sxx_db_no_noise), vmax=np.max(Sxx_db_no_noise))
# plt.show()
#
# # Save as a wav file we can playback
# power = dB2power(Sxx_db_no_noise)
# write('first_crack_roast_1_no_background.wav', sample_rate, power, 32)
