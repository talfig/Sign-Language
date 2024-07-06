import numpy as np


def load_data_from_npz(filename):
    with np.load(filename) as data:
        images = data['data']
        labels = data['labels']
    return images, labels
