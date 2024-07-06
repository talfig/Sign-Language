import numpy as np


def load_data_from_npz(filename):
    with np.load(filename) as data:
        images = np.array(data['data'])
        labels = np.array(data['labels'])
    return images, labels
