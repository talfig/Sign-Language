import numpy as np
import pandas as pd
from upload_npz import load_data_from_npz


# Load data
def convert_to_csv(filename):
    pixels, labels = load_data_from_npz(filename)

    # Add labels as the first column
    pixels_with_labels = np.hstack((labels.reshape(-1, 1), pixels))  # Horizontal stack
    columns = ['label'] + [f'pixel_{i}' for i in range(pixels.shape[1])]

    # Convert to DataFrame
    df = pd.DataFrame(pixels_with_labels, columns=columns)

    # Save to CSV
    df.to_csv('pixels_with_labels.csv', index=False)
