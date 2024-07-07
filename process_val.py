import os
import numpy as np
from PIL import Image


def val_to_npz(image_folder, output_npz):
    labels = []
    data = []
    i = 0

    for filename in os.listdir(image_folder):
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            img_path = os.path.join(image_folder, filename)  # Fixed the path construction
            img = Image.open(img_path).convert('L')  # Convert image to grayscale
            labels.append(i)
            img_array = np.array(img).flatten()  # Flatten the image to a 1D array
            data.append(img_array)

        print(f"Processed file {i}: {filename}")  # Changed to 'file' for clarity
        i += 1

    data = np.array(data)
    labels = np.array(labels)

    # Save data and labels to an NPZ file with compression
    np.savez_compressed(output_npz, data=data, labels=labels)
    print(f"Data saved to {output_npz}")


image_folder = r'C:\Users\xbpow\Downloads\Sign_Language\ASL\asl_alphabet_test\asl_alphabet_test'
output_npz = 'validation.npz'  # Output NPZ file with compression
val_to_npz(image_folder, output_npz)
