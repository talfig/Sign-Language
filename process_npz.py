import os
import numpy as np
from PIL import Image


def process_images_to_npz(image_folder, output_npz):
    labels = []
    data = []
    i = 0

    for class_folder in os.listdir(image_folder):
        class_path = os.path.join(image_folder, class_folder)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                if filename.endswith('.png') or filename.endswith('.jpg'):
                    img_path = os.path.join(class_path, filename)
                    img = Image.open(img_path).convert('L')
                    labels.append(i)
                    img_array = np.array(img).flatten()
                    data.append(img_array)

        print(f"Processed folder {i}: {class_folder}")
        i += 1

    data = np.array(data)
    labels = np.array(labels)

    # Save data and labels to an NPZ file with compression
    np.savez_compressed(output_npz, data=data, labels=labels)
    print(f"Data saved to {output_npz}")


image_folder = r'C:\Users\xbpow\Downloads\Sign_Language\archive\asl_alphabet_train\asl_alphabet_train'
output_npz = 'dataset.npz'  # Output NPZ file with compression
process_images_to_npz(image_folder, output_npz)
