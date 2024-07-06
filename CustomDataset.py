from torch.utils.data import Dataset
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, pixels, labels, transform=None):
        self.pixels = pixels
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.pixels[idx].reshape(200, 200)
        img = Image.fromarray(img.astype('uint8'), 'L')  # Convert numpy array to PIL Image in grayscale mode
        label = self.labels[idx].astype('int64')  # Convert labels to Long tensor

        if self.transform:
            img = self.transform(img)

        return img, label
