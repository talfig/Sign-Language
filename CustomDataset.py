from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, pixels, labels, transform=None):
        self.pixels = pixels
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.pixels[idx].reshape(200, 200).astype('float32')
        label = self.labels[idx].astype('int64')

        if self.transform:
            img = self.transform(img)

        return img, label
