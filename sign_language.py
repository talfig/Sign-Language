import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.models as models
from torchvision.models import ResNet18_Weights

from CustomDataset import CustomDataset
from train_eval import train_model, evaluate_model
from upload_npz import load_data_from_npz

# Define your device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
filename = r'C:\Users\xbpow\Downloads\Sign_Language\dataset.npz'
pixels, labels = load_data_from_npz(filename)

# Count unique classes
NUM_CLASSES = len(np.unique(labels))

# Load pre-trained ResNet101 model
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# Modify the first convolutional layer to accept 1 input channel instead of 3
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

# Modify the fully connected layer to output the correct number of classes
num_fc = model.fc.in_features
model.fc = nn.Linear(num_fc, NUM_CLASSES)

# Move the model to the specified device
model = model.to(device)

epochs = 5  # number of single passes on the network

loss_fn = nn.CrossEntropyLoss(ignore_index=26)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

dataset = CustomDataset(pixels, labels, transform=transform)
dataloader = DataLoader(dataset, batch_size=200, shuffle=True)

if __name__ == '__main__':
    # Clear any cached GPU memory
    torch.cuda.empty_cache()

    train_model(model=model,
                dataloader=dataloader,
                epochs=epochs,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device)

    evaluate_model(model=model,
                   dataloader=dataloader,
                   device=device)

    name = 'model_weights.pth'
    torch.save(model.state_dict(), name)  # Saves the model weights and biases
