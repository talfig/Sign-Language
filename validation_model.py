import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
from torchvision.models import ResNet18_Weights

from CustomDataset import CustomDataset
from train_evaluate import evaluate_model, predict_and_display
from upload_npz import load_data_from_npz

# Define your device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
filename = r'C:\Users\xbpow\Downloads\Sign_Language\validation.npz'
pixels, labels = load_data_from_npz(filename)

# Load the state dictionary from the file
weight_path = r'C:\Users\xbpow\Downloads\Sign_Language\model_weights.pth'
state_dict = torch.load(weight_path, map_location=torch.device(device))

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

# Load the state dictionary into the model
model.load_state_dict(state_dict)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
])

dataset = CustomDataset(pixels, labels, transform=transform)
dataloader = DataLoader(dataset, batch_size=200, shuffle=True)

evaluate_model(model=model,
               dataloader=dataloader,
               device=device)

predict_and_display(model, dataloader, device)
