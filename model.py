import torch.nn as nn
import torch.nn.functional as F
import torch

class PlantVillageCNN(nn.Module):
    def __init__(self, n_classes=38): # Changed default to 38
        super(PlantVillageCNN, self).__init__()
        
        # 1. Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1)
        
        # Pooling Layer (Halves the height and width at each step)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        
        # 2. Linear/Dense Layers
        # Input size (224x224) downsampled 5 times: 224 -> 112 -> 56 -> 28 -> 14 -> 7
        # Final spatial size is 7x7. Channel depth is 64.
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=n_classes) # n_classes = 38

    def forward(self, x):
        # Input: [Batch_Size, 3, 224, 224]
        x = self.pool(F.relu(self.conv1(x))) # State: [Batch_Size, 32, 112, 112]
        x = self.pool(F.relu(self.conv2(x))) # State: [Batch_Size, 64, 56, 56]
        x = self.pool(F.relu(self.conv3(x))) # State: [Batch_Size, 64, 28, 28]
        x = self.pool(F.relu(self.conv4(x))) # State: [Batch_Size, 64, 14, 14]
        x = self.pool(F.relu(self.conv5(x))) # State: [Batch_Size, 64, 7, 7]
        
        # Flatten: [Batch_Size, 64, 7, 7] -> [Batch_Size, 64 * 7 * 7]
        x = torch.flatten(x, start_dim=1)
        
        # Dense Layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # Returns raw logits for 38 classes
        
        return x
    