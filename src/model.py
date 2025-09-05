import torch
import torch.nn as nn
import torch.nn.functional as F

class SoxDegradationClassifier(nn.Module):
    """
    A CNN-based model to predict a sequence of sox effects and their parameters.
    """
    def __init__(self, n_channels, output_size):
        super().__init__()

        # --- CNN Backbone ---
        # This is a simplified version of the ResNet-ish model from the original notebook.
        self.n_channels = n_channels
        self.conv1 = nn.Conv2d(in_channels=self.n_channels, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)

        # --- Fully Connected Head ---
        # Calculate the flattened size after convolutions and pooling
        # We need a dummy input with the correct shape to infer the flattened size.
        # The shape is (batch, channels, height, width). Height is n_mels from config.
        # We need a dummy input to infer the flattened size after the CNN backbone.
        with torch.no_grad():
            # The dataset pads/crops spectrograms to a fixed size.
            # From config.yaml: n_mels=64, and time dimension is 500.
            dummy_input = torch.randn(1, self.n_channels, 64, 500)
            dummy_output = self._forward_features(dummy_input)
            flattened_size = dummy_output.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_size)

    def _forward_features(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = torch.flatten(x, 1) # Flatten all dimensions except batch

        # Pass through the fully connected head
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # No activation on the final layer

        return x
