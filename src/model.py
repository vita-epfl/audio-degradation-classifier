import torch
import torch.nn as nn
import torch.nn.functional as F
from src.pann_pytorch.models import Cnn14
import os
import gdown
import logging

class PANNsWithHead(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        # Load the pre-trained PANNs model from the local source.
        # The model requires specific parameters for initialization.
        base_model = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527)

        # Define the path for the pre-trained checkpoint.
        # We'll store it in a local directory to avoid re-downloading.
        checkpoint_dir = "panns_data"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "Cnn14_mAP=0.431.pth")

        # URL for the pre-trained model checkpoint.
        # This is from the official PANNs repository, hosted on Zenodo.
        checkpoint_url = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"

        # Check if the checkpoint exists and is valid.
        # If not, download it.
        if not os.path.exists(checkpoint_path):
            logging.info(f"Checkpoint not found at {checkpoint_path}. Downloading...")
            gdown.download(checkpoint_url, checkpoint_path, quiet=False)
            logging.info("Download complete.")

        # Load the checkpoint and handle potential corruption.
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            base_model.load_state_dict(checkpoint['model'])
        except RuntimeError as e:
            logging.error(f"Failed to load checkpoint: {e}")
            logging.info("The checkpoint file might be corrupted. Deleting and re-downloading.")
            os.remove(checkpoint_path)
            gdown.download(checkpoint_url, checkpoint_path, quiet=False)
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            base_model.load_state_dict(checkpoint['model'])



        # The feature extractor of the Cnn14 model consists of a batch normalization
        # layer and a series of convolutional blocks. We'll combine them into a
        # sequential module.
        self.features = nn.Sequential(
            base_model.bn0,
            base_model.conv_block1,
            base_model.conv_block2,
            base_model.conv_block3,
            base_model.conv_block4,
            base_model.conv_block5,
            base_model.conv_block6
        )

        # We'll replace the final classifier with our own head.
        # The original model has a head for AudioSet tagging (527 classes).
        # We need a head that outputs our required `output_size`.
        self.fc_head = nn.Linear(2048, output_size)

    def forward(self, x, mixup_lambda=None):
        """
        The input x is a spectrogram of shape (batch_size, 1, n_mels, time_steps).
        We need to process it through the PANNs feature extractor and then our custom head.
        """
        # Transpose to (batch_size, 1, time_steps, n_mels) and apply batch norm
        x = x.transpose(1, 2)
        x = self.features[0](x) # bn0
        x = x.transpose(1, 2)

        # Convolutional blocks
        x = self.features[1](x, pool_size=(2, 2), pool_type='avg')
        x = self.features[2](x, pool_size=(2, 2), pool_type='avg')
        x = self.features[3](x, pool_size=(2, 2), pool_type='avg')
        x = self.features[4](x, pool_size=(2, 2), pool_type='avg')
        x = self.features[5](x, pool_size=(2, 2), pool_type='avg')
        x = self.features[6](x, pool_size=(1, 1), pool_type='avg')

        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        # Pass through our custom head
        output = self.fc_head(x)
        return output


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
