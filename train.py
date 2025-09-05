import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import warnings
from tqdm import tqdm
import yaml
from easydict import EasyDict
import torchaudio.transforms as T

from src.dataset import DegradationDataset
from src.model import SoxDegradationClassifier

# --- Setup --- 
warnings.filterwarnings("ignore", category=UserWarning, module='torchaudio')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
with open('config.yaml', 'r') as f:
    cfg = EasyDict(yaml.safe_load(f))

with open('effects_config.yaml', 'r') as f:
    EFFECTS_CONFIG = yaml.safe_load(f)

DATASET_DIR = Path('/work/vita/datasets/maestro-v3.0.0/maestro_full_train')

# --- Custom Loss Function ---
class CombinedLoss(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.classification_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()

    def forward(self, y_pred, y_true):
        # Reshape predictions and labels to (batch, max_effects, item_size)
        item_size = self.dataset.label_item_size
        max_effects = self.dataset.max_effects
        num_types = self.dataset.num_effect_types

        y_pred = y_pred.view(-1, max_effects, item_size)
        y_true = y_true.view(-1, max_effects, item_size)

        # Split into classification and regression parts
        pred_logits = y_pred[:, :, :num_types]
        pred_params = y_pred[:, :, num_types:]
        true_classes = y_true[:, :, :num_types].argmax(dim=-1)
        true_params = y_true[:, :, num_types:]

        # Calculate classification loss
        # We need to reshape for CrossEntropyLoss: (N, C, ...)
        loss_cls = self.classification_loss(pred_logits.permute(0, 2, 1), true_classes)

        # Calculate regression loss only for active effects
        # Create a mask for active effects
        mask = y_true[:, :, :num_types].sum(dim=-1) > 0
        if mask.any():
            loss_reg = self.regression_loss(pred_params[mask], true_params[mask])
        else:
            loss_reg = torch.tensor(0.0, device=y_pred.device)

        # Combine losses (we can tune the weights)
        total_loss = loss_cls + loss_reg
        return total_loss, loss_cls, loss_reg

# --- Main Training Logic ---
def main():
    """Main training loop."""
    # --- Data Loading ---
    logging.info("Initializing dataset...")
    dataset = DegradationDataset(
        clean_audio_dir=DATASET_DIR,
        sox_effects_config=EFFECTS_CONFIG
    )
    # We set num_workers=0 because our on-the-fly generation is CPU-bound and can cause
    # bottlenecks with multiprocessing. For I/O-bound tasks, >0 is better.
    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=0)

    # --- Model, Loss, Optimizer ---
    logging.info("Initializing model...")
    _, sample_label = dataset[0]
    # The Maestro dataset is stereo, so we have 2 input channels.
    model = SoxDegradationClassifier(n_channels=2, output_size=sample_label.shape[0])

    criterion = CombinedLoss(dataset)
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logging.info(f"Using device: {device}")
    if device.type == 'cuda':
        logging.info(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # --- Spectrogram Transform (on GPU) ---

    mel_spectrogram = T.MelSpectrogram(
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        f_min=cfg.f_min,
        f_max=cfg.f_max
    ).to(device)

    amplitude_to_db = T.AmplitudeToDB().to(device)

    # --- Training Loop ---
    logging.info("Starting training...")
    for epoch in range(cfg.training.num_epochs):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{cfg.training.num_epochs}")

        for i, (waveforms, labels) in enumerate(progress_bar):
            waveforms, labels = waveforms.to(device), labels.to(device)

            # Generate spectrograms on the GPU
            spectrograms = amplitude_to_db(mel_spectrogram(waveforms))

            optimizer.zero_grad()
            outputs = model(spectrograms)
            loss, loss_c, loss_r = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(
                loss=f'{loss.item():.4f}', 
                cls=f'{loss_c.item():.4f}', 
                reg=f'{loss_r.item():.4f}'
            )

    logging.info('Finished Training')

    # --- Save Model --- 
    logging.info('Saving model...')
    output_dir = Path('work')
    output_dir.mkdir(exist_ok=True)
    model_path = output_dir / 'model.pth'
    torch.save(model.state_dict(), model_path)
    logging.info(f'Model saved to {model_path}')

if __name__ == '__main__':
    main()
