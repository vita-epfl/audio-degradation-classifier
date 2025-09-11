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
import torchaudio
import torchaudio.transforms as T
import wandb
import os
import argparse
import random
import numpy as np
from datetime import datetime

from src.dataset import DegradationDataset
from src.model import get_model

# --- Setup --- 
warnings.filterwarnings("ignore", category=UserWarning, module='torchaudio')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Custom Loss Function ---
class CombinedLoss(nn.Module):
    def __init__(self, dataset, reg_loss_weight=1.0, label_smoothing=0.05, param_loss_type='mse'):
        super().__init__()
        self.dataset = dataset
        self.classification_loss = nn.CrossEntropyLoss(reduction='none', label_smoothing=label_smoothing)
        if param_loss_type == 'mse':
            self.regression_loss = nn.MSELoss(reduction='none')
        elif param_loss_type == 'smooth_l1':
            self.regression_loss = nn.SmoothL1Loss(reduction='none')
        else:
            raise ValueError(f"Unknown param_loss_type: {param_loss_type}")
        self.reg_loss_weight = reg_loss_weight

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
        cls_loss_per_slot = self.classification_loss(pred_logits.permute(0, 2, 1), true_classes)  # shape (batch, max_effects)

        # Create a mask for active effects
        mask = y_true[:, :, :num_types].sum(dim=-1) > 0  # shape (batch, max_effects)

        # Apply mask to classification loss
        if mask.any():
            loss_cls = (cls_loss_per_slot * mask).sum() / mask.sum()
        else:
            loss_cls = torch.tensor(0.0, device=y_pred.device)

        # Calculate regression loss only for active effects
        # Apply mask to regression loss
        if mask.any():
            loss_reg = self.regression_loss(pred_params, true_params)  # shape (batch, max_effects, max_params)
            loss_reg = (loss_reg * mask.unsqueeze(-1)).sum() / (mask.sum() * self.dataset.max_params)
        else:
            loss_reg = torch.tensor(0.0, device=y_pred.device)

        # Combine losses
        total_loss = loss_cls + self.reg_loss_weight * loss_reg
        return total_loss, loss_cls, loss_reg

def generate_and_log_samples(model, dataset, epoch, device, cfg, mel_spectrogram, amplitude_to_db):
    logging.info(f"Generating audio sample for epoch {epoch}...")
    model.eval()  # Set model to evaluation mode

    # --- Deterministic Setup ---
    # Use epoch number as seed to get the same degradation for the same epoch number
    seed = epoch 
    random.seed(seed)
    np.random.seed(seed)

    # 1. Load a fixed clean audio file for consistency
    clean_file_path = dataset.clean_audio_files[0] 
    clean_waveform, sample_rate = torchaudio.load(clean_file_path)

    # Resample and select clip just like in the dataset
    if sample_rate != cfg.sample_rate:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=cfg.sample_rate)
        clean_waveform = resampler(clean_waveform)
        sample_rate = cfg.sample_rate

    clip_samples = int(cfg.clip_length * sample_rate)
    if clean_waveform.shape[1] > clip_samples:
        # Use a fixed start point for reproducibility
        start = (clean_waveform.shape[1] - clip_samples) // 2 
        clean_waveform = clean_waveform[:, start:start + clip_samples]

    # 2. Generate ground truth degradation
    true_effect_chain = dataset.sox_generator.generate(num_effects_range=(1, 3))
    true_degraded_audio = dataset.sox_generator.apply_effects(clean_waveform, sample_rate, true_effect_chain)

    # 3. Get model's prediction
    with torch.no_grad():
        # Prepare input for model
        mono_waveform = torch.mean(true_degraded_audio, dim=0).unsqueeze(0).to(device)
        spectrogram = amplitude_to_db(mel_spectrogram(mono_waveform))
        spectrogram = spectrogram.unsqueeze(1) # Add channel dim

        predicted_label = model(spectrogram, None)

    # 4. Decode prediction
    predicted_effect_chain = dataset.decode_label(predicted_label.squeeze(0).cpu())

    # 5. Log degradations to W&B as text in a comparison table
    html_table = f"""
    <table style="border: 1px solid black; border-collapse: collapse;">
        <tr>
            <th style="border: 1px solid black; padding: 8px;">Ground Truth</th>
            <td style="border: 1px solid black; padding: 8px;">{' | '.join(true_effect_chain)}</td>
        </tr>
        <tr>
            <th style="border: 1px solid black; padding: 8px;">Predicted</th>
            <td style="border: 1px solid black; padding: 8px;">{' | '.join(predicted_effect_chain)}</td>
        </tr>
    </table>
    """
    wandb.log({
        f"epoch_{epoch}_comparison": wandb.Html(html_table)
    })

    logging.info("Degradations logged to W&B.")
    model.train() # Set model back to training mode

# --- Main Training Logic ---
def main(args):
    """Main training loop."""
    # --- Configuration ---
    with open(args.config, 'r') as f:
        cfg = EasyDict(yaml.safe_load(f))
    with open(args.effects_config, 'r') as f:
        EFFECTS_CONFIG = yaml.safe_load(f)

    DATASET_DIR = Path(cfg.dataset_dir)

    # --- W&B Initialization ---
    project_name = os.environ.get("PROJECT_NAME", "audio-degradation-classifier")
    wandb.init(project=project_name, config=cfg)

    # --- Data Loading ---
    logging.info("Initializing dataset...")
    dataset = DegradationDataset(
        clean_audio_dir=DATASET_DIR,
        sox_effects_config=EFFECTS_CONFIG,
        cfg=cfg
    )
    # We set num_workers=0 because our on-the-fly generation is CPU-bound and can cause
    # bottlenecks with multiprocessing. For I/O-bound tasks, >0 is better.
    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=0)

    # --- Model, Loss, Optimizer ---
    logging.info("Initializing model...")
    _, sample_label = dataset[0]
    # The Maestro dataset is stereo, so we have 2 input channels.
    # However, PANNs models expect single-channel (mono) audio.
    # We will average the channels in the spectrogram generation step.
    model = get_model(cfg, output_size=sample_label.shape[0])

    criterion = CombinedLoss(dataset, reg_loss_weight=cfg.training.reg_loss_weight)
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

    # --- Output Directory Setup ---
    date_str = datetime.now().strftime('%Y%m%d')
    folder_name = f"{cfg.model.name}_{cfg.training.num_epochs}epochs_{date_str}"
    output_dir = Path(cfg.training.output_dir) / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Training Loop ---
    logging.info("Starting training...")
    for epoch in range(cfg.training.num_epochs):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{cfg.training.num_epochs}")

        for i, (waveforms, labels) in enumerate(progress_bar):
            waveforms, labels = waveforms.to(device), labels.to(device)

            # Generate spectrograms on the GPU
            # PANNs expect mono, so we average the channels.
            mono_waveforms = torch.mean(waveforms, dim=1)
            spectrograms = amplitude_to_db(mel_spectrogram(mono_waveforms))
            # Add a channel dimension to match the model's expected input shape (batch, channel, n_mels, time_steps).
            spectrograms = spectrograms.unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(spectrograms, None) # mixup_lambda is None
            loss, loss_c, loss_r = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(
                loss=f'{loss.item():.4f}', 
                cls=f'{loss_c.item():.4f}', 
                reg=f'{loss_r.item():.4f}'
            )

            # Log metrics to W&B
            wandb.log({
                'epoch': epoch,
                'step': i,
                'loss': loss.item(),
                'classification_loss': loss_c.item(),
                'regression_loss': loss_r.item()
            })

        # --- Save Checkpoint ---
        # --- Sample Generation ---
        if (epoch + 1) % cfg.training.get('sample_every', 10) == 0:
            generate_and_log_samples(model, dataset, epoch + 1, device, cfg, mel_spectrogram, amplitude_to_db)

        # --- Save Checkpoint ---
        if (epoch + 1) % cfg.training.get('save_every_n_epoch', 1) == 0:
            checkpoint_path = output_dir / f'model_epoch_{epoch + 1}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f'Checkpoint saved to {checkpoint_path}')

    logging.info('Finished Training')

    # --- Save Final Model --- 
    logging.info('Saving final model...')
    model_path = output_dir / 'model.pth'
    torch.save(model.state_dict(), model_path)
    logging.info(f'Model saved to {model_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model to classify audio degradations.')
    parser.add_argument('--config', type=str, default='config_workstation.yaml', 
                        help='Path to the configuration file.')
    parser.add_argument('--effects-config', type=str, default='effects_config.yaml', 
                        help='Path to the effects configuration file.')
    
    args = parser.parse_args()
    main(args)
