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


def normalize_spectrogram(spectrogram, eps=1e-8, dim=(-2, -1)) -> torch.Tensor:  # expects (..., freq, time) shape
    """
    Normalize spectrogram by subtracting mean and dividing by std across frequency and time dimensions.
    
    Args:
        spectrogram: Input spectrogram tensor
        eps: Small epsilon to avoid division by zero
        dim: Dimensions to compute mean and std across
    
    Returns:
        Normalized spectrogram tensor
    """
    mean = spectrogram.mean(dim=dim, keepdim=True)
    std = spectrogram.std(dim=dim, keepdim=True)
    return (spectrogram - mean) / (std + eps)


def apply_mixup(spectrograms, labels, mixup_lambda, max_effects=5, num_effect_types=10, max_params=5):
    """
    Apply mixup augmentation to spectrograms and labels with proper handling of classification and regression parts.
    
    Args:
        spectrograms: Batch of spectrograms (batch, channels, freq, time)
        labels: Batch of flattened labels (batch, max_effects * label_item_size)
        mixup_lambda: Mixing coefficient (0-1)
        max_effects: Maximum number of effects per sample
        num_effect_types: Number of effect types (for classification)
        max_params: Maximum parameters per effect (for regression)
    
    Returns:
        mixed_spectrograms, mixed_labels
    """
    batch_size = spectrograms.shape[0]
    label_item_size = num_effect_types + max_params
    
    # Generate random permutation for mixing
    indices = torch.randperm(batch_size, device=spectrograms.device)
    
    # Mix spectrograms
    mixed_spectrograms = mixup_lambda * spectrograms + (1 - mixup_lambda) * spectrograms[indices]
    
    # Mix labels - need to handle classification and regression separately
    labels_reshaped = labels.view(batch_size, max_effects, label_item_size)
    labels_indices = labels[indices].view(batch_size, max_effects, label_item_size)
    
    # Split into classification (one-hot logits) and regression (parameters) parts
    class_logits = labels_reshaped[:, :, :num_effect_types]  # (batch, max_effects, num_effect_types)
    class_logits_indices = labels_indices[:, :, :num_effect_types]
    
    reg_params = labels_reshaped[:, :, num_effect_types:]  # (batch, max_effects, max_params)
    reg_params_indices = labels_indices[:, :, num_effect_types:]
    
    # Mix classification logits (these are already in logit form, can interpolate directly)
    mixed_class_logits = mixup_lambda * class_logits + (1 - mixup_lambda) * class_logits_indices
    
    # Mix regression parameters
    mixed_reg_params = mixup_lambda * reg_params + (1 - mixup_lambda) * reg_params_indices
    
    # Combine back
    mixed_labels_reshaped = torch.cat([mixed_class_logits, mixed_reg_params], dim=-1)
    mixed_labels = mixed_labels_reshaped.view(batch_size, -1)  # Flatten back
    
    return mixed_spectrograms, mixed_labels


# --- Custom Loss Function ---
class CombinedLoss(nn.Module):
    def __init__(self, dataset, reg_loss_weight=1.0, label_smoothing=0.05, param_loss_type='smooth_l1'):
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

        # Build dict for number of params per effect type
        self.effect_num_params = {name: len(params) for name, params in self.dataset.sox_generator.effects_config.items()}
        self.effect_map_inv = {v: k for k, v in self.dataset.effect_map.items()}

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

        if cfg.training.mask_active_effects:
            # Calculate regression loss only for active effects
            # Create per-parameter mask based on actual number of params for each effect type
            param_mask = torch.zeros_like(pred_params)  # (batch, max_effects, max_params)
            for i in range(y_pred.shape[0]):
                for j in range(max_effects):
                    if mask[i, j]:
                        effect_idx = true_classes[i, j].item()
                        effect_name = self.effect_map_inv[effect_idx]
                        num_p = self.effect_num_params[effect_name]
                        param_mask[i, j, :num_p] = 1

        # Apply mask to regression loss
        if mask.any():
            loss_reg = self.regression_loss(pred_params, true_params)  # shape (batch, max_effects, max_params)
            if param_mask.any():
                loss_reg = (loss_reg * param_mask).sum() / param_mask.sum()
            else:
                loss_reg = loss_reg.sum() / mask.sum()
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
        # Normalize spectrogram per sample
        spectrogram = normalize_spectrogram(spectrogram)
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

    # --- SpecAugment for data augmentation ---
    freq_masking = T.FrequencyMasking(freq_mask_param=2).to(device)  # Mask 2 frequency bins
    time_masking = T.TimeMasking(time_mask_param=10).to(device)     # Mask 10 time steps

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
            # Normalize spectrogram per sample
            spectrograms = normalize_spectrogram(spectrograms)
            if cfg.training.apply_spec_augment:
                # Apply SpecAugment data augmentation during training
                spectrograms = freq_masking(spectrograms)
                spectrograms = time_masking(spectrograms)
            # Add a channel dimension to match the model's expected input shape (batch, channel, n_mels, time_steps).
            spectrograms = spectrograms.unsqueeze(1)

            if cfg.training.apply_mixup:

                # Apply Mixup augmentation
                mixup_lambda = torch.rand(1, device=device).item()  # Random lambda between 0 and 1
                if mixup_lambda > 0.5:  # Apply mixup with 50% probability, or you can adjust this
                    spectrograms, labels = apply_mixup(
                        spectrograms, labels, mixup_lambda, 
                        max_effects=dataset.max_effects,
                        num_effect_types=dataset.num_effect_types,
                        max_params=dataset.max_params
                    )
                else:
                    mixup_lambda = 1.0  # No mixup applied

            optimizer.zero_grad()
            outputs = model(spectrograms, mixup_lambda if cfg.training.apply_mixup else None)  # Pass mixup_lambda to model
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
