from pathlib import Path
import yaml
from easydict import EasyDict
import torch
import torchaudio.transforms as T

from src.dataset import DegradationDataset
from src.model import get_model

# --- Configuration ---
CONFIG_PATH = 'config_scitas.yaml'
EFFECTS_CONFIG_PATH = 'effects_config.yaml'

# --- Main Test Logic ---
def main():
    """Tests the model specified in the config file."""
    print(f"--- Testing Model from {CONFIG_PATH} ---")

    # 1. Load configurations
    with open(CONFIG_PATH, 'r') as f:
        cfg = EasyDict(yaml.safe_load(f))
    with open(EFFECTS_CONFIG_PATH, 'r') as f:
        effects_config = yaml.safe_load(f)

    # 2. Initialize the dataset to get shapes
    dataset = DegradationDataset(
        clean_audio_dir=Path(cfg.dataset_dir),
        sox_effects_config=effects_config,
        cfg=cfg
    )

    # 3. Get a sample to determine input/output sizes
    print("\nFetching a sample for shape inference...")
    waveform, label = dataset[0]
    output_size = label.shape[0]

    print(f"Input shape (waveform): {waveform.shape}")
    print(f"Output size (label): {output_size}")

    # 4. Initialize the model
    print("\nInitializing the model...")
    model = get_model(cfg, output_size)

    # 5. Create spectrogram transforms
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        f_min=cfg.f_min,
        f_max=cfg.f_max
    )
    amplitude_to_db = T.AmplitudeToDB()

    # 6. Perform a forward pass
    print("Performing a forward pass...")
    # Convert to mono and generate spectrogram
    mono_waveform = torch.mean(waveform, dim=0, keepdim=True)
    spectrogram = amplitude_to_db(mel_spectrogram(mono_waveform))
    
    # Add a batch dimension to the spectrogram
    output = model(spectrogram.unsqueeze(0))

    # 7. Check the output shape
    print(f"\nModel output shape: {output.shape}")
    print(f"Expected output shape: (1, {output_size})")
    assert output.shape == (1, output_size), "Model output shape is incorrect!"

    print("\nTest complete. The model appears to be working correctly.")

if __name__ == '__main__':
    main()
