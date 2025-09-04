from pathlib import Path
from src.dataset import DegradationDataset
from src.model import SoxDegradationClassifier

# --- Configuration ---
DATASET_DIR = Path('/home/alefevre/datasets/maestro-v3.0.0/maestro_full_train/')

EFFECTS_CONFIG = {
    'equalizer': {
        'frequency': (300, 8000),
        'width_q': (0.5, 2.0),
        'gain': (-30, 30)
    },
    'reverb': {
        'reverberance': (0, 100),
        'hf_damping': (0, 100),
        'room_scale': (0, 100)
    }
}

# --- Main Test Logic ---
def main():
    """Tests the SoxDegradationClassifier model."""
    print("--- Testing SoxDegradationClassifier ---")

    # 1. Initialize the dataset to get shapes
    dataset = DegradationDataset(
        clean_audio_dir=DATASET_DIR,
        sox_effects_config=EFFECTS_CONFIG
    )

    # 2. Get a sample to determine input/output sizes
    print("\nFetching a sample for shape inference...")
    spectrogram, label = dataset[0]
    input_shape = spectrogram.shape
    output_size = label.shape[0]

    print(f"Input shape (spectrogram): {input_shape}")
    print(f"Output size (label): {output_size}")

    # 3. Initialize the model
    print("\nInitializing the model...")
    n_channels = input_shape[0]
    model = SoxDegradationClassifier(n_channels=n_channels, output_size=output_size)

    # 4. Perform a forward pass
    print("Performing a forward pass...")
    # Add a batch dimension to the spectrogram
    output = model(spectrogram.unsqueeze(0))

    # 5. Check the output shape
    print(f"\nModel output shape: {output.shape}")
    print(f"Expected output shape: (1, {output_size})")
    assert output.shape == (1, output_size), "Model output shape is incorrect!"

    print("\nTest complete. The model appears to be working correctly.")

if __name__ == '__main__':
    main()
