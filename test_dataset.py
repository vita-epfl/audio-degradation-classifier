from pathlib import Path
from src.dataset import DegradationDataset

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
    """Tests the DegradationDataset."""
    print("--- Testing DegradationDataset ---")

    # 1. Initialize the dataset
    dataset = DegradationDataset(
        clean_audio_dir=DATASET_DIR,
        sox_effects_config=EFFECTS_CONFIG
    )

    # 2. Get a single sample
    print("\nFetching a sample from the dataset...")
    spectrogram, label = dataset[0]

    # 3. Print information about the sample
    print(f"\nSpectrogram shape: {spectrogram.shape}")
    print(f"Spectrogram type: {spectrogram.dtype}")
    print(f"\nLabel shape: {label.shape}")
    print(f"Label type: {label.dtype}")
    print(f"Label tensor (first 10 elements): {label[:10]}...")

    # Check the expected label size
    num_effects = len(EFFECTS_CONFIG)
    max_params = max(len(p) for p in EFFECTS_CONFIG.values())
    expected_label_size = dataset.max_effects * (num_effects + max_params)
    print(f"\nExpected label size: {expected_label_size}")
    assert label.shape[0] == expected_label_size, "Label size does not match expected size!"

    print("\nTest complete. The dataset appears to be working correctly.")

if __name__ == '__main__':
    main()
