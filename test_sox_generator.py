import random
from pathlib import Path
from src.sox_degradation import SoxEffectGenerator

# --- Configuration ---
DATASET_DIR = Path('/home/alefevre/datasets/maestro-v3.0.0/maestro_full_train/')
OUTPUT_DIR = Path('work/degraded_samples/')

EFFECTS_CONFIG = {
    'equalizer': {
        'frequency': (300, 8000),   # Hz
        'width_q': (0.5, 2.0),       # Q-factor
        'gain': (-30, 30)            # dB
    },
    # We can add more effects here later
    # 'reverb': { ... },
}

# --- Main Test Logic ---
def main():
    """Runs a single test of the SoxEffectGenerator."""
    print("--- Testing SoxEffectGenerator ---")

    # 1. Initialize the generator
    generator = SoxEffectGenerator(EFFECTS_CONFIG)

    # 2. Get a list of clean audio files
    clean_audio_files = list(DATASET_DIR.glob('*.wav'))
    if not clean_audio_files:
        print(f"Error: No .wav files found in {DATASET_DIR}")
        return

    # 3. Pick a random clean file
    input_file = random.choice(clean_audio_files)
    print(f"Selected clean file: {input_file.name}")

    # 4. Generate a random effect chain
    # For this test, let's apply between 1 and 3 equalizer effects
    effect_chain = generator.generate(num_effects_range=(1, 3))
    print(f"Generated effect chain: {effect_chain}")

    # 5. Apply the effects
    output_file = OUTPUT_DIR / f"degraded_{input_file.name}"
    print(f"Applying effects and saving to: {output_file}")
    generator.apply_effects(input_file, output_file, effect_chain)

    print("\nTest complete. Check the output file to verify.")

if __name__ == '__main__':
    main()
