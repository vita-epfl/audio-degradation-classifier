import torch
import torchaudio
import numpy as np
import random
from pathlib import Path
import yaml
from easydict import EasyDict

from src.sox_degradation import SoxEffectGenerator

class DegradationDataset(torch.utils.data.Dataset):
    def __init__(self, clean_audio_dir, sox_effects_config, config_path='config.yaml'):
        super().__init__()

        print("Initializing dataset...")
        self.clean_audio_files = sorted(list(Path(clean_audio_dir).glob('*.wav')))
        if not self.clean_audio_files:
            raise FileNotFoundError(f"No .wav files found in {clean_audio_dir}")

        # Load audio processing config
        with open(config_path) as f:
            self.cfg = EasyDict(yaml.safe_load(f))

        # Setup Sox Effect Generator
        self.sox_generator = SoxEffectGenerator(sox_effects_config)

        # --- Label Encoding Setup ---
        self.effect_map = {name: i for i, name in enumerate(self.sox_generator.available_effects)}
        self.num_effect_types = len(self.effect_map)
        self.max_effects = 5  # Max number of effects in a chain we can predict

        # Determine the max number of parameters any single effect has
        self.max_params = 0
        for effect_name, params in sox_effects_config.items():
            self.max_params = max(self.max_params, len(params))

        # The representation for one effect slot: one-hot(effect_type) + params
        self.label_item_size = self.num_effect_types + self.max_params

        # --- Torchaudio Transforms ---
        self.resampler = None # Will be initialized on demand


        print(f"Found {len(self.clean_audio_files)} clean audio files.")
        print("Dataset initialized.")

    def __len__(self):
        return len(self.clean_audio_files)

    def __getitem__(self, index):
        # For now, we generate a new degradation for each request, regardless of index.
        # This makes the dataset effectively infinite.

        # 1. Load a random clean audio file and take a short clip
        clean_file_path = random.choice(self.clean_audio_files)
        waveform, sr = torchaudio.load(clean_file_path)

        # Resample if necessary
        if sr != self.cfg.sample_rate:
            if self.resampler is None or self.resampler.orig_freq != sr:
                self.resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.cfg.sample_rate)
            waveform = self.resampler(waveform)
            sr = self.cfg.sample_rate

        # Take a random clip
        clip_samples = int(self.cfg.clip_length * sr)
        if waveform.shape[1] > clip_samples:
            start = random.randint(0, waveform.shape[1] - clip_samples)
            waveform = waveform[:, start:start + clip_samples]

        # 2. Generate a random sox effect chain
        effect_chain = self.sox_generator.generate(num_effects_range=(1, 5))

        # 3. Apply effects in-memory
        degraded_waveform = self.sox_generator.apply_effects(waveform, sr, effect_chain)

        # 5. Pad/truncate waveform to a fixed length
        # This ensures all items in a batch have the same size
        fixed_length_samples = int(self.cfg.clip_length * self.cfg.sample_rate)
        if degraded_waveform.shape[1] < fixed_length_samples:
            # Pad if shorter
            pad_size = fixed_length_samples - degraded_waveform.shape[1]
            degraded_waveform = torch.nn.functional.pad(degraded_waveform, (0, pad_size))
        elif degraded_waveform.shape[1] > fixed_length_samples:
            # Truncate if longer
            degraded_waveform = degraded_waveform[:, :fixed_length_samples]

        # 6. Encode the label
        label = self.encode_label(effect_chain)

        return degraded_waveform, label

    def encode_label(self, effect_chain):
        """
        Encodes a list of effect strings into a fixed-size tensor.
        Output shape: (max_effects, num_effect_types + max_params)
        """
        label = torch.zeros(self.max_effects, self.label_item_size)

        for i, effect_str in enumerate(effect_chain):
            if i >= self.max_effects:
                break  # Stop if we exceed the max number of effects

            parts = effect_str.split()
            effect_name = parts[0]
            params = [float(p) for p in parts[1:]]

            # Set the one-hot vector for the effect type
            effect_idx = self.effect_map[effect_name]
            label[i, effect_idx] = 1.0

            # Normalize and store parameters
            param_config = self.sox_generator.effects_config[effect_name]
            param_keys = list(param_config.keys())  # e.g., ['frequency', 'width_q', 'gain']
            param_offset = self.num_effect_types

            for j, param_value in enumerate(params):
                param_name = param_keys[j]
                min_val, max_val = param_config[param_name]
                # Normalize to [0, 1]
                normalized_param = (param_value - min_val) / (max_val - min_val)
                label[i, param_offset + j] = normalized_param

        return label.flatten()  # Flatten for use in the model
