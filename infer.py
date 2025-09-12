import torch
import torchaudio
import torchaudio.transforms as T
import yaml
from easydict import EasyDict
import argparse
import numpy as np

from src.model import SoxDegradationClassifier

# --- Global Variables ---
model = None
cfg = None
effects_config = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Placeholder for label decoding information
effect_map_inv = None
param_keys_map = None
num_effect_types = 0
max_params = 0
label_item_size = 0
max_effects = None  # Will be set from config

# --- Decoding Logic ---
def decode_output(output_tensor):
    """Decodes a model output tensor into a human-readable effect chain."""
    if not all([effect_map_inv, param_keys_map]):
        return "Error: Decoding information not initialized."

    # Get no_effect index
    no_effect_idx = None
    for idx, name in effect_map_inv.items():
        if name == 'no_effect':
            no_effect_idx = idx
            break

    output_tensor = output_tensor.view(max_effects, label_item_size)
    
    detected_effects = []

    for i in range(max_effects):
        effect_slot = output_tensor[i]
        
        # --- Effect Type Classification ---
        logits = effect_slot[:num_effect_types]
        
        pred_effect_idx = torch.argmax(logits).item()
        
        # Skip if it's 'no_effect'
        if pred_effect_idx == no_effect_idx:
            continue
            
        effect_name = effect_map_inv.get(pred_effect_idx, "unknown_effect")

        # --- Parameter Regression ---
        raw_params = effect_slot[num_effect_types:num_effect_types + max_params]
        param_config = effects_config.get(effect_name, {})
        param_keys = param_keys_map.get(effect_name, [])
        
        # Denormalize and format parameters
        param_strs = []
        for j, key in enumerate(param_keys):
            if key in param_config:
                min_val, max_val = param_config[key]
                val = raw_params[j].item() * (max_val - min_val) + min_val
                param_strs.append(f"{val:.2f}")

        effect_str = f"{effect_name} {' '.join(param_strs)}"
        detected_effects.append(effect_str)

    if not detected_effects:
        return "No effects detected."

    return "\n".join(detected_effects)

def predict(audio_path):
    """
    Takes a path to an audio file and returns the predicted
    degradation effect chain as a string.
    """
    if model is None:
        return "Error: Model not loaded."

    try:
        waveform, sr = torchaudio.load(audio_path)
    except Exception as e:
        return f"Error loading audio file: {e}"

    waveform = waveform.to(device)
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True) # Convert to mono

    # --- Preprocessing ---
    if sr != cfg.sample_rate:
        resampler = T.Resample(orig_freq=sr, new_freq=cfg.sample_rate).to(device)
        waveform = resampler(waveform)

    mel_spectrogram = T.MelSpectrogram(
        sample_rate=cfg.sample_rate, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
        n_mels=cfg.n_mels, f_min=cfg.f_min, f_max=cfg.f_max
    ).to(device)
    amplitude_to_db = T.AmplitudeToDB().to(device)
    spectrogram = amplitude_to_db(mel_spectrogram(waveform))

    # --- Prediction ---
    with torch.no_grad():
        model.eval()
        output = model(spectrogram)
        
    # --- Decode and Return ---
    readable_prediction = decode_output(output.squeeze(0))
    return readable_prediction

# --- Main Execution ---
def main():
    global model, cfg, effects_config, effect_map_inv, param_keys_map, num_effect_types, max_params, label_item_size

    parser = argparse.ArgumentParser(description="Run inference to classify audio degradation effects.")
    parser.add_argument('--weights', type=str, required=True, help='Path to the model weights file (e.g., work/model.pth).')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the training config file.')
    parser.add_argument('--effects-config', type=str, default='effects_config.yaml', help='Path to the effects config YAML file.')
    parser.add_argument('--audio-file', type=str, required=True, help='Path to the input audio file.')

    args = parser.parse_args()

    # --- Load Configs ---
    try:
        with open(args.config, 'r') as f:
            cfg = EasyDict(yaml.safe_load(f))
        with open(args.effects_config, 'r') as f:
            effects_config = yaml.safe_load(f)
    except FileNotFoundError as e:
        print(f"Error: Config file not found - {e}")
        return

    # Set max_effects from config
    global max_effects
    max_effects = cfg.max_effects

    # --- Setup Decoding Information ---
    effect_map = {name: i for i, name in enumerate(effects_config.keys())}
    effect_map['no_effect'] = len(effect_map)  # Add no_effect class
    effect_map_inv = {i: name for name, i in effect_map.items()}
    num_effect_types = len(effect_map)
    
    max_params = 0
    param_keys_map = {}
    for name, params in effects_config.items():
        max_params = max(max_params, len(params))
        param_keys_map[name] = list(params.keys())
    
    label_item_size = num_effect_types + max_params
    output_size = max_effects * label_item_size

    # --- Load Model ---
    try:
        model = SoxDegradationClassifier(n_channels=1, output_size=output_size)
        model.load_state_dict(torch.load(args.weights, map_location=device))
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Model and configs loaded. Analyzing audio file: {args.audio_file}")
    
    # --- Run Prediction ---
    prediction = predict(args.audio_file)
    
    print("\n--- Detected Effects ---")
    print(prediction)
    print("------------------------\n")

if __name__ == '__main__':
    main()
