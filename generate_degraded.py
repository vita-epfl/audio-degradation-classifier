import argparse
import random
from pathlib import Path

import torch
import torchaudio
import torchaudio.transforms as T
import yaml
from easydict import EasyDict

from src.sox_degradation import SoxEffectGenerator


def load_config(config_path: str) -> EasyDict:
    with open(config_path, 'r') as f:
        return EasyDict(yaml.safe_load(f))


def load_effects_config(effects_config_path: str):
    with open(effects_config_path, 'r') as f:
        return yaml.safe_load(f)


def pick_clip(waveform: torch.Tensor, sr: int, target_seconds: float) -> torch.Tensor:
    """
    Pick a random clip of target_seconds; if shorter, pad; if longer, random crop.
    waveform: (channels, samples)
    """
    target_len = int(target_seconds * sr)
    if waveform.shape[1] == target_len:
        return waveform
    if waveform.shape[1] > target_len:
        start = random.randint(0, waveform.shape[1] - target_len)
        return waveform[:, start:start + target_len]
    # pad end
    pad = target_len - waveform.shape[1]
    return torch.nn.functional.pad(waveform, (0, pad))


def ensure_mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.dim() == 1:
        return waveform.unsqueeze(0)
    if waveform.shape[0] == 1:
        return waveform
    # average to mono
    return waveform.mean(dim=0, keepdim=True)


def process_file(wav_path: Path, out_dir: Path, cfg: EasyDict, sox_gen: SoxEffectGenerator, idx: int):
    waveform, sr = torchaudio.load(wav_path)
    waveform = ensure_mono(waveform)

    # resample
    if sr != cfg.sample_rate:
        resampler = T.Resample(orig_freq=sr, new_freq=cfg.sample_rate)
        waveform = resampler(waveform)
        sr = cfg.sample_rate

    # clip to length
    waveform = pick_clip(waveform, sr, cfg.clip_length)

    # generate effects and apply
    chain = sox_gen.generate(num_effects_range=(1, 5))
    degraded = sox_gen.apply_effects(waveform, sr, chain)

    # ensure fixed length after effects (effects could change length slightly)
    degraded = pick_clip(degraded, sr, cfg.clip_length)

    # save
    out_wav = out_dir / f"degraded_{idx:05d}.wav"
    out_txt = out_dir / f"degraded_{idx:05d}.txt"
    torchaudio.save(out_wav, degraded, sr)
    out_txt.write_text("\n".join(chain))


def main():
    parser = argparse.ArgumentParser(description="Generate degraded audio clips from a clean dataset using SoX effects.")
    parser.add_argument('--clean-dir', type=str, required=True, help='Directory with clean .wav files.')
    parser.add_argument('--out-dir', type=str, required=True, help='Output directory to write degraded wavs and effect chains.')
    parser.add_argument('--num-files', type=int, default=50, help='Number of degraded clips to generate.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to training audio config (sample_rate, clip_length, etc).')
    parser.add_argument('--effects-config', type=str, default='effects_config.yaml', help='Path to effects configuration YAML.')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed for reproducibility.')
    args = parser.parse_args()

    random.seed(args.seed)

    clean_dir = Path(args.clean_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    effects_cfg = load_effects_config(args.effects_config)
    sox_gen = SoxEffectGenerator(effects_cfg)

    wav_files = sorted(clean_dir.glob('*.wav'))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {clean_dir}")

    print(f"Found {len(wav_files)} wav files. Generating {args.num_files} degraded clips into {out_dir}...")

    for i in range(args.num_files):
        src = random.choice(wav_files)
        try:
            process_file(src, out_dir, cfg, sox_gen, i)
        except Exception as e:
            print(f"Failed on {src}: {e}")

    print("Done.")


if __name__ == '__main__':
    main()
