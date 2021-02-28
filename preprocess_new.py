import os
import csv

from pathlib import Path
from typing import Dict

from tqdm import tqdm
import hydra
from omegaconf import OmegaConf, DictConfig

from sklearn.preprocessing import StandardScaler
import numpy as np
import torch

import tgt
import pyworld as pw
import scipy.io.wavfile as sci_wav

import audio
from data import kss
from utils import get_alignment, standard_norm, remove_outlier, average_by_duration


@hydra.main(config_name="conf/preprocess_inspiron")
def main(cfg):
    path = Path(cfg.data_dir, cfg.preprocessed_path)
    script_path = Path(cfg.data_dir, cfg.script_path)

    # Make directories for preprocessed files
    path.mkdir(exist_ok=True)
    for sub_dir in ["wav", "mel", "alignment", "f0", "energy"]:
        Path(path, sub_dir).mkdir(exist_ok=True)

    # Load transcript
    with script_path.open('r', encoding='utf-8') as f:
        script = list(csv.reader(f, delimiter='|', quotechar=None))
    
    # Make scalers
    scalers = {
        'f0':     StandardScaler(copy=False),
        'energy': StandardScaler(copy=False),
        'mel':    StandardScaler(copy=False),
    }

    # Convert wav files
    meta = []
    for wav_path, _, _, text, _, _ in tqdm(script):
        wav_path = Path(wav_path)
        text = preprocess(cfg,
            # Original wav file path : ex) /data/kss/3/3_4229.wav
            Path(cfg.data_dir, wav_path),
            # TextGrid file path : ex) /data/kss/TextGrid/3_4229.TextGrid
            Path(cfg.data_dir, cfg.text_grid_path, f"{wav_path.stem}.TextGrid"),
            # Converted wav file path : ex) /data/kss/preprocessed/wav/3_4229.wav
            Path(path, "wav", wav_path.name),
            # alignment file path : ex) /data/kss/preprocessed/alignment/align-3_4229.npy
            Path(path, "alignment", f"align-{wav_path.stem}.npy"),
            # fundamental frequency(f0) file path : ex) /data/kss/preprocessed/f0/f0-3_4229.npy
            Path(path, "f0", f"f0-{wav_path.stem}.npy"),
            # energy file path : ex) /data/kss/preprocessed/energy/energy-3_4229.npy
            Path(path, "energy", f"energy-{wav_path.stem}.npy"),
            # mel-spectrogram file path : ex) /data/kss/preprocessed/mel/mel-3_4229.npy
            Path(path, "mel", f"mel-{wav_path.stem}.npy"),
            scalers,
        )
        if text is None: continue

        meta.append(f"{wav_path.stem}|{text}")

    # Write meta files : train.txt, val.txt
    # ex) 2_1378|{ᄋ ᅧ ᄀ ᅵ ᄋ ᅦ ᄉ ᅥ  ᄀ ᅩ ᄉ ᅦ sp ᄉ ᅥ }
    with Path(path, "train.txt").open('w', encoding='utf-8') as f:
        f.write('\n'.join([line for line in meta if line.startswith(('2', '3', '4'))]))
    with Path(path, "val.txt").open('w', encoding='utf-8') as f:
        f.write('\n'.join([line for line in meta if line.startswith('1')]))

    # Write statistics
    for key, scaler in scalers.items(): # keys: f0, energy, mel
        np.save(Path(path, f"{key}_stat.npy"), np.array([scaler.mean_, scaler.scale_]))


def preprocess(
        cfg: OmegaConf,
        original_wav_path: Path,
        text_grid_path: Path,
        converted_wav_path: Path,
        alignment_path: Path,
        f0_path: Path,
        energy_path: Path,
        mel_path: Path,
        scalers: Dict[str, StandardScaler],
    ):
    # Convert kss data into PCM encoded wavs
    if not converted_wav_path.exists():
        os.system(f"ffmpeg -i {original_wav_path} -ac 1 -ar 22050 {converted_wav_path} >/dev/null 2>&1")

    # Get alignments
    text_grid = tgt.io.read_textgrid(text_grid_path)
    phone, duration, start, end = get_alignment(text_grid.get_tier_by_name('phones'))
    # ['A','B','$','C'] --> '{A B} {C}' : '$' represents silent phone
    text = ' '.join([f"{{{x.strip()}}}" for x in ' '.join(phone).split('$')])

    if start >= end: return None

    # Read and trim wav files
    _, wav = sci_wav.read(converted_wav_path) # wav : int16
    trim_start = int(start * cfg.sampling_rate)
    trim_end   = int(end   * cfg.sampling_rate)
    wav = wav[trim_start:trim_end].astype(np.float32)

    # Compute fundamental frequency
    f0, _ = pw.dio(
        wav.astype(np.float64),
        cfg.sampling_rate, 
        frame_period=cfg.hop_length / cfg.sampling_rate * 1000
    )
    f0 = f0[:sum(duration)]

    # Compute mel-scale spectrogram and energy
    mel, energy = audio.tools.get_mel_from_wav(
        torch.FloatTensor(wav),
        cfg.sampling_rate,
        cfg.max_wav_value
    )
    mel = mel[:, :sum(duration)].numpy()
    energy = energy[:sum(duration)].numpy()

    # Preprocessing for scaling
    f0, energy = remove_outlier(f0), remove_outlier(energy)
    f0, energy = average_by_duration(f0, duration), average_by_duration(energy, duration)

    if mel.shape[1] >= cfg.max_seq_len: return None

    # Save alignment, fundamental frequency(f0), energy, mel-spectogram
    np.save(alignment_path, duration, allow_pickle=False)
    np.save(f0_path, f0, allow_pickle=False)
    np.save(energy_path, energy, allow_pickle=False)
    np.save(mel_path, mel.T, allow_pickle=False)

    # Fit scalers
    scalers['f0'].partial_fit(f0[f0!=0].reshape(-1, 1))
    scalers['energy'].partial_fit(energy[energy != 0].reshape(-1, 1))
    scalers['mel'].partial_fit(mel.T)

    return text


if __name__ == "__main__":
    main()