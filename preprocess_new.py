from pathlib import Path
from data import kss

import hydra
from omegaconf import DictConfig, OmegaConf

def write_metadata(train, val, out_dir):
    with Path(out_dir, 'train.txt').open('w', encoding='utf-8') as f:
        for m in train:
            f.write(m + '\n')
    with Path(out_dir, 'val.txt').open('w', encoding='utf-8') as f:
        for m in val:
            f.write(m + '\n')

@hydra.main(config_name='conf/preprocess')
def main(cfg):
    in_dir = Path(cfg.data_path)
    out_dir = Path(cfg.preprocessed_path)

    for sub_dir in ["mel", "alignment", "f0", "energy"]:
        (out_dir / sub_dir).mkdir(exist_ok=True)

    # kss version 1.4
    if not (in_dir / "wavs_bak").exists():
        (in_dir / "wavs").mkdir()
        os.system(f"mv {in_dir.parent / cfg.meta_name} {in_dir}")
        for i in range(1, 5) : 
            os.system(f"mv {in_dir / str(i)} {in_dir / 'wavs'}")
        os.system(f"mv {in_dir / 'wavs'} {in_dir / 'wavs_bak'}")
        (in_dir / "wavs").mkdir()

    train, val = kss.build_from_path(in_dir, out_dir, cfg.meta_name)
    write_metadata(train, val, out_dir)

if __name__ == "__main__":
    main()