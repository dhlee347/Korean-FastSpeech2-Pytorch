from pathlib import Path
import csv
import numpy as np
import datasets

_CITATION = ""
_DESCRIPTION = ""
_HOMEPAGE = "https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset"
_LICENSE = ""


class KoreanSingleSpeech(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.4.0")

    def _info(self):
        features = datasets.Features({
            "id": datasets.Value("string"),
            "text": datasets.Value("string"),
            "mel": datasets.Value("binary"),
            "mel_shape1": datasets.Value("int32"),
            "D": datasets.Value("binary"),
            "f0": datasets.Value("binary"),
            "energy": datasets.Value("binary")
        })
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )        

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"meta_path": Path(self.config.data_dir, "train.txt")}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"meta_path": Path(self.config.data_dir, "val.txt")}            
            ),
        ]

    def _generate_examples(self, meta_path):
        with meta_path.open('r', encoding='utf-8') as f:
            meta = list(csv.reader(f, delimiter='|', quotechar=None))
        
        for id, text in meta:
            mel    = np.load(Path(self.config.data_dir, "mel",       f"mel-{id}.npy"))
            D      = np.load(Path(self.config.data_dir, "alignment", f"align-{id}.npy"))
            f0     = np.load(Path(self.config.data_dir, "f0",        f"f0-{id}.npy"))
            energy = np.load(Path(self.config.data_dir, "energy",    f"energy-{id}.npy"))
            yield id, {
                "id": id,
                "text": text,
                "mel": mel.tobytes(),
                "mel_shape1": mel.shape[1],
                "D": D.tobytes(),
                "f0": f0.tobytes(),
                "energy": energy.tobytes()
            }




