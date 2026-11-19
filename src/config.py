"""Configuration centrale du projet de reconnaissance d'emotions faciales."""
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

# Ordre alphabetique = ordre utilise par torchvision.datasets.ImageFolder
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOTIONS_FR = {
    "angry": "Colere",
    "disgust": "Degout",
    "fear": "Peur",
    "happy": "Joie",
    "neutral": "Neutre",
    "sad": "Tristesse",
    "surprise": "Surprise",
}
NUM_CLASSES = len(EMOTIONS)


@dataclass
class TrainConfig:
    # Donnees
    data_dir: Path = DATA_DIR / "fer2013"
    image_size: int = 48          # 48 pour le CNN maison, 96/224 pour le transfer learning
    grayscale: bool = True
    val_split: float = 0.1        # part du train utilisee en validation si pas de dossier val/

    # Modele
    arch: str = "cnn"             # "cnn" | "resnet18" | "efficientnet_b0"
    dropout: float = 0.4
    pretrained: bool = True       # uniquement pour les archis torchvision

    # Optimisation
    epochs: int = 60
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    use_class_weights: bool = True
    class_weight_power: float = 0.5   # 1.0 = inverse pur, 0.5 = adouci, 0.0 = aucun
    warmup_epochs: int = 3
    early_stopping_patience: int = 12

    # Divers
    num_workers: int = 2
    seed: int = 42
    amp: bool = True              # mixed precision (GPU uniquement)
    output_dir: Path = OUTPUT_DIR
    run_name: str = "baseline"

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in d.items()}

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.run_name
