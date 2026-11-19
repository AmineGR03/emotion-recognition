"""Chargement des donnees FER2013 (ou tout dataset au format ImageFolder)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

try:
    from .config import EMOTIONS, TrainConfig
except ImportError:  # execution directe du fichier
    from config import EMOTIONS, TrainConfig

# Statistiques FER2013 (niveaux de gris)
GRAY_MEAN, GRAY_STD = [0.5077], [0.2550]
RGB_MEAN, RGB_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def build_transforms(cfg: TrainConfig, train: bool):
    """Augmentations. L'entrainement sur les visages tolere le flip horizontal,
    de petites rotations/translations, mais pas le flip vertical."""
    mean, std = (GRAY_MEAN, GRAY_STD) if cfg.grayscale else (RGB_MEAN, RGB_STD)
    ops = [transforms.Grayscale(num_output_channels=1)] if cfg.grayscale else \
          [transforms.Grayscale(num_output_channels=3)]
    if train:
        ops += [
            transforms.RandomResizedCrop(cfg.image_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=12, translate=(0.08, 0.08), shear=6),
            transforms.ColorJitter(brightness=0.25, contrast=0.25),
        ]
    else:
        ops += [transforms.Resize((cfg.image_size, cfg.image_size))]
    ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    if train:
        ops += [transforms.RandomErasing(p=0.25, scale=(0.02, 0.12))]
    return transforms.Compose(ops)


def _check_dir(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Dossier introuvable: {path}\n"
            "Attendu: data/fer2013/train/<emotion>/*.jpg et data/fer2013/test/<emotion>/*.jpg\n"
            "Voir scripts/prepare_fer2013.py (conversion du CSV Kaggle) "
            "ou scripts/make_dummy_data.py (jeu factice pour tester le pipeline)."
        )


def build_dataloaders(cfg: TrainConfig):
    """Retourne (train_loader, val_loader, test_loader, classes).

    Structure attendue:
        data_dir/train/<classe>/*.jpg
        data_dir/val/<classe>/*.jpg    (optionnel : sinon split du train)
        data_dir/test/<classe>/*.jpg   (optionnel)
    """
    data_dir = Path(cfg.data_dir)
    _check_dir(data_dir / "train")

    train_tf, eval_tf = build_transforms(cfg, True), build_transforms(cfg, False)
    train_full = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    classes = train_full.classes

    val_dir = data_dir / "val"
    if val_dir.exists():
        train_ds = train_full
        val_ds = datasets.ImageFolder(val_dir, transform=eval_tf)
        train_targets = np.array(train_full.targets)
    else:
        # Split stratifie train/val a partir du meme dossier
        targets = np.array(train_full.targets)
        rng = np.random.default_rng(cfg.seed)
        train_idx, val_idx = [], []
        for c in range(len(classes)):
            idx = np.where(targets == c)[0]
            rng.shuffle(idx)
            n_val = max(1, int(len(idx) * cfg.val_split))
            val_idx += idx[:n_val].tolist()
            train_idx += idx[n_val:].tolist()
        val_base = datasets.ImageFolder(data_dir / "train", transform=eval_tf)
        train_ds = Subset(train_full, train_idx)
        val_ds = Subset(val_base, val_idx)
        train_targets = targets[train_idx]

    test_dir = data_dir / "test"
    test_ds = datasets.ImageFolder(test_dir, transform=eval_tf) if test_dir.exists() else None

    common = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                  pin_memory=torch.cuda.is_available(),
                  persistent_workers=cfg.num_workers > 0)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common) if test_ds else None

    return train_loader, val_loader, test_loader, classes, train_targets


def compute_class_weights(targets, num_classes: int, power: float = 1.0) -> torch.Tensor:
    """Ponderation a la frequence inverse, adoucie par un exposant.

    FER2013 est tres desequilibre (disgust ~1.5% des images, happy ~25%).
    power=1.0 donne l'inverse pur : la classe rare recoit un poids enorme
    (9.39 pour disgust), ce qui la fait sur-predire massivement -- rappel
    eleve mais precision ruinee. power=0.5 (racine carree) ramene ce poids
    a 3.06 : la classe reste favorisee sans que le modele la crie partout.
    power=0.0 revient a ne pas ponderer du tout."""
    counts = np.bincount(np.asarray(targets), minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    weights = (counts.sum() / (num_classes * counts)) ** power
    weights = weights / weights.mean()      # moyenne 1 : l'echelle de la loss ne bouge pas
    return torch.tensor(weights, dtype=torch.float32)
