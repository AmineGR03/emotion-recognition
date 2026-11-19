"""Evaluation d'un modele entraine : accuracy, rapport par classe, matrice de confusion.

Exemple:
    python src/evaluate.py --checkpoint outputs/baseline/best_model.pt --split test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import TrainConfig  # noqa: E402
from dataset import build_dataloaders  # noqa: E402
from model import build_model  # noqa: E402
from utils import get_device, load_checkpoint, plot_confusion_matrix, save_json  # noqa: E402


def load_model_from_checkpoint(path: Path, device):
    ckpt = load_checkpoint(path, map_location=device)
    cfg = ckpt.get("config", {})
    classes = cfg.get("classes")
    model = build_model(cfg.get("arch", "cnn"), len(classes), cfg.get("in_channels", 1),
                        float(cfg.get("dropout", 0.4)), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, cfg, classes


@torch.no_grad()
def predict_loader(model, loader, device):
    y_true, y_pred, y_prob = [], [], []
    for images, labels in loader:
        logits = model(images.to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        y_prob.append(probs)
        y_pred.append(probs.argmax(1))
        y_true.append(labels.numpy())
    return (np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(y_prob))


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluation du modele d'emotions")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args()

    device = get_device()
    model, ckpt_cfg, classes = load_model_from_checkpoint(args.checkpoint, device)

    cfg = TrainConfig(
        data_dir=args.data_dir or Path(ckpt_cfg.get("data_dir", TrainConfig().data_dir)),
        arch=ckpt_cfg.get("arch", "cnn"),
        image_size=int(ckpt_cfg.get("image_size", 48)),
        grayscale=bool(ckpt_cfg.get("grayscale", True)),
        batch_size=args.batch_size, num_workers=0,
    )
    _, val_loader, test_loader, _, _ = build_dataloaders(cfg)
    loader = test_loader if (args.split == "test" and test_loader is not None) else val_loader
    if args.split == "test" and test_loader is None:
        print("Pas de dossier test/ : evaluation sur la validation.")

    y_true, y_pred, y_prob = predict_loader(model, loader, device)

    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, top_k_accuracy_score)
    acc = accuracy_score(y_true, y_pred)
    top2 = top_k_accuracy_score(y_true, y_prob, k=2, labels=list(range(len(classes))))
    report = classification_report(y_true, y_pred, target_names=classes, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))

    print(f"\nAccuracy top-1 : {acc:.4f}")
    print(f"Accuracy top-2 : {top2:.4f}\n")
    print(report)

    out_dir = args.checkpoint.parent
    plot_confusion_matrix(cm, classes, out_dir / f"confusion_matrix_{args.split}.png")
    save_json(out_dir / f"metrics_{args.split}.json", {
        "accuracy": float(acc), "top2_accuracy": float(top2),
        "classes": classes, "confusion_matrix": cm.tolist(),
        "report": classification_report(y_true, y_pred, target_names=classes,
                                        output_dict=True, zero_division=0),
    })
    print(f"Matrice de confusion : {out_dir / f'confusion_matrix_{args.split}.png'}")


if __name__ == "__main__":
    main()
