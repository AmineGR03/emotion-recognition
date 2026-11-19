"""Entrainement du modele de reconnaissance d'emotions faciales.

Exemple:
    python src/train.py --arch cnn --epochs 60 --batch-size 64 --run-name baseline
    python src/train.py --arch resnet18 --image-size 96 --lr 3e-4 --run-name resnet18
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import TrainConfig, NUM_CLASSES  # noqa: E402
from dataset import build_dataloaders, compute_class_weights  # noqa: E402
from model import build_model, count_parameters  # noqa: E402
from utils import (AverageMeter, get_device, plot_history, save_checkpoint,  # noqa: E402
                   save_json, set_seed)


def parse_args() -> TrainConfig:
    d = TrainConfig()
    p = argparse.ArgumentParser(description="Entrainement reconnaissance d'emotions faciales")
    p.add_argument("--data-dir", type=Path, default=d.data_dir)
    p.add_argument("--arch", default=d.arch, choices=["tiny", "cnn", "resnet18", "efficientnet_b0"])
    p.add_argument("--image-size", type=int, default=d.image_size)
    p.add_argument("--rgb", action="store_true", help="entree 3 canaux au lieu du niveau de gris")
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--dropout", type=float, default=d.dropout)
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--weight-power", type=float, default=d.class_weight_power,
                   help="exposant des poids de classe : 1.0 inverse pur, 0.5 adouci, 0.0 aucun")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument("--patience", type=int, default=d.early_stopping_patience)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--run-name", default=d.run_name)
    a = p.parse_args()
    return TrainConfig(
        data_dir=a.data_dir, arch=a.arch, image_size=a.image_size, grayscale=not a.rgb,
        epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, weight_decay=a.weight_decay,
        dropout=a.dropout, use_class_weights=not a.no_class_weights,
        class_weight_power=a.weight_power, amp=not a.no_amp,
        num_workers=a.num_workers, early_stopping_patience=a.patience, seed=a.seed,
        run_name=a.run_name,
    )


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, amp=False):
    train = optimizer is not None
    model.train(train)
    loss_meter, correct, total = AverageMeter(), 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            if train:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer); scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
            loss_meter.update(loss.item(), labels.size(0))
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return loss_meter.avg, correct / max(total, 1)


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    device = get_device()
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, classes, train_targets = build_dataloaders(cfg)
    num_classes = len(classes)
    in_channels = 1 if cfg.grayscale else 3

    if cfg.arch not in ("cnn", "tiny") and cfg.image_size < 64:
        print(f"[!] {cfg.arch} divise la resolution par 32 : en {cfg.image_size}x{cfg.image_size} "
              f"la carte de features finale est minuscule. Utilisez --image-size 96 ou plus.")

    model = build_model(cfg.arch, num_classes, in_channels, cfg.dropout, cfg.pretrained).to(device)

    print(f"Device            : {device}")
    print(f"Architecture      : {cfg.arch} ({count_parameters(model):,} parametres)")
    print(f"Classes ({num_classes})      : {classes}")
    print(f"Images train/val  : {len(train_loader.dataset)} / {len(val_loader.dataset)}")

    weights = None
    if cfg.use_class_weights:
        weights = compute_class_weights(
            train_targets, num_classes, cfg.class_weight_power).to(device)
        print("Poids de classes  : " + ", ".join(f"{c}={w:.2f}" for c, w in zip(classes, weights)))

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=max(cfg.warmup_epochs, 1))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.epochs - cfg.warmup_epochs, 1), eta_min=cfg.lr * 0.01)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[max(cfg.warmup_epochs, 1)])
    scaler = torch.amp.GradScaler(device.type, enabled=cfg.amp and device.type == "cuda")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_acc, best_epoch, epochs_without_gain = 0.0, 0, 0

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device,
                                    optimizer, scaler, cfg.amp)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss); history["val_acc"].append(va_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        flag = ""
        if va_acc > best_acc:
            best_acc, best_epoch, epochs_without_gain = va_acc, epoch, 0
            save_checkpoint(run_dir / "best_model.pt", model, optimizer, epoch, best_acc,
                            {**cfg.to_dict(), "classes": classes, "in_channels": in_channels})
            flag = "  <- meilleur modele sauvegarde"
        else:
            epochs_without_gain += 1

        print(f"Epoch {epoch:3d}/{cfg.epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | "
              f"{time.time() - t0:.1f}s{flag}")

        if epochs_without_gain >= cfg.early_stopping_patience:
            print(f"Early stopping : aucune amelioration depuis {epochs_without_gain} epochs.")
            break

    save_checkpoint(run_dir / "last_model.pt", model, optimizer, epoch, best_acc,
                    {**cfg.to_dict(), "classes": classes, "in_channels": in_channels})
    save_json(run_dir / "history.json", history)
    save_json(run_dir / "config.json", {**cfg.to_dict(), "classes": classes,
                                        "best_val_acc": best_acc, "best_epoch": best_epoch})
    plot_history(history, run_dir / "training_curves.png")

    print(f"\nMeilleure accuracy validation : {best_acc:.4f} (epoch {best_epoch})")
    print(f"Artefacts ecrits dans : {run_dir}")
    if test_loader is not None:
        print("Evaluation sur le test set :")
        print(f"  python src/evaluate.py --checkpoint {run_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
