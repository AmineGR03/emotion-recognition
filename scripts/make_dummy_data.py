"""Genere un mini jeu de donnees synthetique pour verifier que tout le pipeline
tourne (entrainement, evaluation, prediction) sans telecharger FER2013.

    python scripts/make_dummy_data.py --out data/dummy --per-class 40
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def make_image(class_idx: int, rng: np.random.Generator, size: int = 48) -> Image.Image:
    """Motif different par classe + bruit : apprenable mais sans aucun sens reel."""
    yy, xx = np.mgrid[0:size, 0:size] / size
    freq = 2 + class_idx
    base = 0.5 + 0.35 * np.sin(freq * np.pi * xx + class_idx) * np.cos(freq * np.pi * yy)
    noise = rng.normal(0, 0.08, (size, size))
    arr = np.clip(base + noise, 0, 1) * 255
    return Image.fromarray(arr.astype(np.uint8), mode="L")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/dummy"))
    p.add_argument("--per-class", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    for split, n in (("train", args.per_class), ("test", max(args.per_class // 4, 4))):
        for ci, emo in enumerate(EMOTIONS):
            d = args.out / split / emo
            d.mkdir(parents=True, exist_ok=True)
            for k in range(n):
                make_image(ci, rng).save(d / f"{emo}_{k:04d}.png")
    print(f"Jeu factice ecrit dans {args.out} "
          f"({len(EMOTIONS)} classes x {args.per_class} images d'entrainement)")


if __name__ == "__main__":
    main()
