"""Convertit le CSV FER2013 en arborescence ImageFolder utilisable par src/train.py.

    python scripts/prepare_fer2013.py --csv data/_download/fer2013.csv.zip --out data/fer2013

Accepte directement une archive .zip contenant fer2013.csv, ou le CSV decompresse.

Sources du CSV :
  - Kaggle (officiel)  : https://www.kaggle.com/datasets/deadskull7/fer2013
  - Miroir public (MIT): https://huggingface.co/datasets/chitradrishti/fer2013
                         fichier fer2013.csv.zip

Produit :
  data/fer2013/train/<emotion>/*.png   (28 709 images, Usage=Training)
  data/fer2013/test/<emotion>/*.png    ( 7 178 images, Public+PrivateTest)

Si vous avez deja un dataset en dossiers (CK+, RAF-DB, AffectNet...), ce script
est inutile : pointez simplement --data-dir dessus.
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

# Mapping officiel FER2013
FER_LABELS = {0: "angry", 1: "disgust", 2: "fear", 3: "happy",
              4: "sad", 5: "surprise", 6: "neutral"}


def open_csv(path: Path):
    """Retourne un flux texte sur le CSV, que l'entree soit un .csv ou un .zip."""
    if path.suffix.lower() == ".zip":
        zf = zipfile.ZipFile(path)
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            sys.exit(f"Aucun .csv dans l'archive {path}")
        print(f"Archive : {path.name} -> {names[0]}")
        return io.TextIOWrapper(zf.open(names[0]), encoding="utf-8"), zf
    return path.open("r", encoding="utf-8"), None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True, help="fer2013.csv ou fer2013.csv.zip")
    p.add_argument("--out", type=Path, default=Path("data/fer2013"))
    p.add_argument("--limit", type=int, default=0, help="ne traiter que N lignes (test rapide)")
    args = p.parse_args()

    if not args.csv.exists():
        sys.exit(f"Fichier introuvable : {args.csv}")

    stream, zf = open_csv(args.csv)
    counts: dict[str, int] = {}
    written = skipped = 0

    try:
        reader = csv.DictReader(stream)
        cols = {c.lower(): c for c in (reader.fieldnames or [])}
        if "emotion" not in cols or "pixels" not in cols:
            sys.exit(f"Colonnes inattendues : {reader.fieldnames} "
                     "(attendu 'emotion', 'pixels' et idealement 'Usage')")
        c_emo, c_pix = cols["emotion"], cols["pixels"]
        c_use = cols.get("usage")

        # Pre-creation des dossiers : evite 36 000 appels a mkdir
        for split in ("train", "test"):
            for label in FER_LABELS.values():
                (args.out / split / label).mkdir(parents=True, exist_ok=True)

        for i, row in enumerate(reader):
            if args.limit and i >= args.limit:
                break
            try:
                label = FER_LABELS[int(row[c_emo])]
                pixels = np.array(row[c_pix].split(), dtype=np.uint8)
                if pixels.size != 48 * 48:
                    raise ValueError(f"{pixels.size} pixels au lieu de 2304")
            except (KeyError, ValueError) as e:
                skipped += 1
                if skipped <= 5:
                    print(f"  [!] ligne {i + 2} ignoree : {e}")
                continue

            split = "train" if (c_use is None or str(row[c_use]).strip() == "Training") else "test"
            Image.fromarray(pixels.reshape(48, 48), mode="L").save(
                args.out / split / label / f"{split}_{i:06d}.png")
            key = f"{split}/{label}"
            counts[key] = counts.get(key, 0) + 1
            written += 1
            if written % 5000 == 0:
                print(f"  {written} images ecrites...")
    finally:
        stream.close()
        if zf:
            zf.close()

    print(f"\n{written} images ecrites, {skipped} ligne(s) ignoree(s). Repartition :")
    total = {"train": 0, "test": 0}
    for k in sorted(counts):
        print(f"  {k:22s} {counts[k]:6d}")
        total[k.split('/')[0]] += counts[k]
    print(f"  {'TOTAL train':22s} {total['train']:6d}")
    print(f"  {'TOTAL test':22s} {total['test']:6d}")
    print(f"\nDataset pret : {args.out}")
    print(f"Entrainement : python src/train.py --data-dir {args.out} --epochs 60")


if __name__ == "__main__":
    main()
