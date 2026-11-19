"""Exporte un checkpoint d'entrainement en version allegee, prete a deployer.

Un checkpoint d'entrainement contient l'etat de l'optimiseur AdamW, qui garde
deux moments par parametre : il pese donc environ trois fois le modele lui-meme.
Cet etat ne sert qu'a reprendre un entrainement interrompu, jamais a predire.

    python scripts/export_model.py outputs/balanced/best_model.pt modele.pt
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", type=Path, help="checkpoint d'entrainement")
    p.add_argument("destination", type=Path, help="fichier allege a creer")
    a = p.parse_args()

    if not a.source.exists():
        raise SystemExit(f"Introuvable : {a.source}")

    ck = torch.load(a.source, map_location="cpu", weights_only=False)
    manquant = [k for k in ("model_state", "config") if k not in ck]
    if manquant:
        raise SystemExit(f"Checkpoint incomplet, cles absentes : {manquant}")

    # on ne garde que ce dont la prediction a besoin
    leger = {"model_state": ck["model_state"], "config": ck["config"]}
    for cle in ("classes", "epoch", "best_acc"):
        if cle in ck:
            leger[cle] = ck[cle]

    a.destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(leger, a.destination)

    avant = a.source.stat().st_size / 1e6
    apres = a.destination.stat().st_size / 1e6
    print(f"Source      : {a.source}  ({avant:.1f} Mo)")
    print(f"Destination : {a.destination}  ({apres:.1f} Mo)")
    print(f"Reduction   : {100 * (1 - apres / avant):.0f} %")
    if "best_acc" in leger:
        print(f"Exactitude validation du modele exporte : {leger['best_acc']:.4f}")


if __name__ == "__main__":
    main()
