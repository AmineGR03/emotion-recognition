"""Telecharge le modele de detection de visage YuNet (~230 Ko) depuis OpenCV Zoo.

    python scripts/download_face_model.py

Necessaire avec OpenCV >= 5 (les cascades de Haar y ont ete supprimees).
Sous OpenCV 4, le repli Haar fonctionne sans telechargement.

Note : opencv_zoo stocke ses modeles via Git LFS. L'URL raw.githubusercontent.com
ne renvoie que le pointeur LFS (131 octets) : il faut passer par
media.githubusercontent.com/media/... pour obtenir le fichier reel.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

URL = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
       "face_detection_yunet/face_detection_yunet_2023mar.onnx")
SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
EXPECTED_SIZE = 232589
DEST = Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx"


def check(path: Path) -> bool:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest == SHA256


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists() and check(DEST):
        print(f"Deja present et verifie : {DEST} ({DEST.stat().st_size / 1024:.0f} Ko)")
        return

    print(f"Telechargement depuis {URL}")
    urllib.request.urlretrieve(URL, DEST)
    size = DEST.stat().st_size

    if size < 1000:
        DEST.unlink()
        sys.exit(f"Echec : {size} octets recus (pointeur Git LFS). URL invalide ?")
    if not check(DEST):
        sys.exit(f"Echec : empreinte SHA256 incorrecte (attendu {SHA256}).")

    print(f"Modele enregistre et verifie : {DEST} ({size / 1024:.0f} Ko)")


if __name__ == "__main__":
    main()
