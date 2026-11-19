"""Detection de visages, compatible OpenCV 4 et OpenCV 5.

Trois backends, essayes dans cet ordre :
  1. YuNet (reseau de neurones, cv2.FaceDetectorYN) si le modele ONNX est present
     -> le plus precis, seul backend disponible sous OpenCV >= 5
  2. Cascade de Haar (cv2.CascadeClassifier) -> OpenCV 4 uniquement
  3. Aucun detecteur : l'image entiere est traitee comme un visage

Telecharger le modele YuNet :
    python scripts/download_face_model.py
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
YUNET_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"

_backend = None  # ("yunet", detector) | ("haar", cascade) | ("none", None)


def _try_yunet():
    if not hasattr(cv2, "FaceDetectorYN") or not YUNET_PATH.exists():
        return None
    try:
        det = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320),
                                        score_threshold=0.75, nms_threshold=0.3, top_k=50)
        return ("yunet", det)
    except cv2.error:
        return None


def _try_haar():
    if not hasattr(cv2, "CascadeClassifier"):
        return None  # supprime dans OpenCV 5
    path = getattr(getattr(cv2, "data", None), "haarcascades", "") + \
        "haarcascade_frontalface_default.xml"
    if not Path(path).exists():
        return None
    cascade = cv2.CascadeClassifier(path)
    return None if cascade.empty() else ("haar", cascade)


def get_backend():
    """Retourne (nom, objet) du backend de detection disponible."""
    global _backend
    if _backend is None:
        _backend = _try_yunet() or _try_haar() or ("none", None)
        if _backend[0] == "none":
            print("[face_detect] Aucun detecteur de visage disponible "
                  "(OpenCV 5 sans modele YuNet). L'image entiere sera classee.\n"
                  "              Lancez : python scripts/download_face_model.py")
    return _backend


def detect_faces(bgr_image: np.ndarray, scale_factor: float = 1.1,
                 min_neighbors: int = 5, min_size: int = 48) -> list[tuple[int, int, int, int]]:
    """Retourne une liste de boites (x, y, w, h) triees par taille decroissante."""
    name, det = get_backend()
    if name == "yunet":
        h, w = bgr_image.shape[:2]
        det.setInputSize((w, h))
        _, faces = det.detect(bgr_image)
        if faces is None:
            return []
        boxes = []
        for f in faces:
            x, y, bw, bh = (int(round(v)) for v in f[:4])
            if bw >= min_size and bh >= min_size:
                boxes.append((max(x, 0), max(y, 0), bw, bh))
    elif name == "haar":
        gray = cv2.equalizeHist(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY))
        faces = det.detectMultiScale(gray, scaleFactor=scale_factor,
                                     minNeighbors=min_neighbors,
                                     minSize=(min_size, min_size))
        boxes = [tuple(int(v) for v in f) for f in faces]
    else:
        return []
    return sorted(boxes, key=lambda b: -b[2] * b[3])


def crop_face(bgr_image: np.ndarray, box, margin: float = 0.15) -> np.ndarray:
    """Recadre le visage avec une marge, en restant dans l'image."""
    x, y, w, h = box
    mx, my = int(w * margin), int(h * margin)
    x0, y0 = max(x - mx, 0), max(y - my, 0)
    x1 = min(x + w + mx, bgr_image.shape[1])
    y1 = min(y + h + my, bgr_image.shape[0])
    crop = bgr_image[y0:y1, x0:x1]
    return crop if crop.size else bgr_image
