"""Prediction de l'emotion sur une image (ou un dossier d'images).

Exemples:
    python src/predict.py --checkpoint outputs/baseline/best_model.pt --image photo.jpg
    python src/predict.py --checkpoint outputs/baseline/best_model.pt --image dossier/ --save-annotated outputs/annotes
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import EMOTIONS_FR  # noqa: E402
from face_detect import crop_face, detect_faces  # noqa: E402
from utils import get_device, load_checkpoint  # noqa: E402
from model import build_model  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COLORS = {  # BGR
    "angry": (60, 60, 220), "disgust": (90, 160, 90), "fear": (150, 80, 160),
    "happy": (60, 200, 250), "neutral": (190, 190, 190), "sad": (220, 150, 70),
    "surprise": (230, 120, 230),
}


class EmotionPredictor:
    """Charge un checkpoint et predit l'emotion d'un visage (image BGR OpenCV)."""

    def __init__(self, checkpoint: Path, device=None):
        self.device = device or get_device()
        ckpt = load_checkpoint(Path(checkpoint), map_location=self.device)
        cfg = ckpt.get("config", {})
        self.classes = cfg["classes"]
        self.image_size = int(cfg.get("image_size", 48))
        self.grayscale = bool(cfg.get("grayscale", True))
        self.in_channels = 1 if self.grayscale else 3
        self.model = build_model(cfg.get("arch", "cnn"), len(self.classes),
                                 self.in_channels, float(cfg.get("dropout", 0.4)),
                                 pretrained=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device).eval()
        if self.grayscale:
            self.mean, self.std = np.array([0.5077]), np.array([0.2550])
        else:
            self.mean = np.array([0.485, 0.456, 0.406])
            self.std = np.array([0.229, 0.224, 0.225])

    def _preprocess(self, face_bgr: np.ndarray) -> torch.Tensor:
        img = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY if self.grayscale else cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        arr = img.astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = arr[..., None]
        arr = (arr - self.mean) / self.std
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()
    def predict_face(self, face_bgr: np.ndarray) -> dict:
        probs = torch.softmax(self.model(self._preprocess(face_bgr)), dim=1)[0].cpu().numpy()
        idx = int(probs.argmax())
        return {
            "label": self.classes[idx],
            "label_fr": EMOTIONS_FR.get(self.classes[idx], self.classes[idx]),
            "confidence": float(probs[idx]),
            "probabilities": {c: float(p) for c, p in zip(self.classes, probs)},
        }

    def predict_image(self, bgr_image: np.ndarray, detect: bool = True) -> list[dict]:
        """Detecte les visages puis predit. Si aucun visage detecte, utilise l'image entiere."""
        boxes = detect_faces(bgr_image) if detect else []
        if not boxes:
            res = self.predict_face(bgr_image)
            res["box"] = (0, 0, bgr_image.shape[1], bgr_image.shape[0])
            res["face_detected"] = False
            return [res]
        results = []
        for box in boxes:
            res = self.predict_face(crop_face(bgr_image, box))
            res["box"] = box
            res["face_detected"] = True
            results.append(res)
        return results


def annotate(image: np.ndarray, results: list[dict]) -> np.ndarray:
    out = image.copy()
    for r in results:
        x, y, w, h = r["box"]
        color = COLORS.get(r["label"], (0, 255, 0))
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        text = f"{r['label_fr']} {r['confidence'] * 100:.0f}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x, max(y - th - 10, 0)), (x + tw + 8, y), color, -1)
        cv2.putText(out, text, (x + 4, max(y - 6, th)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (20, 20, 20), 2, cv2.LINE_AA)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Prediction d'emotion sur image(s)")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True, help="fichier image ou dossier")
    p.add_argument("--no-detect", action="store_true", help="ne pas detecter de visage")
    p.add_argument("--save-annotated", type=Path, default=None, help="dossier de sortie")
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args()

    paths = sorted(q for q in args.image.rglob("*") if q.suffix.lower() in IMG_EXT) \
        if args.image.is_dir() else [args.image]
    if not paths:
        raise SystemExit(f"Aucune image trouvee dans {args.image}")

    predictor = EmotionPredictor(args.checkpoint)
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"[!] Illisible : {path}")
            continue
        results = predictor.predict_image(image, detect=not args.no_detect)
        print(f"\n{path.name} - {len(results)} visage(s)")
        for i, r in enumerate(results, 1):
            top = sorted(r["probabilities"].items(), key=lambda kv: -kv[1])[:args.top_k]
            detail = ", ".join(f"{EMOTIONS_FR.get(c, c)} {v * 100:.1f}%" for c, v in top)
            flag = "" if r["face_detected"] else "  (aucun visage detecte, image entiere)"
            print(f"  Visage {i}: {r['label_fr']} ({r['confidence'] * 100:.1f}%) | {detail}{flag}")
        if args.save_annotated:
            args.save_annotated.mkdir(parents=True, exist_ok=True)
            out_path = args.save_annotated / f"{path.stem}_annote.jpg"
            cv2.imwrite(str(out_path), annotate(image, results))
            print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
