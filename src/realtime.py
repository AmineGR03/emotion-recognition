"""Reconnaissance d'emotions en temps reel sur la webcam.

    python src/realtime.py --checkpoint outputs/baseline/best_model.pt
    (q ou Echap pour quitter, s pour sauvegarder une capture)
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import EMOTIONS_FR  # noqa: E402
from face_detect import crop_face, detect_faces  # noqa: E402
from predict import COLORS, EmotionPredictor  # noqa: E402


class Smoother:
    """Moyenne glissante des probabilites : evite le clignotement des predictions."""

    def __init__(self, window: int = 7, num_classes: int = 7):
        self.buffer = deque(maxlen=window)
        self.num_classes = num_classes

    def update(self, probs: np.ndarray) -> np.ndarray:
        self.buffer.append(probs)
        return np.mean(self.buffer, axis=0)


def draw_bars(frame, probs: dict, origin=(10, 70), width=170, height=18):
    """Barres de probabilite par emotion, en haut a gauche."""
    x, y = origin
    for i, (label, value) in enumerate(sorted(probs.items(), key=lambda kv: -kv[1])):
        yy = y + i * (height + 6)
        cv2.rectangle(frame, (x, yy), (x + width, yy + height), (60, 60, 60), -1)
        cv2.rectangle(frame, (x, yy), (x + int(width * value), yy + height),
                      COLORS.get(label, (0, 200, 0)), -1)
        cv2.putText(frame, f"{EMOTIONS_FR.get(label, label)} {value * 100:4.1f}%",
                    (x + 6, yy + height - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)


def main() -> None:
    p = argparse.ArgumentParser(description="Emotions faciales en temps reel (webcam)")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--smooth", type=int, default=7, help="taille de la fenetre de lissage")
    p.add_argument("--detect-every", type=int, default=3,
                   help="detecter un visage toutes les N frames (perf)")
    p.add_argument("--save-dir", type=Path, default=Path("outputs/captures"))
    args = p.parse_args()

    predictor = EmotionPredictor(args.checkpoint)
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if sys.platform == "win32" else 0)
    if not cap.isOpened():
        raise SystemExit(f"Webcam {args.camera} inaccessible. Essayez --camera 1.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)

    smoother = Smoother(args.smooth, len(predictor.classes))
    boxes: list = []
    frame_id, fps, t_prev = 0, 0.0, time.time()
    print("Appuyez sur 'q' ou Echap pour quitter, 's' pour sauvegarder une image.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # effet miroir

        if frame_id % max(args.detect_every, 1) == 0 or not boxes:
            boxes = detect_faces(frame, min_size=60)
        frame_id += 1

        main_probs = None
        for i, box in enumerate(sorted(boxes, key=lambda b: -b[2] * b[3])):
            res = predictor.predict_face(crop_face(frame, box))
            probs = np.array([res["probabilities"][c] for c in predictor.classes])
            if i == 0:  # lissage sur le plus grand visage seulement
                probs = smoother.update(probs)
                main_probs = {c: float(v) for c, v in zip(predictor.classes, probs)}
            idx = int(probs.argmax())
            label, conf = predictor.classes[idx], float(probs[idx])
            x, y, w, h = box
            color = COLORS.get(label, (0, 255, 0))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f"{EMOTIONS_FR.get(label, label)} {conf * 100:.0f}%"
            cv2.putText(frame, text, (x, max(y - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2, cv2.LINE_AA)

        if main_probs:
            draw_bars(frame, main_probs)

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
        t_prev = now
        cv2.putText(frame, f"{fps:4.1f} FPS | {len(boxes)} visage(s)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Reconnaissance d'emotions faciales", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            args.save_dir.mkdir(parents=True, exist_ok=True)
            out = args.save_dir / f"capture_{int(time.time())}.jpg"
            cv2.imwrite(str(out), frame)
            print(f"Capture enregistree : {out}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
