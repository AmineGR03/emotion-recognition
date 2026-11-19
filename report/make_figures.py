"""Genere les figures du rapport a partir des donnees reelles du projet."""
import sys, glob, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))
EMO = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
FR = {"angry":"Colere","disgust":"Degout","fear":"Peur","happy":"Joie",
      "neutral":"Neutre","sad":"Tristesse","surprise":"Surprise"}

# ---------- 1. Distribution des classes ----------
counts = {s: [len(glob.glob(str(ROOT/"data/fer2013"/s/e/"*.png"))) for e in EMO]
          for s in ("train","test")}
x = np.arange(len(EMO)); w = 0.38
fig, ax = plt.subplots(figsize=(7.2,3.4))
b1 = ax.bar(x-w/2, counts["train"], w, label="Train (28 709)", color="#3b6ea5")
b2 = ax.bar(x+w/2, counts["test"],  w, label="Test (7 178)",  color="#d98a3b")
ax.bar_label(b1, fontsize=7, padding=1); ax.bar_label(b2, fontsize=7, padding=1)
ax.set_xticks(x, [FR[e] for e in EMO], rotation=20)
ax.set_ylabel("Nombre d'images"); ax.set_ylim(0, 8200)
ax.set_title("Distribution des classes FER2013")
ax.legend(frameon=False); ax.spines[["top","right"]].set_visible(False)
ax.grid(axis="y", alpha=.25)
fig.tight_layout(); fig.savefig(FIG/"distribution.pdf"); plt.close(fig)

# ---------- 2. Echantillons reels ----------
fig, axes = plt.subplots(len(EMO), 6, figsize=(6.2, 7.4))
for r, e in enumerate(EMO):
    files = sorted(glob.glob(str(ROOT/"data/fer2013/train"/e/"*.png")))[:6]
    for c in range(6):
        ax = axes[r, c]; ax.axis("off")
        if c < len(files):
            ax.imshow(np.array(Image.open(files[c])), cmap="gray", vmin=0, vmax=255)
    axes[r,0].set_ylabel(FR[e])
    axes[r,0].axis("on"); axes[r,0].set_xticks([]); axes[r,0].set_yticks([])
    for s in axes[r,0].spines.values(): s.set_visible(False)
fig.suptitle("Echantillons FER2013 (48x48, niveaux de gris)", y=.995, fontsize=11)
fig.tight_layout(); fig.savefig(FIG/"samples.pdf"); plt.close(fig)

# ---------- 3. Effet des augmentations ----------
from config import TrainConfig
from dataset import build_transforms, GRAY_MEAN, GRAY_STD
cfg = TrainConfig(data_dir=ROOT/"data/fer2013")
tf = build_transforms(cfg, train=True)
src = Image.open(sorted(glob.glob(str(ROOT/"data/fer2013/train/happy/*.png")))[3])
fig, axes = plt.subplots(2, 7, figsize=(8.2, 2.6))
axes[0,0].imshow(np.array(src), cmap="gray", vmin=0, vmax=255)
axes[0,0].set_title("original", fontsize=8)
for i, ax in enumerate(axes.ravel()[1:], 1):
    t = tf(src).numpy()[0]
    t = t*GRAY_STD[0] + GRAY_MEAN[0]
    ax.imshow(np.clip(t,0,1), cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"aug. {i}", fontsize=8)
for ax in axes.ravel(): ax.axis("off")
fig.suptitle("Pipeline d'augmentation applique a une meme image", fontsize=10)
fig.tight_layout(); fig.savefig(FIG/"augmentation.pdf"); plt.close(fig)

# ---------- 4. Histogramme d'intensite ----------
vals = []
for e in EMO:
    for f in sorted(glob.glob(str(ROOT/"data/fer2013/train"/e/"*.png")))[:300]:
        vals.append(np.array(Image.open(f), dtype=np.float32).ravel()/255.)
vals = np.concatenate(vals)
fig, ax = plt.subplots(figsize=(6.4,2.8))
ax.hist(vals, bins=64, color="#3b6ea5", alpha=.85, density=True)
ax.axvline(0.5077, color="#c0392b", ls="--", lw=1.4, label="moyenne du jeu complet = 0,5077")
ax.axvspan(0.5077-0.2550, 0.5077+0.2550, color="#c0392b", alpha=.08,
           label="+/- ecart-type = 0,2550")
ax.set_xlabel("Intensite normalisee"); ax.set_ylabel("Densite")
ax.set_title("Intensites de pixels (echantillon de 2 100 images train)")
ax.legend(frameon=False, fontsize=8); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig(FIG/"histogram.pdf"); plt.close(fig)

# ---------- 5. Poids de classes ----------
from dataset import compute_class_weights
targets = np.concatenate([[i]*counts["train"][i] for i in range(7)])
wgt = compute_class_weights(targets, 7).numpy()
fig, ax = plt.subplots(figsize=(6.4,2.8))
bars = ax.bar([FR[e] for e in EMO], wgt, color=["#c0392b" if v>2 else "#3b6ea5" for v in wgt])
ax.bar_label(bars, fmt="%.2f", fontsize=8)
ax.axhline(1.0, color="gray", ls=":", lw=1)
ax.set_ylabel("Poids dans la loss"); ax.set_ylim(0, 10.6)
ax.set_title("Ponderation inverse a la frequence des classes")
ax.tick_params(axis="x", rotation=20); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig(FIG/"weights.pdf"); plt.close(fig)

print("Figures generees :")
for f in sorted(FIG.glob("*.pdf")): print("  ", f.name, f"{f.stat().st_size/1024:.0f} Ko")
json.dump({"counts": counts, "weights": wgt.tolist(),
           "pixel_mean": float(vals.mean()), "pixel_std": float(vals.std())},
          open(Path(__file__).parent/"stats.json","w"), indent=2)
