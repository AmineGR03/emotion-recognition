# Reconnaissance d'émotions faciales (Deep Learning)

Classification des émotions sur visage : entraînement d'un CNN, évaluation,
prédiction sur image et **détection en temps réel sur la webcam**.

7 émotions : `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`.

**Résultat obtenu** : 58,87 % d'exactitude sur les 7 178 images de test de
FER2013 (77,42 % en tenant compte des deux premières propositions), avec un
réseau de 144 567 paramètres entraîné en 12 minutes sur processeur central.

---

## 1. Installation

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (Linux/macOS : source .venv/bin/activate)
pip install -r requirements.txt
python scripts/download_face_model.py   # détecteur de visage YuNet (233 Ko)
```

Pour un GPU NVIDIA, installez la variante CUDA de PyTorch depuis
<https://pytorch.org>. Elle n'est pas nécessaire : le réseau léger s'entraîne
en 12 minutes sur processeur.

## 2. Utilisation immédiate

Le modèle entraîné est fourni (`modele_leger.pt`, 0,6 Mo). Aucun entraînement
n'est requis pour l'utiliser :

```bash
python src/realtime.py --checkpoint modele_leger.pt          # webcam
python src/predict.py  --checkpoint modele_leger.pt --image photo.jpg
```

`q` ou `Échap` pour quitter la webcam, `s` pour enregistrer une capture. Les
probabilités sont lissées sur une fenêtre glissante (`--smooth`) pour éviter le
clignotement, et la détection de visage ne tourne qu'une image sur trois
(`--detect-every`).

## 3. Données

### Option A — FER2013

1. Téléchargez `fer2013.csv` : <https://www.kaggle.com/datasets/deadskull7/fer2013>
2. Convertissez-le en arborescence d'images :

```bash
python scripts/prepare_fer2013.py --csv data/fer2013.csv --out data/fer2013
```

Vous obtenez 35 887 images 48×48 en niveaux de gris :

```
data/fer2013/
├── train/{angry,disgust,fear,happy,neutral,sad,surprise}/*.png   (28 709)
└── test/ {angry,disgust,fear,happy,neutral,sad,surprise}/*.png   ( 7 178)
```

### Option B — tout jeu au format ImageFolder

CK+, RAF-DB, AffectNet ou vos propres images : même structure
`train/<emotion>/*.jpg`, puis `--data-dir`.

### Option C — jeu factice, pour tester la chaîne en deux minutes

```bash
python scripts/make_dummy_data.py --out data/dummy --per-class 40
```

## 4. Entraînement

```bash
# Réseau léger (recommandé) : 144 567 paramètres, ~12 s par époque sur CPU
python src/train.py --data-dir data/fer2013 --arch tiny --epochs 60 --weight-power 0.5 --run-name tiny

# Réseau complet : 4 819 783 paramètres, ~312 s par époque sur CPU
python src/train.py --data-dir data/fer2013 --arch cnn --epochs 60 --weight-power 0.5 --run-name cnn

# Transfert ImageNet (nécessite un GPU en pratique)
python src/train.py --data-dir data/fer2013 --arch resnet18 --image-size 96 --lr 3e-4 --run-name resnet18
```

Options utiles : `--batch-size`, `--dropout`, `--weight-power`,
`--no-class-weights`, `--patience`, `--rgb`, `--num-workers`, `--seed`.

Artefacts écrits dans `outputs/<run-name>/` : `best_model.pt`, `last_model.pt`,
`history.json`, `config.json`, `training_curves.png`.

### Les deux architectures

| | `cnn` | `tiny` |
|---|---|---|
| Paramètres | 4 819 783 | **144 567** |
| Durée d'une époque (CPU) | 312 s | **12,3 s** |
| Exactitude test | 55,71 %\* | **58,87 %** |

\* entraînement interrompu à l'époque 15 sur 60 : la comparaison n'est pas à
l'avantage du réseau complet, qui n'a jamais convergé. Le point à retenir est
qu'un réseau que l'on peut entraîner jusqu'au bout vaut mieux qu'un réseau
surdimensionné que l'on ne peut pas.

### Ce qui est mis en place pendant l'entraînement

| Technique | Pourquoi |
|---|---|
| Augmentation (recadrage, miroir, rotation, cisaillement, jitter, effacement) | FER2013 est petit, le réseau sur-apprend vite |
| Pondération des classes adoucie (`--weight-power 0.5`) | `disgust` ne représente que 1,5 % des images, mais l'inverse pur le fait sur-prédire |
| Lissage d'étiquettes 0,1 | les émotions sont ambiguës ; attention, cela rend le réseau sous-confiant |
| AdamW + échauffement + recuit cosinus | convergence stable |
| Écrêtage du gradient, BatchNorm, Dropout2d | régularisation |
| Précision mixte (GPU seulement) | environ deux fois plus rapide |
| Arrêt anticipé + sauvegarde du meilleur modèle | pas de calcul perdu |

## 5. Évaluation

```bash
python src/evaluate.py --checkpoint outputs/tiny/best_model.pt --split test
```

Affiche l'exactitude (premier et deux premiers choix) et le rapport
précision/rappel/F1 par émotion, puis écrit `confusion_matrix_test.png` et
`metrics_test.json` à côté du checkpoint.

Résultats obtenus, reproduits dans [`results/`](results/) :

| Émotion | Précision | Rappel | F1 |
|---|---|---|---|
| Joie | 0,799 | 0,857 | **0,827** |
| Surprise | 0,655 | 0,828 | 0,731 |
| Neutre | 0,494 | 0,685 | 0,574 |
| Colère | 0,504 | 0,491 | 0,497 |
| Tristesse | 0,467 | 0,447 | 0,457 |
| Dégoût | 0,350 | 0,252 | 0,293 |
| Peur | 0,381 | 0,114 | 0,176 |

**Repère** : sur FER2013 l'accord entre annotateurs humains plafonne vers 65 %
et l'état de l'art tourne autour de 73 %. La `peur` reste le point faible — sa
signature se confond avec la surprise et la tristesse en 48×48 niveaux de gris.

## 6. Déployer sur une autre machine

Aucun réentraînement n'est nécessaire. Le checkpoint d'entraînement contient
l'état de l'optimiseur, qui pèse deux fois le modèle et ne sert qu'à reprendre
un entraînement interrompu. Le script d'export le retire :

```bash
python scripts/export_model.py outputs/tiny/best_model.pt modele_leger.pt
```

À copier sur la machine cible : `src/`, `requirements.txt`, `modele_leger.pt`
(0,6 Mo) et `models/face_detection_yunet_2023mar.onnx` (233 Ko). **Le dossier
`data/` est inutile** — il ne sert qu'à entraîner. Un modèle entraîné sur GPU
se recharge sans modification sur une machine sans carte graphique.

## 7. Structure du projet

```
emotion-recognition/
├── README.md
├── requirements.txt
├── modele_leger.pt           # modèle prêt à l'emploi (0,6 Mo)
├── src/
│   ├── config.py             # hyperparamètres, classes, chemins
│   ├── dataset.py            # ImageFolder, augmentation, découpage, poids de classes
│   ├── model.py              # TinyEmotionCNN, EmotionCNN, resnet18, efficientnet_b0
│   ├── train.py              # boucle d'entraînement
│   ├── evaluate.py           # métriques par classe, matrice de confusion
│   ├── face_detect.py        # détection de visage (YuNet, Haar, repli)
│   ├── predict.py            # inférence sur image, classe EmotionPredictor
│   ├── realtime.py           # webcam temps réel
│   └── utils.py
├── scripts/
│   ├── prepare_fer2013.py    # CSV Kaggle -> ImageFolder
│   ├── download_face_model.py# téléchargement vérifié de YuNet (SHA-256)
│   ├── export_model.py       # checkpoint -> modèle déployable
│   ├── make_dummy_data.py    # jeu synthétique
│   └── run_tiny.ps1          # lancement de l'entraînement léger
├── results/                  # métriques, courbes, matrice de confusion
├── demo/                     # exemples de prédictions annotées
├── report/                   # rapport LaTeX et PDF
├── data/                     (non versionné)
├── models/                   (non versionné, voir download_face_model.py)
└── outputs/                  (non versionné)
```

## 8. Utiliser le modèle depuis votre propre code

```python
import cv2
from src.predict import EmotionPredictor

predictor = EmotionPredictor("modele_leger.pt")
image = cv2.imread("photo.jpg")
for r in predictor.predict_image(image):
    print(r["label_fr"], round(r["confidence"], 3), r["box"])
```

## 9. Points connus et pistes d'amélioration

- **Les probabilités affichées sont sous-estimées** d'environ 9 points, à cause
  du lissage d'étiquettes. Une température de 0,701 ajustée sur la validation
  les recale (confiance moyenne 60,3 % pour 58,9 % d'exactitude réelle).
- **La `peur` est mal reconnue** (rappel 0,114). Le levier est la résolution
  d'entrée, pas la pondération.
- Entraîner `resnet18` ou `efficientnet_b0` en 96×96 et les combiner au CNN.
- Augmentation au moment du test (moyenne image + miroir) : un à deux points.
- Alignement des visages par points caractéristiques avant classification.
- Grad-CAM pour visualiser les zones du visage qui portent la décision.
- Fusionner plusieurs jeux de données (FER2013 + RAF-DB) pour mieux généraliser.

## 10. Limites et usage responsable

L'expression faciale n'est pas l'émotion ressentie : ce modèle prédit une
**expression apparente**, avec les biais connus de FER2013 selon l'âge,
l'origine et le genre des visages. À ne pas utiliser pour des décisions
concernant des personnes — recrutement, surveillance, évaluation. Usage
pédagogique, recherche et démonstration.
