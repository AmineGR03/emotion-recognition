"""Architectures de reconnaissance d'emotions faciales."""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    from .config import NUM_CLASSES
except ImportError:
    from config import NUM_CLASSES


class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU (x2) -> MaxPool -> Dropout."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.block(x)


class EmotionCNN(nn.Module):
    """CNN maison pour images 48x48 en niveaux de gris (style VGG, ~4M parametres).

    48 -> 24 -> 12 -> 6 -> 3 puis global average pooling.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, in_channels: int = 1, dropout: float = 0.4):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 64, dropout * 0.5),
            ConvBlock(64, 128, dropout * 0.6),
            ConvBlock(128, 256, dropout * 0.7),
            ConvBlock(256, 512, dropout * 0.7),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


class TinyEmotionCNN(nn.Module):
    """CNN leger pour images 48x48 en niveaux de gris (~144k parametres).

    Trois choix expliquent l'ecart de cout avec EmotionCNN :

    1. Reduction spatiale des le premier bloc. EmotionCNN garde 64 canaux en
       48x48 sur deux convolutions, ce qui coute a lui seul ~86 MMAC. Ici on
       descend a 24x24 apres une seule convolution a 16 canaux.
    2. Canaux divises par quatre (16/32/64/128 au lieu de 64/128/256/512).
    3. Convolutions classiques et non separables : sur CPU les convolutions
       "depthwise" sont mal optimisees et annulent souvent leur avantage
       theorique.

    Resultat : ~19 MMAC par image contre ~470 pour EmotionCNN, soit 25 fois
    moins de calcul, pour 33 fois moins de parametres.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, in_channels: int = 1, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            # 48 -> 24
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 24 -> 12
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(dropout * 0.5),
            # 12 -> 6
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(dropout * 0.75),
            # 6 -> 3
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(dropout),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def _adapt_first_conv(conv: nn.Conv2d) -> nn.Conv2d:
    """Transforme une conv RGB pre-entrainee en conv 1 canal (somme des poids)."""
    new = nn.Conv2d(1, conv.out_channels, conv.kernel_size, conv.stride,
                    conv.padding, bias=conv.bias is not None)
    with torch.no_grad():
        new.weight.copy_(conv.weight.sum(dim=1, keepdim=True))
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


def build_model(arch: str = "cnn", num_classes: int = NUM_CLASSES, in_channels: int = 1,
                dropout: float = 0.4, pretrained: bool = True) -> nn.Module:
    """Fabrique le modele demande.

    - "tiny"            : CNN leger (~144k param.), rapide sur CPU, entree 48x48
    - "cnn"             : CNN maison, entrainable de zero sur CPU, entree 48x48
    - "resnet18"        : transfer learning ImageNet, entree >= 96x96 recommandee
    - "efficientnet_b0" : idem, plus precis mais plus lourd
    """
    arch = arch.lower()
    if arch == "cnn":
        return EmotionCNN(num_classes, in_channels, dropout)
    if arch == "tiny":
        return TinyEmotionCNN(num_classes, in_channels, dropout)

    from torchvision import models

    if arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
        if in_channels == 1:
            net.conv1 = _adapt_first_conv(net.conv1)
        net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(net.fc.in_features, num_classes))
        return net

    if arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        net = models.efficientnet_b0(weights=weights)
        if in_channels == 1:
            net.features[0][0] = _adapt_first_conv(net.features[0][0])
        in_f = net.classifier[1].in_features
        net.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
        return net

    raise ValueError(f"Architecture inconnue: {arch} (tiny | cnn | resnet18 | efficientnet_b0)")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
