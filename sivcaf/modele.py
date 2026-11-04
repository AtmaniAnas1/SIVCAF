"""Réseau de neurones convolutif multi-tâches : type de document + qualité.

Deux branches convolutives :
  * vue globale (page entière 224x224 RGB) -> mise en page, cadrage, exposition ;
  * vignette de détail (zone 224x224 en haute résolution, gris) -> netteté.
La tête "type" exploite la vue globale ; la tête "qualité" les deux branches.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .config import QUALITES, TYPES


def bloc(c_in: int, c_out: int, n_conv: int = 2) -> nn.Sequential:
    couches = []
    for i in range(n_conv):
        couches += [nn.Conv2d(c_in if i == 0 else c_out, c_out, 3, padding=1, bias=False),
                    nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
    couches.append(nn.MaxPool2d(2))
    return nn.Sequential(*couches)


class Branche(nn.Module):
    def __init__(self, canaux_entree: int, largeurs=(24, 48, 96, 160)):
        super().__init__()
        self.tige = nn.Sequential(nn.Conv2d(canaux_entree, largeurs[0], 5, stride=2, padding=2, bias=False),
                                  nn.BatchNorm2d(largeurs[0]), nn.ReLU(inplace=True))      # 112
        self.corps = nn.Sequential(bloc(largeurs[0], largeurs[1], 1),                       # 56
                                   bloc(largeurs[1], largeurs[2]),                          # 28
                                   bloc(largeurs[2], largeurs[3]),                          # 14
                                   bloc(largeurs[3], largeurs[3]))                          # 7
        self.sortie = largeurs[3]

    def forward(self, x):
        return torch.flatten(nn.functional.adaptive_avg_pool2d(self.corps(self.tige(x)), 1), 1)


class CNNFactures(nn.Module):
    def __init__(self, n_types: int = len(TYPES), n_qualites: int = len(QUALITES)):
        super().__init__()
        self.globale = Branche(3)
        self.detail = Branche(1, (16, 32, 64, 96))
        dg, dd = self.globale.sortie, self.detail.sortie
        self.tete_type = nn.Sequential(nn.Dropout(0.3), nn.Linear(dg, 128), nn.ReLU(inplace=True),
                                       nn.Dropout(0.2), nn.Linear(128, n_types))
        self.tete_qualite = nn.Sequential(nn.Dropout(0.3), nn.Linear(dg + dd, 128), nn.ReLU(inplace=True),
                                          nn.Dropout(0.2), nn.Linear(128, n_qualites))

    def forward(self, globale, detail):
        fg = self.globale(globale)
        fd = self.detail(detail)
        return self.tete_type(fg), self.tete_qualite(torch.cat([fg, fd], 1))


MOYENNE, ECART = 0.5, 0.25


def vers_tenseurs(globales: np.ndarray, details: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """uint8 (N,H,W,3) et (N,H,W) -> tenseurs float normalisés (N,C,H,W)."""
    g = torch.from_numpy(globales).permute(0, 3, 1, 2).float().div_(255).sub_(MOYENNE).div_(ECART)
    d = torch.from_numpy(details).unsqueeze(1).float().div_(255).sub_(MOYENNE).div_(ECART)
    return g, d


def sauvegarder(modele: CNNFactures, chemin, metriques: dict):
    torch.save({"etat": modele.state_dict(), "types": TYPES, "qualites": QUALITES, "metriques": metriques}, chemin)


def charger(chemin) -> tuple[CNNFactures, dict]:
    paquet = torch.load(chemin, map_location="cpu", weights_only=False)
    modele = CNNFactures(len(paquet["types"]), len(paquet["qualites"]))
    modele.load_state_dict(paquet["etat"])
    modele.eval()
    return modele, paquet
