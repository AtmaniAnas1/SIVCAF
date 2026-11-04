"""Chargement des documents et préparation des entrées du CNN."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import TAILLE_ENTREE


def charger_image(chemin: str | Path) -> np.ndarray:
    """Charge une image (ou la 1re page d'un PDF) en RGB uint8."""
    chemin = Path(chemin)
    if chemin.suffix.lower() == ".pdf":
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(str(chemin))
        page = pdf[0]
        img = np.array(page.render(scale=200 / 72).to_pil().convert("RGB"))
        pdf.close()
        return img
    donnees = np.fromfile(str(chemin), dtype=np.uint8)  # gère les chemins non ASCII sous Windows
    img = cv2.imdecode(donnees, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Image illisible : {chemin.name}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def redimensionner_max(img: np.ndarray, cote_max: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = cote_max / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def vignette_detail(img: np.ndarray, taille: int = TAILLE_ENTREE, cote_ref: int = 896) -> np.ndarray:
    """Extrait, à résolution quasi native, la zone 224x224 la plus riche en contours.

    Le flou se juge sur le détail des caractères : on cherche donc la zone
    la plus texturée de l'image ramenée à 896 px, en niveaux de gris.
    """
    gris = cv2.cvtColor(redimensionner_max(img, cote_ref), cv2.COLOR_RGB2GRAY)
    h, w = gris.shape
    if h < taille or w < taille:
        gris = cv2.copyMakeBorder(gris, 0, max(0, taille - h), 0, max(0, taille - w), cv2.BORDER_REPLICATE)
        h, w = gris.shape
    energie = np.abs(cv2.Sobel(gris, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(gris, cv2.CV_32F, 0, 1))
    integrale = cv2.integral(energie)
    pas = taille // 2
    meilleur, pos = -1.0, (0, 0)
    for y in range(0, h - taille + 1, pas):
        for x in range(0, w - taille + 1, pas):
            e = integrale[y + taille, x + taille] - integrale[y, x + taille] - integrale[y + taille, x] + integrale[y, x]
            if e > meilleur:
                meilleur, pos = e, (y, x)
    y, x = pos
    return gris[y:y + taille, x:x + taille]


def entrees_cnn(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Renvoie (vue globale RGB 224x224, vignette de détail gris 224x224) en uint8."""
    globale = cv2.resize(img, (TAILLE_ENTREE, TAILLE_ENTREE), interpolation=cv2.INTER_AREA)
    return globale, vignette_detail(img)
