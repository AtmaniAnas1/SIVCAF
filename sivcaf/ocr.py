"""Reconnaissance optique de caractères.

Moteur utilisé, par ordre de préférence :
  1. Tesseract (via pytesseract) s'il est installé — rapide ;
  2. EasyOCR (réseau de neurones, installé via pip) — modèles téléchargés au 1er usage.
"""
from __future__ import annotations

import os
import shutil
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .pretraitement import redimensionner_max

CHEMINS_TESSERACT = [r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                     r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"]


def _tesseract():
    exe = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
    exe = exe or next((c for c in CHEMINS_TESSERACT if Path(c).exists()), None)
    if not exe:
        return None
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = exe
    try:
        langues = pytesseract.get_languages()
    except Exception:
        return None
    return pytesseract, ("fra" if "fra" in langues else "eng")


@lru_cache(maxsize=1)
def moteur():
    """Renvoie (nom, fonction image_rgb -> texte)."""
    choix = os.environ.get("SIVCAF_OCR", "").lower()
    t = _tesseract() if choix in ("", "tesseract") else None
    if t:
        pytesseract, langue = t
        return f"tesseract ({langue})", lambda img: pytesseract.image_to_string(img, lang=langue, config="--psm 4")

    import easyocr
    lecteur = easyocr.Reader(["fr", "en"], gpu=False, verbose=False)

    def lire(img):
        res = lecteur.readtext(img, paragraph=False)
        return _reconstruire_lignes(res)

    return "easyocr (fr)", lire


def _reconstruire_lignes(resultats) -> str:
    """Regroupe les boîtes EasyOCR en lignes de texte (ordre de lecture)."""
    # Sur une photo légèrement inclinée, les montants alignés à droite « glissent »
    # d'une ligne : on estime l'inclinaison médiane du texte et on la compense.
    angles = [np.arctan2(p[1][1] - p[0][1], p[1][0] - p[0][0]) for p, t, _ in resultats if len(t) >= 4]
    a = float(np.median(angles)) if angles else 0.0
    rot = np.array([[np.cos(-a), -np.sin(-a)], [np.sin(-a), np.cos(-a)]])
    boites = []
    for pts, txt, conf in resultats:
        pts = np.array(pts, dtype=float) @ rot.T
        boites.append((pts[:, 1].mean(), pts[:, 0].min(), pts[:, 1].max() - pts[:, 1].min(), txt))
    boites.sort()
    lignes, courante, y_ref, h_ref = [], [], None, None
    for yc, x, h, txt in boites:
        if y_ref is not None and abs(yc - y_ref) > 0.6 * max(h_ref, h):
            lignes.append(courante)
            courante = []
        if not courante:
            y_ref, h_ref = yc, h
        courante.append((x, txt))
    if courante:
        lignes.append(courante)
    # Tabulation entre blocs : distingue les colonnes des espaces internes (« 193 532,82 »)
    return "\n".join("\t".join(t for _, t in sorted(l)) for l in lignes)


def preparer(img: np.ndarray) -> np.ndarray:
    """Mise à l'échelle et amélioration locale du contraste pour l'OCR."""
    img = redimensionner_max(img, 2000)
    h, w = img.shape[:2]
    if max(h, w) < 1400:
        # Petites captures d'écran : les polices fines ne se lisent bien qu'agrandies franchement
        s = (2200 if max(h, w) < 1100 else 1400) / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)
    gris = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gris = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gris)
    return gris


def lire_texte(img: np.ndarray) -> tuple[str, str]:
    nom, lire = moteur()
    return lire(preparer(img)), nom
