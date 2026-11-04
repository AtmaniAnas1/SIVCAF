"""Contrôle qualité par vision classique (OpenCV) et redressement du document.

Ces mesures complètent la tête "qualité" du CNN : elles sont explicables
(on peut dire *pourquoi* un document est rejeté) et servent aussi à
redresser la page avant l'OCR.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from .pretraitement import redimensionner_max, vignette_detail

# Seuils calibrés sur le jeu synthétique (voir `python -m sivcaf calibrer`)
SEUIL_NETTETE = 180.0        # variance du laplacien sur la vignette de détail
SEUIL_SOMBRE = 110.0         # luminosité moyenne du document
SEUIL_CLAIR = 215.0          # au-delà, avec un contraste faible : image délavée
SEUIL_CONTRASTE = 12.0       # écart-type des niveaux de gris du document
SEUIL_INCLINAISON = 12.0     # degrés
SEUIL_SURFACE_MIN = 0.12     # part de l'image occupée par le document


@dataclass
class RapportQualite:
    nettete: float
    luminosite: float
    contraste: float
    surface_document: float
    inclinaison: float
    bords_touches: int
    mode: str                       # "scan" (pleine page) ou "photo"
    alertes: list[str] = field(default_factory=list)

    def dict(self):
        return asdict(self)


def _masque_papier(img_petite: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_petite, cv2.COLOR_RGB2HSV)
    v = cv2.GaussianBlur(hsv[..., 2], (5, 5), 0)
    _, clair = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    peu_sature = (hsv[..., 1] < 70).astype(np.uint8) * 255
    m = cv2.bitwise_and(clair, peu_sature)
    noyau = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    return cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_CLOSE, noyau), cv2.MORPH_OPEN, noyau)


def detecter_document(img: np.ndarray):
    """Renvoie (contour du document à l'échelle d'origine ou None, infos de cadrage)."""
    petite = redimensionner_max(img, 600)
    s = img.shape[0] / petite.shape[0]
    h, w = petite.shape[:2]
    m = _masque_papier(petite)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, {"surface": 0.0, "bords": 0, "angle": 0.0}
    c = max(contours, key=cv2.contourArea)
    surface = cv2.contourArea(c) / (h * w)
    marge = 3
    x, y, bw, bh = cv2.boundingRect(c)
    bords = 0
    for touche, longueur_bord in (
            (np.count_nonzero(m[:marge, :]) / marge, w), (np.count_nonzero(m[-marge:, :]) / marge, w),
            (np.count_nonzero(m[:, :marge]) / marge, h), (np.count_nonzero(m[:, -marge:]) / marge, h)):
        bords += touche > 0.25 * longueur_bord
    (_, _), (rw, rh), angle = cv2.minAreaRect(c)
    angle = angle if rw >= rh else angle - 90  # ramène l'angle dans [-45, 45]
    angle = (angle + 45) % 90 - 45
    approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
    quad = approx.reshape(-1, 2).astype(np.float32) * s if len(approx) == 4 else None
    return quad, {"surface": float(surface), "bords": int(bords), "angle": float(angle)}


def analyser_qualite(img: np.ndarray) -> RapportQualite:
    quad, cadre = detecter_document(img)
    detail = vignette_detail(img)
    nettete = float(cv2.Laplacian(detail, cv2.CV_64F).var())

    petite = cv2.cvtColor(redimensionner_max(img, 600), cv2.COLOR_RGB2GRAY)
    zone = petite
    if cadre["surface"] > 0.05 and cadre["bords"] < 4:
        m = _masque_papier(redimensionner_max(img, 600)) > 0
        if m.sum() > 1000:
            zone = petite[m]
    luminosite, contraste = float(np.mean(zone)), float(np.std(zone))

    # Une page qui occupe presque tout le cadre est un scan (ou une photo bien recadrée)
    mode = "scan" if cadre["bords"] >= 3 and cadre["surface"] > 0.75 else "photo"
    alertes = []
    if nettete < SEUIL_NETTETE:
        alertes.append(f"Netteté insuffisante ({nettete:.0f} < {SEUIL_NETTETE:.0f})")
    if luminosite < SEUIL_SOMBRE:
        alertes.append(f"Image trop sombre (luminosité {luminosite:.0f})")
    elif contraste < SEUIL_CONTRASTE:
        alertes.append(f"Image surexposée (contraste {contraste:.0f})" if luminosite > SEUIL_CLAIR
                       else f"Contraste trop faible ({contraste:.0f})")
    if abs(cadre["angle"]) > SEUIL_INCLINAISON and cadre["surface"] > 0.2:
        alertes.append(f"Document fortement incliné ({cadre['angle']:.0f}°)")
    if mode == "photo":
        if 0 < cadre["bords"] < 4 and cadre["surface"] > 0.2:
            alertes.append(f"Document coupé par le bord de l'image ({cadre['bords']} côté(s))")
        if cadre["surface"] < SEUIL_SURFACE_MIN:
            alertes.append(f"Document trop petit dans l'image ({cadre['surface']:.0%})")
    return RapportQualite(nettete=round(nettete, 1), luminosite=round(luminosite, 1),
                          contraste=round(contraste, 1), surface_document=round(cadre["surface"], 3),
                          inclinaison=round(cadre["angle"], 1), bords_touches=cadre["bords"],
                          mode=mode, alertes=alertes)


def _ordonner(pts: np.ndarray) -> np.ndarray:
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.float32([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]])


def redresser(img: np.ndarray) -> np.ndarray:
    """Corrige la perspective d'une photo de document (sinon renvoie l'image telle quelle)."""
    quad, cadre = detecter_document(img)
    if quad is None or cadre["bords"] > 0 or cadre["surface"] < 0.15:
        return img
    tl, tr, br, bl = _ordonner(quad)
    l = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    if l < 100 or h < 100:
        return img
    M = cv2.getPerspectiveTransform(np.float32([tl, tr, br, bl]), np.float32([[0, 0], [l, 0], [l, h], [0, h]]))
    return cv2.warpPerspective(img, M, (l, h), flags=cv2.INTER_CUBIC)
