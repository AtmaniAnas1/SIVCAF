"""Orientation de chaque document vers le traitement approprié."""
from __future__ import annotations

from .config import (FILES_TRAITEMENT, LIBELLES_QUALITES, LIBELLES_TYPES, SEUIL_CONFIANCE_TYPE,
                     SEUIL_REJET_QUALITE)

FILE_RESCAN = "Re-numérisation demandée"
FILE_MANUELLE = "Vérification manuelle"
FILE_DOUBLONS = "Contrôle des doublons"
FILE_TRI = "Tri manuel (hors factures)"


def evaluer_qualite(qualite_cnn: str, proba_conforme: float, alertes: list[str]) -> tuple[bool, list[str]]:
    """Fusionne l'avis du CNN et les mesures OpenCV. Renvoie (rejet, motifs)."""
    p_defaut = 1 - proba_conforme
    motifs = []
    if qualite_cnn != "conforme":
        motifs.append(f"CNN : {LIBELLES_QUALITES[qualite_cnn].lower()} ({p_defaut:.0%})")
    rejet = (p_defaut >= SEUIL_REJET_QUALITE
             or len(alertes) >= 2
             or (p_defaut >= 0.4 and len(alertes) >= 1))
    if rejet:
        motifs += alertes
    return rejet, motifs


def orienter(type_doc: str, confiance_type: float, rejet_qualite: bool, motifs_qualite: list[str],
             anomalies: list[str], doublon: dict | None) -> tuple[str, str, list[str]]:
    """Renvoie (décision, file de traitement, motifs)."""
    if rejet_qualite:
        return "REJETEE", FILE_RESCAN, motifs_qualite or ["Qualité d'image insuffisante"]
    if doublon:
        return "DOUBLON", FILE_DOUBLONS, [f"Doublon de la facture #{doublon['id']} ({doublon['fichier_original']})"]
    motifs = []
    if confiance_type < SEUIL_CONFIANCE_TYPE:
        motifs.append(f"Classification incertaine : {LIBELLES_TYPES[type_doc]} ({confiance_type:.0%})")
    elif type_doc == "autre":
        return "NON_FACTURE", FILE_TRI, ["Le document n'a pas été reconnu comme une facture"]
    motifs += anomalies
    if motifs:
        return "A_VERIFIER", FILE_MANUELLE, motifs
    return "VALIDEE", FILES_TRAITEMENT[type_doc], [f"Classée : {LIBELLES_TYPES[type_doc]}"]
