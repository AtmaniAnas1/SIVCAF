"""Classement du type de document à partir du texte OCR, et fusion avec le CNN.

Le CNN juge la mise en page ; or chaque pays et chaque fournisseur a la sienne.
Le vocabulaire, lui, est stable (« m3 », « kWh », « forfait », « désignation »…) :
on calcule donc un score textuel par type que l'on combine avec l'avis visuel.
"""
from __future__ import annotations

import math
import re

INDICES = {
    "eau": [
        (r"\beaux?\b", 2), (r"assainissement", 3), (r"\bm\s?3\b|m³|\bm'\b", 2), (r"eaux?\s+us[ée]es", 3),
        (r"\b(?:REGIDESO|SEAAL|SEOR|SEACO|SONEDE|SODECI|CAMWATER|SEN'?EAU|Veolia\s+Eau|Eau\s+de\s+Paris|"
         r"RADEEMA|RADEEF|Amendis|Lydec|Redal)\b", 4), (r"(?-i:\bADE\b|\bSDE\b|\bCDE\b|\bONEP\b)", 3),
        (r"\bindex\b", 1), (r"\bcompteur\b", 1), (r"tranche|\bde\s+\d+\s+[àa]\s+\d+\s*m", 1)],
    "electricite_gaz": [
        (r"[ée]lectricit[ée]", 3), (r"\bkWh\b|\bkVA\b|\bkW\b", 3), (r"\bgaz\b", 2), (r"[ée]nergie", 1),
        (r"puissance\s+souscrite|heures?\s+(?:pleines|creuses)|\bHP\b.*\bHC\b", 2),
        (r"\b(?:Sonelgaz|SADEG|STEG|SENELEC|ENEO|EDF|Engie|TotalEnergies|Hydro[- ]Qu[ée]bec)\b", 4),
        (r"(?-i:\bSNEL\b|\bCIE\b|\bONEE\b|\bONE\b|\bSDC\b)", 3), (r"\bPDL\b|\bPCE\b|point\s+de\s+livraison", 2)],
    "telecom": [
        (r"\bforfait\b", 3), (r"\bmobile\b", 2), (r"\binternet\b|\bfibre\b|\bADSL\b|\b4G\b|\b5G\b|\bbox\b", 2),
        (r"\bappels?\b|\bSMS\b|\bdata\b|\b\d+\s?[GM]o\b", 2), (r"hors\s+forfait|communications", 2),
        (r"\b(?:Orange|SFR|Bouygues|Free|Maroc\s*T[ée]l[ée]com|inwi|Alg[ée]rie\s*T[ée]l[ée]com|Mobilis|Djezzy|"
         r"Ooredoo|Tunisie\s*T[ée]l[ée]com|Vodacom|Airtel|Africell|MTN|Moov|Sonatel|Camtel|Expresso|Bell|"
         r"Vid[ée]otron|Rogers|Telus)\b", 4), (r"t[ée]l[ée]com", 2), (r"(?-i:\bIAM\b)", 3),
        # Marques reconnaissables même collées par l'OCR (« inwimo » pour inwi.ma)
        (r"inwi|wana\s*corporate|djezzy|ooredoo|mobilis|vodacom|africell", 4),
        (r"\babonnements?\b", 1), (r"\bcommunications?\b", 1), (r"compte\s*(?:de\s*)?facturation", 1)],
    "ticket_caisse": [
        (r"\bticket\b", 3), (r"\bcaisse\b", 2), (r"merci\s+(?:de\s+votre|pour\s+votre)\s+(?:visite|achat)", 4),
        (r"\b(?:CB|esp[èe]ces|rendu|monnaie|sans\s+contact|carte\s+bancaire)\b", 2), (r"\bcaissier|\bTPE\b", 2),
        (r"\b\d{2}:\d{2}(?::\d{2})?\b", 1)],
    "fournisseur": [
        (r"d[ée]signation", 3), (r"\bqt[ée]\b|quantit[ée]", 3), (r"prix\s+unitaire|\bP\.?\s?U\.?\b", 3),
        (r"\br[ée]f(?:[ée]rence)?\b", 1), (r"bon\s+de\s+(?:commande|livraison)", 2), (r"conditions?\s+de\s+paiement", 2),
        (r"[ée]ch[ée]ance", 1), (r"factur[ée]\s+[àa]|client\s*:", 1), (r"remise|escompte", 1),
        (r"(?-i:\bI\.?C\.?E\b)|\bSIRET\b|\bNIF\b|\bRCCM\b|\bNINEA\b|matricule\s+fiscal", 1)],
}
# Vocabulaire commun à toute facture / reçu : son absence oriente vers « autre »
FACTURE = re.compile(r"factur|invoice|re[çc]u|ticket|total|montant|net\s+[àa]\s+payer|\bTVA\b|\bTTC\b|\bHT\b", re.I)


def scores_texte(texte: str, ratio_hauteur: float | None = None) -> dict[str, float]:
    s = {t: sum(p for motif, p in ind if re.search(motif, texte, re.I)) for t, ind in INDICES.items()}
    if ratio_hauteur and ratio_hauteur > 2.2:  # format étroit et long : ticket de caisse
        s["ticket_caisse"] += 3
    s["autre"] = 0.0 if len(FACTURE.findall(texte)) >= 2 else 6.0
    return s


def probas_texte(scores: dict[str, float], temperature: float = 1.5) -> dict[str, float] | None:
    """Convertit les scores en probabilités ; None si le texte ne donne aucun indice."""
    if max(scores.values()) < 3:
        return None
    m = max(scores.values())
    e = {t: math.exp((v - m) / temperature) for t, v in scores.items()}
    z = sum(e.values())
    return {t: v / z for t, v in e.items()}


def fusionner(p_cnn: dict[str, float], p_txt: dict[str, float] | None, poids_texte: float = 0.65) -> dict[str, float]:
    """Moyenne pondérée. Si le texte montre une facture, l'avis « autre » du CNN est neutralisé :
    une mise en page inconnue (autre pays, autre fournisseur) n'en fait pas un non-document.
    Le poids du CNN diminue d'autant : s'il ne reconnaît pas la mise en page, son avis sur le type vaut peu."""
    if p_txt is None:
        return dict(p_cnn)
    cnn = dict(p_cnn)
    if p_txt.get("autre", 0) < 0.2 and "autre" in cnn:
        poids_texte = 1 - (1 - poids_texte) * (1 - cnn["autre"])
        reste = 1 - cnn["autre"]
        cnn = {t: (v / reste if t != "autre" else 0.0) if reste > 1e-6 else 1 / (len(cnn) - 1)
               for t, v in cnn.items()}
        cnn["autre"] = 0.0
    p = {t: (1 - poids_texte) * cnn.get(t, 0) + poids_texte * p_txt.get(t, 0) for t in cnn}
    z = sum(p.values())
    return {t: round(v / z, 4) for t, v in p.items()}
