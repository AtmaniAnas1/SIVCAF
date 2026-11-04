"""Détection du pays d'émission, de la devise et des identifiants fiscaux.

Chaque pays est décrit par des indices pondérés (devise, identifiants légaux,
indicatif téléphonique, villes, fournisseurs connus). Le pays retenu est celui
qui cumule le plus d'indices dans le texte OCR.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Pays:
    code: str
    nom: str
    devise: str
    taux_tva: tuple[float, ...]
    indices: tuple[tuple[str, float], ...]          # (expression régulière, poids)
    identifiants: tuple[tuple[str, str], ...] = ()  # (libellé, expression régulière capturante)


PAYS = [
    Pays("FR", "France", "EUR", (20, 10, 5.5, 2.1),
         ((r"€|\bEUR\b|\beuros?\b", 2), (r"\bSIRE[TN]\b", 3), (r"\bFR\s?\d{2}\s?\d{9}\b", 3), (r"\+33\b", 3),
          (r"\bIBAN\s*:?\s*FR\d{2}", 3), (r"\b\d{5}\s+(?:Paris|Lyon|Marseille|Toulouse|Nantes|Bordeaux|Lille|Strasbourg)",
                                          2), (r"\b(?:EDF|Engie|TotalEnergies|Veolia|Suez|Orange|SFR|Bouygues|Free)\b", 1),
          (r"\bTVA\s+20\s*%|\bTVA\s+5[,.]5\s*%", 1)),
         (("SIRET", r"\bSIRE[TN]\s*[:\-]?\s*((?:\d\s?){14}|(?:\d\s?){9})"),
          ("TVA", r"\b(FR\s?[0-9A-Z]{2}\s?\d{3}\s?\d{3}\s?\d{3})\b"))),
    Pays("MA", "Maroc", "MAD", (20, 14, 10, 7),
         ((r"(?-i:\bDHS?\b|\bMAD\b)|\bdirhams?\b", 3), (r"(?-i:\bI\.?C\.?E\b)", 3), (r"(?-i:\bI\.F\b)|\bIdentifiant\s+fiscal", 2),
          (r"\bPatente\b", 2), (r"\bCNSS\b", 1), (r"\+212\b|\b0[5-7]\d{2}[ .-]?\d{2}[ .-]?\d{2}[ .-]?\d{2}\b", 1),
          (r"\b(?:Casablanca|Rabat|Marrakech|F[eè]s|Tanger|Agadir|Mekn[eè]s|Oujda|K[ée]nitra|T[ée]touan|Sal[ée])\b", 2),
          (r"(?-i:\b(?:ONEE|ONEP|ONE|IAM|RAK)\b)|\b(?:Lydec|Redal|Amendis|RADEEMA|RADEEF|RADEES|Maroc\s*T[ée]l[ée]com|inwi|"
           r"Orange\s*Maroc)\b", 3), (r"\bMaroc\b|\bMorocco\b|\bRoyaume\b", 2)),
         (("ICE", r"\bI\.?C\.?E\.?\s*[:\-]?\s*((?:\d\s?){15})"), ("IF", r"\bI\.F\.?\s*[:\-]?\s*(\d{6,10})"),
          ("RC", r"\bR\.?C\.?\s*[:\-]?\s*(\d{3,8})"))),
    Pays("DZ", "Algérie", "DZD", (19, 9),
         ((r"(?-i:\bDA\b)|\bDZD\b|\bdinars?\s+alg", 3), (r"\bNIF\b", 2), (r"\bNIS\b", 2), (r"\bArt(?:icle)?\.?\s*imp", 2),
          (r"\+213\b", 3), (r"\b(?:Alger|Oran|Constantine|Annaba|Blida|S[ée]tif|Tlemcen|B[ée]ja[iï]a|Batna|Bab\s*Ezzouar|"
                            r"Dar\s*El\s*Beida|Kouba|Bouzar[ée]ah)\b", 2),
          (r"(?-i:\b(?:SDC|ADE)\b)|\b(?:Sonelgaz|SADEG|SEAAL|SEOR|SEACO|Alg[ée]rie\s*T[ée]l[ée]com|Mobilis|Djezzy|Ooredoo)\b", 3),
          (r"\bAlg[ée]rie\b|\bAlgeria\b|(?-i:\bSPA\b)", 1), (r"\b\.dz\b|\.dz\b", 2)),
         (("NIF", r"\bNIF\s*[:\-]?\s*(\d{15,20})"), ("NIS", r"\bNIS\s*[:\-]?\s*(\d{12,15})"),
          ("RC", r"\bR\.?C\.?\s*[:\-]?\s*(\d{2}\s?[A-Z]\s?\d{5,8})"))),
    Pays("TN", "Tunisie", "TND", (19, 13, 7),
         ((r"\bTND\b|(?-i:\bDT\b)|\bdinars?\s+tun", 3), (r"\bMatricule\s+fiscal|(?-i:\bM\.F\b)", 3), (r"\+216\b", 3),
          (r"\b(?:Tunis|Sfax|Sousse|Nabeul|Bizerte|Gab[eè]s|Ariana|Monastir)\b", 2),
          (r"\b(?:STEG|SONEDE|Tunisie\s*T[ée]l[ée]com|Ooredoo\s*Tunisie|Orange\s*Tunisie)\b", 3),
          (r"\bTunisie\b|\bTunisia\b", 2)),
         (("MF", r"\bMatricule\s+fiscal\s*[:\-]?\s*([0-9]{7}[A-Z]?(?:/[A-Z]/[A-Z]/[A-Z]/\d{3})?)"),)),
    Pays("CD", "RD Congo", "CDF", (16,),
         ((r"(?-i:\bFC\b)|\bCDF\b|\bfrancs?\s+congolais", 3), (r"\bRCCM\b", 2), (r"\bId\.?\s*Nat(?:ionale)?\b", 3),
          (r"\+243\b", 3), (r"\b(?:Kinshasa|Lubumbashi|Goma|Kisangani|Bukavu|Kalemie|Matadi|Kolwezi|Mbuji[- ]Mayi|"
                            r"Katanga)\b", 2),
          (r"(?-i:\bSNEL\b)|\b(?:REGIDESO|Vodacom|Airtel|Africell|Orange\s*RDC)\b", 3), (r"(?-i:\bRDC\b)|\bCongo\b", 2)),
         (("RCCM", r"\bRCCM\s*[:\-]?\s*([A-Z]{2}/[A-Z]{3}/RCCM/[\w\-]+)"),
          ("Id. Nat.", r"\bId\.?\s*Nat(?:ionale)?\.?\s*[:\-]?\s*([\d\-A-Z]{6,})"))),
    Pays("SN", "Sénégal", "XOF", (18, 10),
         ((r"\bF\s?CFA\b|\bXOF\b", 2), (r"\bNINEA\b", 3), (r"\+221\b", 3),
          (r"\b(?:Dakar|Thi[eè]s|Saint[- ]Louis|Ziguinchor|Kaolack|Rufisque|Pikine)\b", 2),
          (r"(?-i:\bSDE\b)|\b(?:SENELEC|SEN'?EAU|Sonatel|Orange\s*S[ée]n[ée]gal|Free\s*S[ée]n[ée]gal|Expresso)\b", 3),
          (r"\bS[ée]n[ée]gal\b", 2)),
         (("NINEA", r"\bNINEA\s*[:\-]?\s*([\d\s]{7,12}\w*)"),)),
    Pays("CI", "Côte d'Ivoire", "XOF", (18, 9),
         ((r"\bF\s?CFA\b|\bXOF\b", 2), (r"\bCC\s*N[°o]?\s*\d", 2), (r"\+225\b", 3),
          (r"\b(?:Abidjan|Bouak[ée]|Yamoussoukro|Cocody|Plateau|Yopougon|San[- ]P[ée]dro)\b", 2),
          (r"(?-i:\bCIE\b)|\b(?:SODECI|MTN\s*CI|Moov|Orange\s*C[ôo]te)\b", 3), (r"\bC[ôo]te\s*d'?\s*Ivoire\b", 3)),
         (("CC", r"\bCC\s*N?[°o]?\s*[:\-]?\s*(\d{6,8}\s?[A-Z])"),)),
    Pays("CM", "Cameroun", "XAF", (19.25,),
         ((r"\bF\s?CFA\b|\bXAF\b", 2), (r"\bNIU\b", 3), (r"\+237\b", 3),
          (r"\b(?:Douala|Yaound[ée]|Bafoussam|Garoua|Bamenda)\b", 2),
          (r"(?-i:\bCDE\b)|\b(?:ENEO|CAMWATER|Camtel|MTN\s*Cameroon|Orange\s*Cameroun)\b", 3), (r"\bCameroun\b", 2)),
         (("NIU", r"\bNIU\s*[:\-]?\s*([A-Z]\d{12}[A-Z])"),)),
    Pays("BE", "Belgique", "EUR", (21, 12, 6),
         ((r"€|\bEUR\b", 1), (r"\bBE\s?0?\d{3}[.\s]?\d{3}[.\s]?\d{3}\b", 3), (r"\bBCE\b", 2), (r"\+32\b", 3),
          (r"\b(?:Bruxelles|Li[eè]ge|Namur|Charleroi|Mons|Anvers|Gand)\b", 2), (r"\bBelgique\b", 2)),
         (("TVA", r"\b(BE\s?0?\d{3}[.\s]?\d{3}[.\s]?\d{3})\b"),)),
    Pays("CH", "Suisse", "CHF", (8.1, 3.8, 2.6, 7.7, 3.7, 2.5),
         ((r"\bCHF\b|(?-i:\bFr\.\s?\d)", 3), (r"\bCHE[-\s]?\d{3}\.\d{3}\.\d{3}", 3), (r"\+41\b", 3),
          (r"\b(?:Gen[eè]ve|Lausanne|Zurich|Berne|Fribourg|Neuch[aâ]tel|Sion)\b", 2), (r"\bSuisse\b", 2)),
         (("IDE", r"\b(CHE[-\s]?\d{3}\.\d{3}\.\d{3})"),)),
    Pays("CA", "Canada", "CAD", (5, 9.975, 13, 15),
         ((r"\bCAD\b|\$\s?CA|\bCA\s?\$", 3), (r"\bTPS\b|\bTVQ\b|\bTVH\b|\bGST\b|\bQST\b", 3), (r"\+1\b", 1),
          (r"\b(?:Montr[ée]al|Qu[ée]bec|Laval|Gatineau|Sherbrooke|Ottawa|Toronto)\b", 2),
          (r"\b(?:Hydro[- ]Qu[ée]bec|Vid[ée]otron|Bell|Rogers|Telus)\b", 3)),
         (("TPS", r"\bTPS\s*(?:N[°o])?\s*[:\-]?\s*(\d{9}\s?RT\s?\d{4})"),)),
]
PAR_CODE = {p.code: p for p in PAYS}
TOUS_TAUX = sorted({t for p in PAYS for t in p.taux_tva} | {0.0})
SYMBOLES_DEVISES = r"€|EUR|DHS?|MAD|DA|DZD|TND|DT|FC|CDF|F\s?CFA|XOF|XAF|CHF|CAD|\$|USD"


def detecter_pays(texte: str) -> tuple[Pays | None, dict[str, float]]:
    scores = {}
    for p in PAYS:
        s = sum(poids for motif, poids in p.indices if re.search(motif, texte, re.I))
        if s:
            scores[p.code] = s
    if not scores:
        return None, {}
    meilleur = max(scores, key=scores.get)
    return (PAR_CODE[meilleur] if scores[meilleur] >= 3 else None), scores


def identifiant_fiscal(texte: str, pays: Pays | None) -> str | None:
    """Premier identifiant légal trouvé, préfixé par son libellé (« ICE 001… »)."""
    candidats = [pays] if pays else []
    candidats += [p for p in PAYS if p is not pays]
    for p in candidats:
        for libelle, motif in p.identifiants:
            m = re.search(motif, texte, re.I)
            if m:
                return f"{libelle} {re.sub(r'\s', '', m.group(1))}"
    return None
