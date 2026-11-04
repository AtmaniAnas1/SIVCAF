"""Extraction des informations utiles du texte OCR et contrôles de cohérence.

Conçu pour des factures de plusieurs pays francophones : formats de montants
variés (« 1 234,56 », « 1.234,56 », « 1 234.56 », « 1'234.56 »), devises
(€, DH, DA, DT, FC, FCFA…), identifiants fiscaux locaux et taux de TVA nationaux.
Le texte OCR sépare les blocs d'une même ligne par des tabulations.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date

from .pays import PAR_CODE, SYMBOLES_DEVISES, TOUS_TAUX, detecter_pays, identifiant_fiscal

# Séparateur de milliers : espace(s), point, apostrophe ; jamais une tabulation (séparateur de colonnes OCR)
_SEP = r"(?:[   ]{1,3}|[.'’])"
RE_MONTANT = re.compile(r"(?<![\d.,])(\d{1,3}(?:" + _SEP + r"\d{3})+[.,]\d{2}|\d+[.,]\d{2})(?!\d)")
RE_MONTANT_3 = re.compile(r"(?<![\d.,])(\d{1,3}(?:[   ]{1,3}\d{3})*[.,]\d{3}|\d+[.,]\d{3})(?!\d)")  # dinar TN
RE_ENTIER = re.compile(r"(?<![\d.,])(\d{1,3}(?:" + _SEP + r"\d{3})+|\d+)(?![\d.,])")
RE_DATE = re.compile(r"\b(\d{1,2})\s?[/.\-]\s?(\d{1,2})\s?[/.,\-]\s?(\d{4}|\d{2})\b")  # « 31/08,2022 » (OCR)
MOIS = {"janv": 1, "jan": 1, "fév": 2, "fev": 2, "févr": 2, "fevr": 2, "mars": 3, "avr": 4, "mai": 5, "juin": 6,
        "juil": 7, "août": 8, "aout": 8, "sept": 9, "sep": 9, "oct": 10, "nov": 11, "déc": 12, "dec": 12}
RE_DATE_TEXTE = re.compile(r"\b(\d{1,2})(?:er)?\s+(janv|f[ée]vr?|mars|avr|mai|juin|juil|ao[uû]t|sept?|oct|nov|d[ée]c)"
                           r"[a-zéû.]*\s+(\d{4})\b", re.I)

# Un numéro contient au moins un chiffre et n'est pas une date
_JETON_NUM = (r"(?!\d{1,2}[/.\-]\d{1,2}[/.,\-]\d{2,4}\b)"
              r"((?:[A-Z]{1,5}[ \-/]{0,2})?\d[\dA-Z\-/_.]{2,}|[A-Z][A-Z\-/_.]*\d[\dA-Z\-/_.]*)")
_N_DEGRE = r"n[ \t]?[°º'’?\"oPes]\.?(?![a-z])"  # « N° » et ses lectures OCR : N' N? N" No NP Ne Ns
_FACTURE = r"\b(?:fac?[th]\w{0,4}|invoice)"       # « Facture » et ses lectures OCR : Fachre, Factwe…
RE_NUMERO_LIBELLE = re.compile(  # « Facture Electricite N° 1428… » : mots intercalés avant le N°
    _FACTURE + r"(?:[ \t]+[^\W\d_]{2,15}){1,3}[ \t]*" + _N_DEGRE + r"[ \t]*[:\-]?[ \t]*" + _JETON_NUM, re.I)
RE_NUMERO = re.compile(_FACTURE + r"[ \t]*(?:" + _N_DEGRE + r"|n(?=[ \t]+\d)|num[ée]ro|#)?[ \t]*[:\-]?[ \t]*"
                       + _JETON_NUM, re.I)
RE_NUMERO_SEUL = re.compile(r"\b" + _N_DEGRE + r"[ \t]*[:\-]?[ \t]*" + _JETON_NUM, re.I)
# Lignes dont le « N° » n'est pas celui de la facture
RE_AUTRES_NUMEROS = re.compile(r"\b(?:PDL|PCE|contrat|compte|client|compteur|ligne|commande|siret|t[ée]l|site|"
                               r"abonn[ée]|police|r[ée]f[ée]rence\s+client)\b", re.I)
RE_TVA_INTRA = re.compile(r"\b(?:FR\s?[0-9A-Z]{2}\s?\d{3}\s?\d{3}\s?\d{3}|BE\s?0?\d{3}[.\s]?\d{3}[.\s]?\d{3}|"
                          r"CHE[-\s]?\d{3}\.\d{3}\.\d{3})\b")
RE_IBAN = re.compile(r"\b(?:FR|BE|CH|MA|TN|DZ|SN|CI|CM|CD|LU|MC)\d{2}(?:\s?[0-9A-Z]{4}){3,7}(?:\s?[0-9A-Z]{1,4})?\b")
RE_RIB = re.compile(r"\bRIB\s*[:\-]?\s*((?:\d[\s\-]?){20,24})", re.I)
RE_TAUX = re.compile(r"(?:\bTVA|\bT\.V\.A|\bVAT|\bTPS|\bTVQ)\s*(?:d[hn]s?|mad|da|dt|fc)?\s*\(?\s*(\d{1,2}(?:[.,]\d{1,3})?)\s*%",
                     re.I)

FOURNISSEURS_CONNUS = re.compile(
    r"\b(REGIDESO|SNEL|SEAAL|Sonelgaz|SADEG|SEOR|Alg[ée]rie\s*T[ée]l[ée]com|Mobilis|Djezzy|Ooredoo|ONEE|Lydec|Redal|"
    r"Amendis|Maroc\s*T[ée]l[ée]com|inwi|STEG|SONEDE|Tunisie\s*T[ée]l[ée]com|SENELEC|SEN'?EAU|Sonatel|SODECI|ENEO|"
    r"CAMWATER|Camtel|Vodacom|Airtel|Africell|EDF|Engie|TotalEnergies|Veolia|Suez|Hydro[- ]Qu[ée]bec|Vid[ée]otron)"
    r"S?\b", re.I)  # « S? » : l'OCR colle souvent une lettre du logo (« SEAALS »)
FORME_LEGALE = re.compile(r"\b(?:S\.?A\.?R\.?L|S\.?A\.?S|S\.?A|SPA|SNC|EURL|SUARL|SARLU|Soci[ée]t[ée]|St[ée]|"
                          r"Ets|[ÉE]tablissements?|Company|Ltd|Inc|GmbH)\b\.?", re.I)

# Devise parfois intercalée dans le libellé : « Total Dh HT », « Montant TVA Dh 20 % »
_DEV = r"(?:\s*(?:d[hn]s?|mad|da|dzd|dt|tnd|fc|cdf|eur|€|f\s?cfa))?"  # « Total DhHT » : pas de \b
LIBELLES_TTC = [r"total" + _DEV + r"\s*t\.?t\.?c", r"montant" + _DEV + r"\s+t\.?t\.?c",r"total\s+tvac", r"tva\s+comprise",r"montant\s+(?:de\s+(?:la|votre)\s+)?facture",
                r"total\s+(?:de\s+la\s+)?facture", r"total\s+g[ée]n[ée]ral", r"montant\s+total"]
LIBELLES_NET = [r"net\s*[àa]\s*payer", r"total\s*[àa]\s*payer", r"montant\s*[àa]\s*payer", r"reste\s*[àa]\s*payer",
                r"[àa]\s+payer\s+avant", r"total\s+d[ûu]"]
LIBELLES_HT = [r"total" + _DEV + r"\s*h\.?t\.?(?:v\.?a\.?)?", r"\bHTVA\b", r"montant" + _DEV + r"\s*h\.?t\.?",
               r"base\s+(?:imposable|h\.?t)", r"total\s+hors\s+taxes?",
               r"sous[- ]total"]
LIBELLES_TVA = [r"dont\s*t\.?v\.?a", r"\bt\.?v\.?a\s*\(?\s*\d{1,2}(?:[.,]\d+)?\s*%\s*\)?", r"total\s*t\.?v\.?a",
                r"montant\s*t\.?v\.?a", r"tax\w*\s*(?:et|el|&)\s*redevances?", r"\bt\.?v\.?a\b"]


@dataclass
class Champs:
    numero: str | None = None
    date_facture: str | None = None      # ISO AAAA-MM-JJ
    emetteur: str | None = None
    pays: str | None = None              # code ISO (FR, MA, DZ, TN, CD, SN…)
    devise: str | None = None
    siret: str | None = None             # identifiant fiscal, préfixé (« SIRET … », « ICE … », « NIF … »)
    tva_intracom: str | None = None
    iban: str | None = None              # IBAN ou RIB
    montant_ht: float | None = None
    taux_tva: float | None = None
    montant_tva: float | None = None
    montant_ttc: float | None = None     # montant de la facture
    net_a_payer: float | None = None     # peut inclure arriérés, timbre…
    anomalies: list[str] = field(default_factory=list)

    def dict(self):
        return asdict(self)


def normaliser(texte: str) -> str:
    """Corrige les défauts OCR fréquents autour des montants et des libellés."""
    t = re.sub(r"(\d)[ \t]*[,.][ \t]*[,.]?[ \t]*(\d{2})(?![\d./\-])", r"\1,\2", texte)  # « 160 ,32 », « 965,.26 »
    t = re.sub(r"(\d)[ \t]+(\d{2})([ \t]*(?:" + SYMBOLES_DEVISES + r")\b)", r"\1,\2\3", t)  # « 168  73 EUR »
    t = re.sub(r"\b[TIl1|][O0][TIl1][A4oO][Ll1]", "Total", t, flags=re.I)        # « TOIAL », « Iotal », « Totol »
    t = re.sub(r"\b[Il1|]VA\b", "TVA", t)
    t = re.sub(r"\bT\.?T\.?C\b", "TTC", t, flags=re.I)
    return t


def vers_float(s: str, decimales: int = 2) -> float:
    """« 45.971,70 », « 53 327.17 », « 1'234,50 », « 15 000 », « 25,210 » (3 décimales) -> float."""
    s = re.sub(r"[\s'’  ]", "", s)
    m = re.search(r"[.,](\d{%d})$" % decimales, s)
    if m:
        return float(re.sub(r"[.,]", "", s[:m.start()]) + "." + m.group(1))
    return float(re.sub(r"[.,]", "", s))


def _montant_apres(texte: str, motifs: list[str], entiers: bool = False, millimes: bool = False,
                   fusion_milliers: bool = True) -> float | None:
    """Montant situé sur la même ligne que le libellé (dernière occurrence du premier libellé trouvé).

    `entiers` : montants sans centimes (franc CFA) ; `millimes` : 3 décimales (dinar tunisien)."""
    for motif in motifs:
        trouves = []
        for ligne in texte.splitlines():
            if not re.search(motif, ligne, re.I):
                continue
            apres = re.split(motif, ligne, maxsplit=1, flags=re.I)[-1]
            # « 4⇥655,93 » : l'OCR a coupé le montant en deux blocs au séparateur de milliers.
            # Jamais sur une ligne de TVA, où le nombre qui précède est le plus souvent le taux (« TVA 20⇥215,40 »).
            if fusion_milliers:
                apres = re.sub(r"(?<![\d.,])(\d{1,3})\t(\d{3}[.,]\d{2})(?!\d)", r"\1 \2", apres)
            apres = re.sub(r"\(\s*\d{1,2}(?:[.,]\d+)?\s*%\s*\)|\d{1,2}(?:[.,]\d+)?\s*%", " ", apres)  # retire les taux
            if millimes and (m3 := RE_MONTANT_3.findall(apres)):
                trouves.append(vers_float(m3[-1], 3))
                continue
            m = RE_MONTANT.findall(apres)
            if not m and (fin := re.search(r"(\d+)[ ](\d{2})\s*$", apres)):  # « 1134 89 » : virgule perdue
                m = [f"{fin.group(1)},{fin.group(2)}"]
            if not m and entiers:
                m = [x for x in RE_ENTIER.findall(apres) if len(re.sub(r"\D", "", x)) >= 2]
            if m:
                trouves.append(vers_float(m[-1]))
        if trouves:
            return trouves[-1]
    return None


def _date(texte: str) -> str | None:
    candidates = []
    for rang, ligne in enumerate(texte.splitlines()):
        dates = []
        for m in RE_DATE.finditer(ligne):
            j, mo, a = (int(x) for x in m.groups())
            dates.append((j, mo, a + 2000 if a < 100 else a))
        for m in RE_DATE_TEXTE.finditer(ligne):
            cle = m.group(2).lower().rstrip(".")
            mo = MOIS.get(cle) or MOIS.get(cle[:4]) or MOIS.get(cle[:3])
            if mo:
                dates.append((int(m.group(1)), mo, int(m.group(3))))
        if not dates:
            continue
        score = 0
        if re.search(r"date\s+(?:de\s+(?:la\s+)?)?facturation|date\s+(?:de\s+(?:la\s+)?)?facture|[ée]mise?\s+le|"
                     r"date\s+d'?\s*[ée](?:mission|dition)|facture\s+du", ligne, re.I):
            score = 3
        elif re.search(r"\bdate\b|\bdu\b|\ble\b", ligne, re.I):
            score = 1
        if re.search(r"[ée]ch[ée]ance|pr[ée]lev|avant\s+le|r[ée]gler|limite|payable|p[ée]riode|\bau\b|relev[ée]|"
                     r"index", ligne, re.I):
            score -= 2
        for j, mo, a in dates:
            try:
                candidates.append((-score, rang, date(a, mo, j)))
            except ValueError:
                continue
    return min(candidates)[2].isoformat() if candidates else None


def _segments(ligne: str) -> list[str]:
    return [s.strip(" -:|") for s in re.split(r"\t|\s{3,}", ligne) if s.strip(" -:|")]


def _emetteur(texte: str) -> str | None:
    lignes = texte.splitlines()
    tete = "\n".join(lignes[:12])
    if m := FOURNISSEURS_CONNUS.search(tete):
        nom = re.sub(r"\s+", " ", m.group(1))
        return nom if nom in (nom.upper(), nom.capitalize(), nom.title()) else nom.upper()  # « REGiDESo »
    for ligne in lignes[:10]:  # segment portant une forme juridique : « Durand SARL », « Société des Eaux… »
        for seg in _segments(ligne):
            if FORME_LEGALE.search(seg) and len(seg) <= 70 and len(re.findall(r"\d", seg)) <= 2:
                return re.sub(r"\s+", " ", seg)
    for ligne in reversed(lignes):  # mention légale de pied de page : « Wana Corporate SA - ICE 0015… »
        m = re.match(r"\s*([A-Za-zÀ-ÿ][\w'’ &.-]{2,60}?)\s*[-–,]?\s*(?:SIRE[TN]|ICE|NIF|RCCM|NINEA|NIU)\b", ligne)
        if m and FORME_LEGALE.search(m.group(1)):
            return re.sub(r"\s+", " ", m.group(1)).strip(" -")
    for ligne in lignes[:6]:
        segs = _segments(ligne)
        if not segs:
            continue
        # Ligne courte sans chiffre (« PHARMACIE⇥CENTRALE ») : un seul nom découpé en blocs par l'OCR
        entiere = re.sub(r"\s+", " ", ligne).strip()
        l = entiere if len(entiere) <= 40 and not re.search(r"\d", entiere) else segs[0]
        l = re.split(r"\b(?:votre|facture|invoice|ticket)\b", l, maxsplit=1, flags=re.I)[0].strip(" -:|")
        propre = len(re.findall(r"[\w '’&.-]", l)) >= 0.9 * len(l)  # écarte le bruit OCR des logos
        if (len(l) >= 3 and propre and re.search(r"[A-Za-zÀ-ÿ]{4}", l) and len(re.findall(r"\d", l)) <= 2
                and not re.search(r"date|n[°º']|total|t[ée]l|siret|tva|iban|[ée]ch[ée]ance|client|ligne|abonn|"
                                  r"\b(?:janv|f[ée]vr|mars|avr|juin|juil|ao[uû]t|sept|oct|nov|d[ée]c)", l, re.I)):
            return re.sub(r"\s+", " ", l)[:80]
    for ligne in reversed(lignes):
        m = re.match(r"\s*([A-Za-zÀ-ÿ][\w'’ &.-]{2,60}?)\s*[-–,]?\s*SIRE[TN]", ligne, re.I)
        if m and re.search(r"[A-Za-zÀ-ÿ]{3}", m.group(1)):
            return re.sub(r"\s+", " ", m.group(1)).strip(" -")
    return None


def _numero(texte: str) -> str | None:
    trouves = [m for motif in (RE_NUMERO, RE_NUMERO_LIBELLE) if (m := motif.search(texte))]
    if trouves:  # la première mention dans le document
        return min(trouves, key=lambda m: m.start()).group(1).strip(".").replace(" ", "")
    for ligne in texte.splitlines():
        for seg in re.split(r"\t", ligne):
            if not RE_AUTRES_NUMEROS.search(seg) and (m := RE_NUMERO_SEUL.search(seg)):
                return m.group(1).strip(".").replace(" ", "")
    return None


def extraire(texte: str) -> Champs:
    c = Champs()
    texte = normaliser(texte)
    pays, _ = detecter_pays(texte)
    if pays:
        c.pays, c.devise = pays.code, pays.devise
    c.numero = _numero(texte)
    c.date_facture = _date(texte)
    c.emetteur = _emetteur(texte)
    c.siret = identifiant_fiscal(texte, pays)
    if m := RE_TVA_INTRA.search(texte):
        c.tva_intracom = re.sub(r"\s", "", m.group(0))
    if m := RE_IBAN.search(texte):
        c.iban = m.group(0)
    elif m := RE_RIB.search(texte):
        c.iban = "RIB " + re.sub(r"[\s\-]", "", m.group(1))
    if m := RE_TAUX.search(texte):
        c.taux_tva = vers_float(m.group(1)) if re.search(r"[.,]\d{2}$", m.group(1)) else \
            float(m.group(1).replace(",", "."))
        if c.taux_tva not in TOUS_TAUX and c.taux_tva / 10 in TOUS_TAUX:  # « 55 % » lu pour « 5,5 % »
            c.taux_tva /= 10
    fmt = {"entiers": c.devise in ("XOF", "XAF"),  # le franc CFA s'écrit sans centimes
           "millimes": c.devise == "TND"}         # le dinar tunisien compte 3 décimales
    c.montant_ttc = _montant_apres(texte, LIBELLES_TTC, **fmt)
    c.net_a_payer = _montant_apres(texte, LIBELLES_NET, **fmt)
    c.montant_ht = _montant_apres(texte, LIBELLES_HT, **fmt)
    c.montant_tva = _montant_apres(texte, LIBELLES_TVA, fusion_milliers=False, **fmt)
    if c.montant_ttc is None:
        c.montant_ttc = c.net_a_payer if c.net_a_payer is not None else _montant_apres(texte, [r"\btotal\b"], **fmt)
    # Déductions simples quand une valeur manque
    if c.montant_ttc is None and c.montant_ht is not None and c.montant_tva is not None:
        c.montant_ttc = round(c.montant_ht + c.montant_tva, 2)
    if c.montant_ht is None and c.montant_ttc is not None and c.montant_tva is not None:
        c.montant_ht = round(c.montant_ttc - c.montant_tva, 2)
    if c.montant_tva is None and c.montant_ttc is not None and c.montant_ht is not None:
        c.montant_tva = round(c.montant_ttc - c.montant_ht, 2)
    return c


def verifier(c: Champs, type_doc: str) -> list[str]:
    """Contrôles métier ; renvoie la liste des anomalies détectées."""
    a = []
    if not c.montant_ttc:
        a.append("Montant de la facture introuvable")
    if not c.date_facture:
        a.append("Date de facture introuvable")
    else:
        d = date.fromisoformat(c.date_facture)
        if d > date.today():
            a.append("Date de facture dans le futur")
        elif (date.today() - d).days > 10 * 365:
            a.append("Facture de plus de 10 ans")
    if type_doc != "ticket_caisse" and not c.numero:
        a.append("Numéro de facture introuvable")
    if c.montant_ht is not None and c.montant_tva is not None and c.montant_ttc is not None:
        tolerance = max(0.05, 0.0005 * c.montant_ttc)
        if abs(c.montant_ht + c.montant_tva - c.montant_ttc) > tolerance:
            a.append(f"Incohérence HT + TVA ≠ TTC ({c.montant_ht:.2f} + {c.montant_tva:.2f} ≠ {c.montant_ttc:.2f})")
        elif c.taux_tva and c.montant_ht and type_doc != "ticket_caisse":
            attendu = c.montant_ht * c.taux_tva / 100
            if abs(attendu - c.montant_tva) > max(0.05, 0.01 * c.montant_tva):
                a.append(f"TVA incohérente avec le taux de {c.taux_tva:g} %")
    if c.taux_tva and c.pays and c.taux_tva not in PAR_CODE[c.pays].taux_tva:
        a.append(f"Taux de TVA de {c.taux_tva:g} % inhabituel pour : {PAR_CODE[c.pays].nom}")
    if type_doc == "fournisseur" and not (c.siret or c.tva_intracom):
        a.append("Identifiant fiscal du fournisseur absent")
    c.anomalies = a
    return a
