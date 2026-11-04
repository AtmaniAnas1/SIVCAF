"""Générateur de factures synthétiques pour l'entraînement du CNN.

Aucun jeu de données réel n'étant fourni, on produit des documents imitant
les mises en page typiques de chaque type de facture, puis on simule une
prise de vue (scan ou photo) et des défauts de qualité (flou, cadrage,
exposition). Chaque image est étiquetée par (type, qualité).
"""
from __future__ import annotations

import csv
import math
import random
from datetime import date, timedelta
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import DATASET, QUALITES, TYPES

DOSSIER_POLICES = Path("C:/Windows/Fonts")
POLICES = {
    "sans": ["arial.ttf", "calibri.ttf", "verdana.ttf", "tahoma.ttf", "segoeui.ttf"],
    "sans_gras": ["arialbd.ttf", "calibrib.ttf", "verdanab.ttf", "tahomabd.ttf", "segoeuib.ttf"],
    "serif": ["times.ttf", "georgia.ttf"],
    "serif_gras": ["timesbd.ttf"],
    "mono": ["consola.ttf", "cour.ttf"],
}
_cache_polices: dict = {}


def police(famille: str, taille: int, rng: random.Random | None = None, nom: str | None = None):
    nom = nom or (rng.choice(POLICES[famille]) if rng else POLICES[famille][0])
    cle = (nom, taille)
    if cle not in _cache_polices:
        try:
            _cache_polices[cle] = ImageFont.truetype(str(DOSSIER_POLICES / nom), taille)
        except OSError:
            _cache_polices[cle] = ImageFont.load_default()
    return _cache_polices[cle]


# --------------------------------------------------------------------------
# Contenu aléatoire
# --------------------------------------------------------------------------
SOCIETES = ["Durand Industrie", "Bureau Plus", "Atelier Martin", "Lefèvre & Fils", "TechnoPro SAS",
            "Imprimerie du Centre", "Logistique Ouest", "Garage Moreau", "Papeterie Blanc",
            "Informatique Services", "Bati Rénov", "Agro Distribution", "Electro Câbles SA"]
ENERGIE = ["EnerFrance", "Volt Énergie", "GazNova", "Lumio Énergie", "Planète Watt"]
EAU = ["Eaux de la Vallée", "Régie des Eaux", "AquaServices", "Syndicat Intercommunal des Eaux"]
TELECOM = ["Telio", "Mobilink", "NetBox", "Connecto", "OndeMobile"]
MAGASINS = ["SUPERMARCHE CASINOVA", "CARREFOUR CITY", "BOULANGERIE DU PORT", "PHARMACIE CENTRALE",
            "STATION TOTALGAZ", "RESTAURANT LE MIDI", "BRICO DEPOT", "LIBRAIRIE PAGE"]
RUES = ["rue de la République", "avenue Jean Jaurès", "boulevard Victor Hugo", "rue des Lilas",
        "place de la Gare", "chemin des Vignes", "rue Pasteur", "allée des Tilleuls"]
VILLES = ["75011 Paris", "69003 Lyon", "13008 Marseille", "31000 Toulouse", "44000 Nantes",
          "33000 Bordeaux", "59000 Lille", "67000 Strasbourg"]
NOMS = ["M. Jean Dupont", "Mme Claire Bernard", "M. Ahmed Benali", "Mme Sophie Laurent",
        "SARL Horizon", "M. Lucas Petit", "Mme Nadia Haddad"]
ARTICLES = ["Ramette papier A4", "Cartouche encre noire", "Câble HDMI 2m", "Clavier USB",
            "Prestation maintenance", "Vis inox 6x40 (boîte)", "Écran 24 pouces", "Licence logiciel",
            "Transport et livraison", "Classeur à levier", "Main d'œuvre (heure)", "Disque SSD 1To",
            "Peinture blanche 10L", "Chaise de bureau", "Câble électrique 2.5mm²"]
PRODUITS_TICKET = ["PAIN COMPLET", "LAIT 1L", "CAFE MOULU", "POMMES KG", "EAU MIN X6", "YAOURT X4",
                   "PATES 500G", "CROISSANT", "SANDWICH", "MENU DU JOUR", "CAFE", "GAZOLE",
                   "DOLIPRANE", "CAHIER", "STYLO BIC", "JUS ORANGE"]
MOIS = ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."]
PHRASES = ["Nous vous remercions pour votre confiance et restons à votre disposition.",
           "Suite à notre entretien téléphonique, nous vous confirmons les éléments suivants.",
           "Le présent document ne constitue pas une facture et n'appelle aucun paiement.",
           "Veuillez trouver ci-joint les informations relatives à votre dossier en cours.",
           "Conformément aux dispositions contractuelles, le délai de préavis est de trois mois.",
           "Nous vous prions d'agréer, Madame, Monsieur, l'expression de nos salutations distinguées.",
           "La réunion est programmée dans nos locaux, merci de confirmer votre présence.",
           "Ce courrier fait suite à votre demande d'information reçue le mois dernier.",
           "Les modalités de mise en œuvre seront précisées lors d'une prochaine rencontre."]


def fmt_montant(v: float) -> str:
    entier, dec = f"{v:,.2f}".split(".")
    return entier.replace(",", " ") + "," + dec


def date_alea(rng: random.Random) -> date:
    return date(2023, 1, 1) + timedelta(days=rng.randint(0, 1000))


def num_facture(rng: random.Random) -> str:
    motif = rng.choice(["FA-{a}-{n:05d}", "F{a}{n:06d}", "{n:08d}", "INV-{n:06d}", "FAC/{a}/{n:04d}"])
    return motif.format(a=rng.randint(2023, 2025), n=rng.randint(1, 999999) % (10 ** 6))


def siret(rng: random.Random) -> str:
    s = "".join(str(rng.randint(0, 9)) for _ in range(14))
    return f"{s[:3]} {s[3:6]} {s[6:9]} {s[9:]}"


def couleur_alea(rng: random.Random, sombre=False):
    base = [rng.randint(0, 120), rng.randint(40, 160), rng.randint(90, 220)]
    rng.shuffle(base)
    if sombre:
        base = [c // 2 for c in base]
    return tuple(base)


def adresse(rng):
    return [f"{rng.randint(1, 180)} {rng.choice(RUES)}", rng.choice(VILLES)]


def montants(rng, taux=20.0, ht=None):
    ht = round(ht if ht is not None else rng.uniform(15, 4800), 2)
    tva = round(ht * taux / 100, 2)
    return ht, tva, round(ht + tva, 2)


class Page:
    """Petite surcouche de dessin pour écrire ligne par ligne."""

    def __init__(self, l=850, h=1200, fond=(255, 255, 255)):
        self.img = Image.new("RGB", (l, h), fond)
        self.d = ImageDraw.Draw(self.img)
        self.l, self.h = l, h

    def texte(self, xy, txt, f, c=(20, 20, 20), ancre="la"):
        self.d.text(xy, txt, font=f, fill=c, anchor=ancre)

    def rect(self, box, fill=None, outline=None, w=1):
        self.d.rectangle(box, fill=fill, outline=outline, width=w)

    def ligne(self, box, c=(120, 120, 120), w=1):
        self.d.line(box, fill=c, width=w)


def bloc_totaux(p: Page, rng, x, y, ht, tva, ttc, taux, f, fb, c_titre=(30, 30, 30)):
    lignes = [("Total HT", ht), (f"TVA {str(taux).replace('.', ',')} %", tva), ("Total TTC", ttc)]
    for i, (lib, v) in enumerate(lignes):
        ff = fb if i == 2 else f
        if i == 2:
            p.rect((x - 10, y - 4, x + 300, y + 28), fill=(235, 235, 235))
        p.texte((x, y), lib, ff, c_titre)
        p.texte((x + 290, y), f"{fmt_montant(v)} €", ff, ancre="ra")
        y += 34
    return y


# --------------------------------------------------------------------------
# Gabarits par type de document
# --------------------------------------------------------------------------
def doc_energie(rng: random.Random) -> Image.Image:
    p = Page()
    coul = couleur_alea(rng)
    f, fb = police("sans", 17, rng), police("sans_gras", 18, rng)
    gaz = rng.random() < 0.4
    p.rect((0, 0, p.l, 110), fill=coul)
    p.texte((40, 30), rng.choice(ENERGIE), police("sans_gras", 38, rng), (255, 255, 255))
    titre = "Facture de gaz naturel" if gaz else rng.choice(["Facture d'électricité", "Votre facture d'énergie"])
    p.texte((40, 135), titre, police("sans_gras", 30, rng), coul)
    d = date_alea(rng)
    p.texte((40, 185), f"Facture N° {num_facture(rng)}", f)
    p.texte((40, 210), f"Date : {d:%d/%m/%Y}", f)
    p.texte((40, 235), f"{'PCE' if gaz else 'PDL'} N° {rng.randint(10**13, 10**14 - 1)}", f)
    x = 500
    for i, l in enumerate([rng.choice(NOMS)] + adresse(rng)):
        p.texte((x, 185 + i * 25), l, fb if i == 0 else f)
    # Histogramme de consommation
    p.texte((40, 300), "Votre consommation sur 12 mois (kWh)", fb, coul)
    p.rect((40, 330, 810, 560), outline=(200, 200, 200))
    base = rng.uniform(150, 600)
    for m in range(12):
        v = base * (1 + 0.6 * math.cos((m + rng.random()) / 12 * 2 * math.pi)) + rng.uniform(-40, 40)
        hb = int(max(10, min(190, v / 6)))
        x0 = 60 + m * 62
        p.rect((x0, 545 - hb, x0 + 40, 545), fill=coul if m < 11 else (240, 160, 40))
        p.texte((x0 + 20, 548), MOIS[m], police("sans", 12), ancre="ma")
    # Détail
    y = 600
    p.texte((40, y), "Détail de votre facture", fb, coul)
    y += 40
    kwh = rng.randint(800, 9000)
    ht = 0.0
    lignes = [("Abonnement", rng.uniform(8, 25) * 2), (f"Consommation {kwh} kWh", kwh * rng.uniform(0.12, 0.25)),
              ("Acheminement", rng.uniform(20, 120)), ("Taxes (CTA, accise)", rng.uniform(10, 90))]
    if not gaz and rng.random() < 0.5:
        lignes.insert(2, ("Heures creuses", rng.uniform(30, 200)))
    for lib, v in lignes:
        p.texte((60, y), lib, f)
        p.texte((790, y), f"{fmt_montant(v)} €", f, ancre="ra")
        p.ligne((60, y + 28, 790, y + 28), (220, 220, 220))
        ht += v
        y += 38
    ht, tva, ttc = montants(rng, 20.0, ht)
    y = bloc_totaux(p, rng, 490, y + 20, ht, tva, ttc, 20, f, fb)
    p.rect((40, y + 30, 810, y + 110), fill=tuple(min(255, c + 150) for c in coul))
    p.texte((60, y + 50), f"Montant à payer : {fmt_montant(ttc)} €", police("sans_gras", 24, rng))
    p.texte((60, y + 82), f"Prélevé le {(d + timedelta(days=15)):%d/%m/%Y}", f)
    p.texte((40, 1150), f"{rng.choice(ENERGIE)} SA - SIRET {siret(rng)} - Service client 09 69 32 15 15",
            police("sans", 12))
    return p.img


def doc_eau(rng: random.Random) -> Image.Image:
    p = Page()
    coul = (rng.randint(0, 40), rng.randint(90, 150), rng.randint(170, 230))
    f, fb = police("sans", 17, rng), police("sans_gras", 18, rng)
    nom = rng.choice(EAU)
    p.d.ellipse((40, 30, 120, 110), fill=coul)
    p.d.polygon([(80, 45), (100, 85), (60, 85)], fill=(255, 255, 255))
    p.texte((140, 45), nom, police("sans_gras", 30, rng), coul)
    p.texte((140, 85), "Service de l'eau et de l'assainissement", f)
    p.texte((40, 150), "FACTURE D'EAU", police("sans_gras", 32, rng), coul)
    d = date_alea(rng)
    p.texte((40, 200), f"Facture N° {num_facture(rng)}", f)
    p.texte((40, 225), f"Date : {d:%d/%m/%Y}", f)
    p.texte((40, 250), f"Contrat N° {rng.randint(100000, 999999)}", f)
    for i, l in enumerate([rng.choice(NOMS)] + adresse(rng)):
        p.texte((480, 200 + i * 25), l, fb if i == 0 else f)
    # Relevé compteur
    y = 310
    p.rect((40, y, 810, y + 40), fill=coul)
    p.texte((55, y + 9), "Relevé de votre compteur", fb, (255, 255, 255))
    y += 50
    anc = rng.randint(100, 5000)
    conso = rng.randint(20, 220)
    for col, txt in zip((55, 260, 450, 640), ("Compteur", "Ancien index", "Nouvel index", "Consommation")):
        p.texte((col, y), txt, fb)
    y += 30
    for col, txt in zip((55, 260, 450, 640), (f"N° {rng.randint(10**6, 10**7)}", f"{anc} m³", f"{anc + conso} m³",
                                              f"{conso} m³")):
        p.texte((col, y), txt, f)
    y += 60
    rubriques = [("Distribution de l'eau potable", conso * rng.uniform(1.0, 2.2)),
                 ("Collecte et traitement des eaux usées", conso * rng.uniform(1.2, 2.5)),
                 ("Organismes publics (Agence de l'eau)", conso * rng.uniform(0.3, 0.6)),
                 ("Abonnement", rng.uniform(15, 60))]
    ht = 0
    for lib, v in rubriques:
        p.rect((40, y, 810, y + 34), fill=(235, 243, 252))
        p.texte((55, y + 7), lib, f)
        p.texte((795, y + 7), f"{fmt_montant(v)} €", f, ancre="ra")
        ht += v
        y += 42
    ht, tva, ttc = montants(rng, 5.5, ht)
    y = bloc_totaux(p, rng, 490, y + 25, ht, tva, ttc, 5.5, f, fb)
    # Gouttes décoratives / prix au m3
    p.texte((40, y + 40), f"Prix moyen du m³ : {fmt_montant(ttc / conso)} € TTC", fb, coul)
    for k in range(rng.randint(0, 3)):
        cx = rng.randint(100, 750)
        p.d.ellipse((cx, y + 90, cx + 30, y + 120), outline=coul, width=3)
    p.texte((40, 1150), f"{nom} - SIRET {siret(rng)} - Urgences 24h/24 : 09 77 40 11 22", police("sans", 12))
    return p.img


def doc_telecom(rng: random.Random) -> Image.Image:
    p = Page()
    coul = couleur_alea(rng)
    f, fb = police("sans", 17, rng), police("sans_gras", 18, rng)
    marque = rng.choice(TELECOM)
    p.d.rounded_rectangle((40, 30, 230, 100), radius=30, fill=coul)
    p.texte((135, 65), marque, police("sans_gras", 30, rng), (255, 255, 255), ancre="mm")
    titre = rng.choice(["Votre facture mobile", "Facture Internet Box", "Votre facture Fibre + Mobile"])
    p.texte((270, 50), titre, police("sans_gras", 28, rng), coul)
    d = date_alea(rng)
    p.texte((40, 140), f"Facture N° {num_facture(rng)}", f)
    p.texte((40, 165), f"Date : {d:%d/%m/%Y}", f)
    p.texte((40, 190), f"Ligne : 0{rng.randint(6, 7)} {rng.randint(10, 99)} {rng.randint(10, 99)} "
                       f"{rng.randint(10, 99)} {rng.randint(10, 99)}", f)
    p.texte((40, 215), f"Compte client : {rng.randint(10**8, 10**9)}", f)
    # Encart "montant" à droite
    forfait = rng.choice([9.99, 15.99, 19.99, 29.99, 39.99, 49.99])
    hors = round(rng.choice([0, 0, rng.uniform(1, 35)]), 2)
    options = round(rng.choice([0, 3, 5, 10]), 2)
    ttc = round(forfait + hors + options, 2)
    ht = round(ttc / 1.2, 2)
    tva = round(ttc - ht, 2)
    p.d.rounded_rectangle((500, 130, 810, 260), radius=18, fill=coul)
    p.texte((655, 150), "Montant total", fb, (255, 255, 255), ancre="ma")
    p.texte((655, 185), f"{fmt_montant(ttc)} €", police("sans_gras", 40, rng), (255, 255, 255), ancre="ma")
    y = 300
    sections = [("Forfait et options", [(f"Forfait {rng.choice(['20Go', '100Go', '5G 200Go', 'Fibre 1Gb/s'])}", forfait),
                                        ("Options", options)]),
                ("Consommations hors forfait", [("Appels vers l'international", hors), ("SMS surtaxés", 0.0)])]
    for titre_s, lignes in sections:
        p.texte((40, y), titre_s, police("sans_gras", 22, rng), coul)
        y += 40
        for lib, v in lignes:
            p.texte((70, y), lib, f)
            p.texte((790, y), f"{fmt_montant(v)} €", f, ancre="ra")
            y += 32
        y += 20
    # Pictos ronds de consommation
    for k, lib in enumerate(["Appels", "SMS", "Internet"]):
        cx = 150 + k * 250
        p.d.ellipse((cx - 55, y, cx + 55, y + 110), outline=coul, width=8)
        p.texte((cx, y + 40), f"{rng.randint(5, 99)}%", fb, ancre="ma")
        p.texte((cx, y + 125), lib, f, ancre="ma")
    y += 180
    bloc_totaux(p, rng, 490, y, ht, tva, ttc, 20, f, fb)
    p.texte((40, 1150), f"{marque} SAS - SIRET {siret(rng)} - Assistance 3900", police("sans", 12))
    return p.img


def doc_fournisseur(rng: random.Random) -> Image.Image:
    p = Page()
    f, fb = police("sans", 16, rng), police("sans_gras", 17, rng)
    coul = couleur_alea(rng, sombre=True)
    soc = rng.choice(SOCIETES)
    p.texte((40, 40), soc, police("sans_gras", 28, rng), coul)
    for i, l in enumerate(adresse(rng) + [f"SIRET : {siret(rng)}", f"TVA : FR{rng.randint(10, 99)}{rng.randint(10**8, 10**9 - 1)}"]):
        p.texte((40, 85 + i * 23), l, f)
    p.texte((810, 40), rng.choice(["FACTURE", "Facture"]), police("serif_gras", 44), coul, ancre="ra")
    d = date_alea(rng)
    p.texte((810, 100), f"N° {num_facture(rng)}", fb, ancre="ra")
    p.texte((810, 125), f"Date : {d:%d/%m/%Y}", f, ancre="ra")
    p.texte((810, 150), f"Échéance : {(d + timedelta(days=30)):%d/%m/%Y}", f, ancre="ra")
    p.rect((470, 200, 810, 320), outline=(120, 120, 120))
    p.texte((485, 210), "Facturé à :", fb)
    for i, l in enumerate([rng.choice(SOCIETES + NOMS)] + adresse(rng)):
        p.texte((485, 240 + i * 24), l, f)
    # Tableau quadrillé
    cols = [40, 140, 480, 560, 680, 810]
    y = 360
    p.rect((40, y, 810, y + 34), fill=coul)
    for c, t in zip(cols, ["Réf.", "Désignation", "Qté", "P.U. HT", "Total HT"]):
        p.texte((c + 8, y + 8), t, fb, (255, 255, 255))
    y += 34
    n = rng.randint(4, 14)
    ht = 0.0
    for _ in range(n):
        q = rng.randint(1, 20)
        pu = round(rng.uniform(2, 450), 2)
        tot = q * pu
        ht += tot
        vals = [f"{rng.choice('ABCDEFGH')}{rng.randint(100, 999)}", rng.choice(ARTICLES), str(q),
                fmt_montant(pu), fmt_montant(tot)]
        for c, t in zip(cols, vals):
            p.texte((c + 8, y + 7), t, f)
        y += 32
        p.ligne((40, y, 810, y), (170, 170, 170))
    for c in cols:
        p.ligne((c, 360, c, y), (170, 170, 170))
    taux = rng.choice([20, 20, 20, 10])
    ht, tva, ttc = montants(rng, taux, ht)
    y = bloc_totaux(p, rng, 490, y + 30, ht, tva, ttc, taux, f, fb)
    p.texte((40, y + 30), "Conditions de paiement : 30 jours à réception de facture.", f)
    p.texte((40, y + 55), "Pénalités de retard : 3 fois le taux d'intérêt légal. Indemnité forfaitaire : 40 €.",
            police("sans", 13))
    iban = "FR76 " + " ".join(str(rng.randint(1000, 9999)) for _ in range(5)) + f" {rng.randint(100, 999)}"
    p.texte((40, 1150), f"IBAN : {iban}  -  BIC : {rng.choice(['BNPAFRPP', 'CEPAFRPP', 'AGRIFRPP'])}",
            police("sans", 13))
    return p.img


def doc_ticket(rng: random.Random) -> Image.Image:
    n = rng.randint(3, 14)
    l, h = rng.randint(330, 420), 520 + n * 30
    p = Page(l, h, fond=(rng.randint(240, 255),) * 3)
    fm = police("mono", rng.randint(17, 20), rng)
    fmb = police("mono", 24, None, "courbd.ttf")
    mag = rng.choice(MAGASINS)
    y = 25
    p.texte((l // 2, y), mag, fmb, ancre="ma"); y += 35
    for t in adresse(rng) + [f"Tel 0{rng.randint(1, 5)} {rng.randint(10, 99)} {rng.randint(10, 99)} {rng.randint(10, 99)} {rng.randint(10, 99)}"]:
        p.texte((l // 2, y), t.upper(), fm, ancre="ma"); y += 24
    y += 10
    p.texte((15, y), "-" * 40, fm); y += 26
    tot = 0.0
    for _ in range(n):
        v = round(rng.uniform(0.5, 25), 2)
        tot += v
        p.texte((15, y), rng.choice(PRODUITS_TICKET)[:18], fm)
        p.texte((l - 15, y), fmt_montant(v), fm, ancre="ra"); y += 28
    p.texte((15, y), "-" * 40, fm); y += 30
    tot = round(tot, 2)
    p.texte((15, y), "TOTAL TTC", fmb); p.texte((l - 15, y), f"{fmt_montant(tot)} EUR", fmb, ancre="ra"); y += 36
    p.texte((15, y), "dont TVA 5,5%", fm); p.texte((l - 15, y), fmt_montant(round(tot - tot / 1.055, 2)), fm, ancre="ra"); y += 28
    p.texte((15, y), rng.choice(["CB SANS CONTACT", "ESPECES", "CARTE BANCAIRE"]), fm); y += 34
    d = date_alea(rng)
    p.texte((15, y), f"{d:%d/%m/%Y} {rng.randint(8, 20):02d}:{rng.randint(0, 59):02d}  TICKET {rng.randint(1000, 9999)}", fm); y += 40
    for k in range(rng.randint(40, 70)):  # code-barres
        x = 30 + k * 5
        if x > l - 30:
            break
        if rng.random() < 0.6:
            p.rect((x, y, x + rng.choice([1, 2, 3]), y + 50), fill=(0, 0, 0))
    y += 65
    p.texte((l // 2, y), "MERCI DE VOTRE VISITE", fm, ancre="ma")
    return p.img


def doc_autre(rng: random.Random) -> Image.Image:
    p = Page()
    f = police(rng.choice(["sans", "serif"]), 18, rng)
    fb = police("sans_gras", 20, rng)
    genre = rng.choice(["lettre", "contrat", "note"])
    y = 60
    if genre == "lettre":
        p.texte((60, y), rng.choice(SOCIETES), fb)
        for i, l in enumerate(adresse(rng)):
            p.texte((60, y + 30 + i * 24), l, f)
        for i, l in enumerate([rng.choice(NOMS)] + adresse(rng)):
            p.texte((480, 200 + i * 24), l, f)
        p.texte((480, 300), f"Le {date_alea(rng):%d/%m/%Y}", f)
        y = 380
        p.texte((60, y), "Objet : " + rng.choice(["Votre dossier", "Convocation", "Information", "Résiliation"]), fb)
        y += 50
        p.texte((60, y), "Madame, Monsieur,", f)
        y += 45
    elif genre == "contrat":
        p.texte((p.l // 2, y), rng.choice(["CONTRAT DE PRESTATION", "CONDITIONS GÉNÉRALES", "CONVENTION"]),
                police("serif_gras", 30), ancre="ma")
        y += 80
    else:
        p.rect((0, 0, p.l, 70), fill=couleur_alea(rng))
        p.texte((40, 20), "NOTE INTERNE", police("sans_gras", 28, rng), (255, 255, 255))
        y = 120
    art = 1
    while y < 1080:
        if genre == "contrat" and rng.random() < 0.3:
            p.texte((60, y), f"Article {art}", fb); art += 1; y += 36
        phrase = " ".join(rng.choice(PHRASES) for _ in range(rng.randint(2, 4)))
        mots, ligne = phrase.split(), ""
        for m in mots:
            essai = (ligne + " " + m).strip()
            if p.d.textlength(essai, font=f) > 720:
                p.texte((60, y), ligne, f); y += 27; ligne = m
            else:
                ligne = essai
        p.texte((60, y), ligne, f)
        y += 50
    return p.img


GABARITS = {"electricite_gaz": doc_energie, "eau": doc_eau, "telecom": doc_telecom,
            "fournisseur": doc_fournisseur, "ticket_caisse": doc_ticket, "autre": doc_autre}


# --------------------------------------------------------------------------
# Mise en scène (scan / photo) et dégradations
# --------------------------------------------------------------------------
def fond_texture(rng, l, h) -> np.ndarray:
    base = np.array([rng.randint(30, 170) for _ in range(3)], dtype=np.float32)
    bruit = cv2.GaussianBlur(np.random.default_rng(rng.randint(0, 10**9)).normal(0, 25, (h // 8 + 1, l // 8 + 1, 1)),
                             (0, 0), 2)
    bruit = cv2.resize(bruit.astype(np.float32), (l, h))[..., None]
    if rng.random() < 0.5:  # veinage type bois
        yy = np.linspace(0, rng.uniform(10, 40), h)[:, None, None]
        bruit = bruit + 15 * np.sin(yy + bruit / 10)
    return np.clip(base + bruit, 0, 255).astype(np.uint8)


def poser_document(rng, doc: np.ndarray, sortie=(1000, 1000), echelle=(0.6, 0.85), decalage=0.0,
                   angle_max=6.0, perspective=0.04, fond=None):
    """Pose le document sur un fond avec rotation/perspective. Renvoie l'image."""
    L, H = sortie
    h, w = doc.shape[:2]
    s = rng.uniform(*echelle) * min(L / w, H / h)
    cx = L / 2 + decalage * L * rng.choice([-1, 1]) * (rng.random() < 0.5)
    cy = H / 2 + decalage * H * rng.choice([-1, 1]) * (cx == L / 2 or rng.random() < 0.5)
    ang = math.radians(rng.uniform(-angle_max, angle_max))
    coins = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]) * s
    rot = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    dst = coins @ rot.T + [cx, cy]
    dst += np.array([[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(4)]) * perspective * max(w, h) * s
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    if fond is None:
        fond = fond_texture(rng, L, H)
    page = cv2.warpPerspective(doc, M, (L, H))
    masque = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (L, H))
    # Ombre portée légère
    ombre = cv2.GaussianBlur(masque, (0, 0), 12).astype(np.float32)[..., None] / 255
    fond = (fond.astype(np.float32) * (1 - 0.35 * np.roll(ombre, (8, 8), (0, 1)))).astype(np.uint8)
    m = masque.astype(np.float32)[..., None] / 255
    return (page * m + fond * (1 - m)).astype(np.uint8)


def eclairage(rng, img, force=0.15):
    """Gradient d'éclairage doux (photo prise à la lumière ambiante)."""
    h, w = img.shape[:2]
    gx, gy = rng.uniform(-1, 1), rng.uniform(-1, 1)
    X, Y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    g = 1 + force * (gx * X + gy * Y) / 2
    return np.clip(img * g[..., None], 0, 255).astype(np.uint8)


def scene(rng, doc_pil: Image.Image, qualite: str, est_ticket: bool) -> np.ndarray:
    doc = np.array(doc_pil)
    mode_scan = rng.random() < 0.45
    if mode_scan:
        if est_ticket:  # un ticket scanné apparaît sur la vitre blanche du scanner
            fond = np.full((1200, 850, 3), rng.randint(225, 250), np.uint8)
            img = poser_document(rng, doc, (850, 1200), (0.75, 0.95), angle_max=2, perspective=0, fond=fond)
        else:
            img = cv2.resize(doc, (850, 1200))
            if qualite != "mal_cadre":
                M = cv2.getRotationMatrix2D((425, 600), rng.uniform(-1.5, 1.5), 1)
                img = cv2.warpAffine(img, M, (850, 1200), borderValue=(250, 250, 250))
    else:
        L, H = (900, 1200) if rng.random() < 0.7 else (1000, 1000)
        img = poser_document(rng, doc, (L, H), (0.72, 0.92) if not est_ticket else (0.55, 0.9))
        img = eclairage(rng, img, 0.2)

    if qualite == "mal_cadre":
        if mode_scan and not est_ticket:
            # Feuille de travers et décalée sur la vitre du scanner : coins noirs visibles
            M = cv2.getRotationMatrix2D((425, 600), rng.choice([-1, 1]) * rng.uniform(9, 25), rng.uniform(0.9, 1.05))
            M[:, 2] += [rng.uniform(-150, 150), rng.uniform(-200, 200)]
            img = cv2.warpAffine(img, M, (850, 1200), borderValue=(rng.randint(0, 60),) * 3)
        else:
            choix = rng.random()
            if choix < 0.6:   # document coupé par le bord du cadre
                img = poser_document(rng, doc, img.shape[1::-1], (0.85, 1.15), decalage=rng.uniform(0.25, 0.4))
            elif choix < 0.85:  # fortement incliné
                img = poser_document(rng, doc, img.shape[1::-1], (0.7, 0.95), angle_max=0, perspective=0.02)
                M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2),
                                            rng.choice([-1, 1]) * rng.uniform(18, 45), 1)
                img = cv2.warpAffine(img, M, img.shape[1::-1], borderMode=cv2.BORDER_REFLECT)
            else:  # document minuscule, perdu dans la photo
                img = poser_document(rng, doc, img.shape[1::-1], (0.18, 0.3), angle_max=25)
    elif qualite == "flou":
        if rng.random() < 0.6:
            img = cv2.GaussianBlur(img, (0, 0), rng.uniform(2.8, 6.0))
        else:  # flou de bougé
            k = rng.randint(13, 31)
            noyau = np.zeros((k, k), np.float32)
            noyau[k // 2, :] = 1 / k
            M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1)
            noyau = cv2.warpAffine(noyau, M, (k, k))
            img = cv2.filter2D(img, -1, noyau / max(noyau.sum(), 1e-6))
    elif qualite == "mauvaise_exposition":
        f = img.astype(np.float32) / 255
        choix = rng.random()
        if choix < 0.4:  # sous-exposé
            f = (f ** rng.uniform(2.2, 3.5)) * rng.uniform(0.25, 0.5)
        elif choix < 0.75:  # surexposé / délavé
            f = 1 - (1 - f) * rng.uniform(0.12, 0.3)
        else:  # ombre forte sur une moitié
            h, w = f.shape[:2]
            X, Y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
            a = rng.uniform(0, 2 * math.pi)
            proj = math.cos(a) * X + math.sin(a) * Y
            masque = 1 / (1 + np.exp(-12 * (proj - rng.uniform(-0.3, 0.3))))
            f = f * (1 - (1 - rng.uniform(0.08, 0.2)) * masque[..., None])
        img = np.clip(f * 255, 0, 255).astype(np.uint8)

    # Bruit capteur + compression JPEG pour tous
    bruit = np.random.default_rng(rng.randint(0, 10**9)).normal(0, rng.uniform(1, 5), img.shape)
    img = np.clip(img + bruit, 0, 255).astype(np.uint8)
    if mode_scan and not est_ticket and qualite != "mal_cadre" and rng.random() < 0.5:
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)  # scan N&B
    return img


def generer_exemple(rng: random.Random, type_doc: str, qualite: str) -> np.ndarray:
    doc = GABARITS[type_doc](rng)
    return scene(rng, doc, qualite, type_doc == "ticket_caisse")


def generer_dataset(n_par_type: int = 400, dossier: Path = DATASET, graine: int = 42,
                    part_conforme: float = 0.55, journal=print) -> Path:
    """Génère n_par_type images par type ; ~55 % conformes, le reste réparti sur les défauts."""
    dossier.mkdir(parents=True, exist_ok=True)
    rng = random.Random(graine)
    defauts = QUALITES[1:]
    lignes = []
    total = n_par_type * len(TYPES)
    for t in TYPES:
        for i in range(n_par_type):
            q = "conforme" if rng.random() < part_conforme else rng.choice(defauts)
            img = generer_exemple(rng, t, q)
            nom = f"{t}_{i:04d}_{q}.jpg"
            cv2.imwrite(str(dossier / nom), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
            lignes.append((nom, t, q))
            if len(lignes) % 200 == 0:
                journal(f"  {len(lignes)}/{total} images générées")
    with open(dossier / "etiquettes.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["fichier", "type", "qualite"])
        w.writerows(lignes)
    return dossier
