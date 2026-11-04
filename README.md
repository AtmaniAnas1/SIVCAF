<p align="center"><img src="sivcaf/static/logo.svg" width="96" alt="Logo SIVCAF"></p>

<h1 align="center">SIVCAF</h1>

<p align="center">
Système Intelligent de Vérification et de Classement Automatique des Factures
</p>

<p align="center">
<img alt="Python" src="https://img.shields.io/badge/Python-3.12-3776AB">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-CPU-EE4C2C">
<img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-4-5C3EE8">
<img alt="Flask" src="https://img.shields.io/badge/Flask-web-000000">
<img alt="SQLite" src="https://img.shields.io/badge/SQLite-3-003B57">
</p>

---

## Présentation

SIVCAF automatise le traitement des factures photographiées ou numérisées. Pour chaque document,
l'application :

1. **classe** le document selon son type à l'aide d'un réseau de neurones convolutif (CNN) ;
2. **détecte** les images inexploitables : floues, mal cadrées ou mal exposées (CNN et mesures OpenCV) ;
3. **extrait** par OCR le numéro, la date, l'émetteur, les identifiants fiscaux, l'IBAN et les montants HT, TVA, TTC ;
4. **vérifie** la cohérence des données : HT + TVA = TTC, taux de TVA, date, champs obligatoires, doublons ;
5. **oriente** le document vers la file de traitement appropriée ;
6. **enregistre** le résultat dans une base SQLite, consultable et corrigeable depuis une interface web.

| Indicateur (jeu de validation synthétique) | Valeur |
|---|---|
| Précision du classement par type | 98,75 % |
| Précision du classement de qualité | 100 % |

Le cahier des charges complet est disponible dans [docs/Cahier_des_charges_SIVCAF.pdf](docs/Cahier_des_charges_SIVCAF.pdf).

## Sommaire

- [Architecture](#architecture)
- [Types de documents et décisions](#types-de-documents-et-décisions)
- [Prise en charge multi-pays](#prise-en-charge-multi-pays)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Interface web et API](#interface-web-et-api)
- [Entraînement du modèle](#entraînement-du-modèle)
- [Base de données](#base-de-données)
- [Structure du projet](#structure-du-projet)
- [Tests](#tests)
- [Auteur](#auteur)

## Architecture

```
 image / PDF
     │
     ▼
 pretraitement.py ──► vue globale 224×224 (RGB)  ─┐
                  └─► vignette de détail 224×224 ─┤   modele.py : CNN à deux branches
                      (zone la plus texturée,     │   ├─ tête « type »    (6 classes)
                       résolution quasi native)   ┘   └─ tête « qualité » (4 classes)
     │
     ├──► qualite.py   : netteté (variance du laplacien), luminosité, contraste,
     │                   détection du contour de la feuille (bords coupés, inclinaison,
     │                   surface) et redressement de perspective
     │
     ├──► routage.evaluer_qualite : fusion avis CNN + alertes OpenCV ─► rejet ?
     │
     ├──► ocr.py        : Tesseract si installé, sinon EasyOCR (fr) sur l'image redressée
     ├──► extraction.py : expressions régulières tolérantes au bruit OCR et contrôles métier
     ├──► bd.py         : détection des doublons (empreinte SHA-256, ou n° + émetteur + montant)
     ├──► routage.orienter : décision et file de traitement
     └──► bd.py         : enregistrement SQLite (data/sivcaf.db)
```

Le CNN comporte deux branches. Une page entière réduite à 224 px ne permet plus de juger la
netteté des caractères : la branche « détail » observe donc un extrait à résolution quasi native.
La tête qualité combine les deux branches ; la tête type n'utilise que la mise en page globale.

| Composant | Technologie |
|---|---|
| Apprentissage profond | PyTorch, torchvision (CPU) |
| Traitement d'image | OpenCV, NumPy, Pillow |
| OCR | Tesseract (pytesseract) ou EasyOCR |
| Lecture PDF | pypdfium2 |
| Application web et API | Flask |
| Persistance | SQLite |

## Types de documents et décisions

| Code | Libellé | File de traitement si validée |
|---|---|---|
| `electricite_gaz` | Facture d'énergie | Comptabilité — charges et abonnements |
| `eau` | Facture d'eau | Comptabilité — charges et abonnements |
| `telecom` | Facture télécom | Comptabilité — charges et abonnements |
| `fournisseur` | Facture fournisseur (achats) | Comptabilité fournisseurs |
| `ticket_caisse` | Ticket de caisse / reçu | Notes de frais |
| `autre` | Document non reconnu | Tri manuel (hors factures) |

| Décision | Condition | File |
|---|---|---|
| `REJETEE` | Qualité insuffisante : CNN ≥ 60 % non conforme, ou 2 alertes OpenCV, ou CNN ≥ 40 % et 1 alerte | Re-numérisation demandée |
| `DOUBLON` | Même fichier, ou même n° / émetteur / montant déjà en base | Contrôle des doublons |
| `NON_FACTURE` | Classé `autre` avec une confiance suffisante | Tri manuel |
| `A_VERIFIER` | Confiance de classement < 70 %, ou anomalie OCR (montant, date ou n° manquant, HT + TVA ≠ TTC) | Vérification manuelle |
| `VALIDEE` | Tous les contrôles sont conformes | Selon le type (tableau ci-dessus) |

Les seuils sont définis dans [sivcaf/config.py](sivcaf/config.py) et [sivcaf/qualite.py](sivcaf/qualite.py).

## Prise en charge multi-pays

Le type final combine deux avis ([sivcaf/classement_texte.py](sivcaf/classement_texte.py)) :

- **l'image (CNN)**, qui juge la mise en page ;
- **le texte OCR**, qui repère le vocabulaire métier (« m3 », « assainissement », « kWh », « forfait »,
  « désignation », « qté ») et les fournisseurs connus (REGIDESO, SEAAL, Sonelgaz, Lydec, ONEE,
  STEG, SENELEC, Maroc Telecom, Djezzy, etc.).

Une mise en page inconnue n'est pas une non-facture : si le CNN répond « document non reconnu »
alors que le texte montre clairement une facture, son avis pèse moins.

[sivcaf/pays.py](sivcaf/pays.py) détecte le pays et la devise, puis applique les taux de TVA et
les identifiants fiscaux correspondants :

| Pays | Devise | TVA (%) | Identifiants |
|---|---|---|---|
| France | EUR | 20 / 10 / 5,5 / 2,1 | SIRET, TVA FR |
| Maroc | MAD (DH) | 20 / 14 / 10 / 7 | ICE, IF, RC, RIB |
| Algérie | DZD (DA) | 19 / 9 | NIF, NIS, RC |
| Tunisie | TND (DT, 3 décimales) | 19 / 13 / 7 | Matricule fiscal |
| RD Congo | CDF (FC) | 16 | RCCM, Id. Nat. |
| Sénégal / Côte d'Ivoire | XOF (FCFA, sans centimes) | 18 / 10 · 18 / 9 | NINEA, CC |
| Cameroun | XAF | 19,25 | NIU |
| Belgique, Suisse, Canada | EUR, CHF, CAD | Taux nationaux | TVA BE, IDE, TPS |

L'extraction distingue le montant de la facture du net à payer, qui peut inclure des arriérés ou
un timbre. Elle reconnaît les formats `1 234,56`, `1.234,56`, `1 234.56`, `1'234.50`,
`25,210 DT` et `15 000 FCFA`.

Pour ajouter un pays ou un fournisseur, compléter `PAYS` dans [sivcaf/pays.py](sivcaf/pays.py) et
`INDICES` dans [sivcaf/classement_texte.py](sivcaf/classement_texte.py), puis ajouter un cas dans
[tests/test_extraction.py](tests/test_extraction.py).

## Installation

Prérequis : Python 3.12.

```bash
git clone https://github.com/AtmaniAnas1/SIVCAF.git
cd SIVCAF
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
```

EasyOCR fonctionne sans autre installation ; ses modèles (environ 100 Mo) sont téléchargés au
premier usage. Si Tesseract est installé avec le pack de langue `fra`, il est utilisé en priorité
car il est plus rapide. La variable d'environnement `SIVCAF_OCR=easyocr` ou `SIVCAF_OCR=tesseract`
force le moteur.

Un modèle entraîné est fourni dans `models/cnn_factures.pt` : l'application est utilisable
immédiatement.

## Utilisation

```bash
# Générer des documents de test (jamais vus à l'entraînement)
python -m sivcaf exemples --n 20

# Analyse en ligne de commande
python -m sivcaf analyser data/exemples
python -m sivcaf analyser facture.jpg --json

# Application web (http://127.0.0.1:5000)
python -m sivcaf serveur
```

| Commande | Rôle |
|---|---|
| `preparer` | Génère le jeu synthétique puis entraîne le CNN (première installation) |
| `generer` | Génère le jeu de données synthétique |
| `entrainer` | Entraîne le CNN |
| `reentrainer` | Réentraîne avec les factures validées dans l'application |
| `exemples` | Crée des documents de test |
| `analyser` | Analyse des fichiers ou des dossiers |
| `reanalyser` | Réanalyse les documents déjà en base |
| `calibrer` | Évalue le taux d'alerte des contrôles OpenCV par classe de qualité |
| `serveur` | Lance l'application web |

## Interface web et API

**Tableau de bord** : compteurs par décision, montant validé, dépôt par glisser-déposer de
plusieurs fichiers, liste filtrable par décision, type ou texte.

**Fiche document** : aperçu, probabilités du CNN, mesures de qualité, motifs d'orientation, champs
OCR modifiables, texte reconnu, actions *Valider*, *Rejeter* et *Marquer traitée*.

**Export CSV** : séparateur `;`, compatible Excel.

| Méthode | Route | Description |
|---|---|---|
| POST | `/api/analyser` | Champ multipart `fichiers` (un ou plusieurs) ; renvoie les résultats |
| GET | `/api/factures?decision=&type_doc=&recherche=` | Liste filtrée |
| GET | `/api/factures/<id>` | Détail complet |
| GET | `/api/statistiques` | Agrégats |

## Entraînement du modèle

Aucun corpus réel n'étant fourni, [sivcaf/synth.py](sivcaf/synth.py) génère des factures
réalistes pour chaque type (mises en page, polices, couleurs, montants cohérents), puis simule la
capture : scan (pleine page, noir et blanc possible) ou photo (fond texturé, perspective, ombre,
éclairage). Les défauts sont étiquetés : flou gaussien ou de bougé, document coupé, fortement
incliné ou trop petit, sous- ou surexposition, ombre portée.

```bash
python -m sivcaf preparer        # environ 2 400 images, quelques minutes sur CPU
```

**Données réelles.** Placer les images dans un dossier avec un fichier `etiquettes.csv`
(`fichier,type,qualite`), puis lancer `python -m sivcaf entrainer --dossier <dossier>`. Mélanger
données réelles et synthétiques est recommandé quand le corpus réel est petit.

**Apprentissage continu.** Chaque facture validée ou corrigée dans l'interface peut servir à
réentraîner le CNN. Elle est déclinée en 12 variantes simulées (photos, scans, flous, mauvais
cadrages) :

```bash
python -m sivcaf reentrainer     # jeu synthétique + factures validées
python -m sivcaf reanalyser      # met à jour les documents en base avec le nouveau modèle
```

## Base de données

Table `factures` (SQLite, `data/sivcaf.db`) : fichier et empreinte, type et confiances, qualité,
probabilités (JSON), mesures OpenCV (JSON), champs extraits, anomalies, texte OCR, moteur OCR,
décision, file de traitement, motifs, statut (`en_attente`, `traitee`, `corrigee`) et durée de
traitement. Les fichiers originaux sont copiés dans `data/uploads/`.

## Structure du projet

```
SIVCAF/
├── sivcaf/
│   ├── __main__.py          Interface en ligne de commande
│   ├── analyse.py           Orchestration de la chaîne de traitement
│   ├── pretraitement.py     Chargement image / PDF, vues pour le CNN
│   ├── modele.py            CNN à deux branches
│   ├── qualite.py           Contrôles qualité OpenCV et redressement
│   ├── ocr.py               Tesseract / EasyOCR
│   ├── extraction.py        Extraction des champs et contrôles métier
│   ├── classement_texte.py  Classement par le texte OCR
│   ├── pays.py              Pays, devises, taux de TVA, identifiants
│   ├── routage.py           Décision et file de traitement
│   ├── bd.py                Persistance SQLite
│   ├── synth.py             Génération du jeu synthétique
│   ├── entrainement.py      Entraînement et réentraînement
│   ├── web.py               Application Flask et API JSON
│   ├── config.py            Configuration et seuils
│   ├── templates/           Gabarits HTML
│   └── static/              Feuille de style et logos
├── models/                  Modèle entraîné et métriques
├── tests/                   Tests d'extraction
├── docs/                    Cahier des charges
└── requirements.txt
```

## Tests

```bash
python tests/test_extraction.py
```

## Auteur

Projet conçu et développé par [AtmaniAnas1](https://github.com/AtmaniAnas1).
