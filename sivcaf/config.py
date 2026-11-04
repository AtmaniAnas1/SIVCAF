"""Configuration centrale de SIVCAF."""
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
DONNEES = RACINE / "data"
DATASET = DONNEES / "dataset"
UPLOADS = DONNEES / "uploads"
MODELES = RACINE / "models"
CHEMIN_MODELE = MODELES / "cnn_factures.pt"
CHEMIN_BD = DONNEES / "sivcaf.db"

# Types de documents reconnus par le CNN (tête "type")
TYPES = ["electricite_gaz", "eau", "telecom", "fournisseur", "ticket_caisse", "autre"]
LIBELLES_TYPES = {
    "electricite_gaz": "Facture d'énergie (électricité / gaz)",
    "eau": "Facture d'eau",
    "telecom": "Facture télécom (mobile / internet)",
    "fournisseur": "Facture fournisseur (achats)",
    "ticket_caisse": "Ticket de caisse / reçu",
    "autre": "Document non reconnu",
}

# États de qualité reconnus par le CNN (tête "qualité")
QUALITES = ["conforme", "flou", "mal_cadre", "mauvaise_exposition"]
LIBELLES_QUALITES = {
    "conforme": "Conforme",
    "flou": "Image floue",
    "mal_cadre": "Document mal cadré",
    "mauvaise_exposition": "Exposition incorrecte",
}

TAILLE_ENTREE = 224          # côté de l'image fournie au CNN
SEUIL_CONFIANCE_TYPE = 0.70  # en dessous : vérification manuelle
SEUIL_REJET_QUALITE = 0.60   # proba "non conforme" au-delà de laquelle on rejette

# File de traitement de destination selon le type
FILES_TRAITEMENT = {
    "electricite_gaz": "Comptabilité — charges & abonnements",
    "eau": "Comptabilité — charges & abonnements",
    "telecom": "Comptabilité — charges & abonnements",
    "fournisseur": "Comptabilité fournisseurs",
    "ticket_caisse": "Notes de frais",
}

EXTENSIONS_ACCEPTEES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".pdf"}

for _d in (DONNEES, UPLOADS, MODELES):
    _d.mkdir(parents=True, exist_ok=True)
