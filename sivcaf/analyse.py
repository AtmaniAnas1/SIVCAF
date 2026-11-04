"""Pipeline complet : image -> CNN -> contrôle qualité -> OCR -> contrôles -> orientation -> base."""
from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from .bd import Base
from .classement_texte import fusionner, probas_texte, scores_texte
from .config import CHEMIN_MODELE, QUALITES, TYPES, UPLOADS
from .extraction import extraire, verifier
from .modele import charger, vers_tenseurs
from .ocr import lire_texte
from .pretraitement import charger_image, entrees_cnn
from .qualite import analyser_qualite, redresser
from .routage import evaluer_qualite, orienter


def empreinte_fichier(chemin: Path) -> str:
    return hashlib.sha256(Path(chemin).read_bytes()).hexdigest()


class Analyseur:
    def __init__(self, base: Base | None = None, chemin_modele: Path = CHEMIN_MODELE):
        if not Path(chemin_modele).exists():
            raise FileNotFoundError(
                f"Modèle introuvable ({chemin_modele}). Lancez d'abord : python -m sivcaf preparer")
        self.modele, paquet = charger(chemin_modele)
        self.types, self.qualites = paquet["types"], paquet["qualites"]
        self.metriques = paquet.get("metriques", {})
        self.base = base or Base()

    def classer(self, img: np.ndarray) -> dict:
        g, d = entrees_cnn(img)
        tg, td = vers_tenseurs(g[None], d[None])
        with torch.no_grad():
            lt, lq = self.modele(tg, td)
        pt, pq = torch.softmax(lt, 1)[0].numpy(), torch.softmax(lq, 1)[0].numpy()
        return {"type": {t: round(float(p), 4) for t, p in zip(self.types, pt)},
                "qualite": {q: round(float(p), 4) for q, p in zip(self.qualites, pq)}}

    def reanalyser(self, id_: int) -> dict:
        """Relance l'analyse d'un document déjà en base (nouveau modèle ou nouvelles règles)."""
        f = self.base.obtenir(id_)
        if f is None:
            raise KeyError(id_)
        r = self.analyser(UPLOADS / f["chemin_stocke"], f["fichier_original"], enregistrer=False, avant_id=id_)
        r["statut"] = "en_attente"
        self.base.remplacer_analyse(id_, r)
        return r

    def analyser(self, chemin: str | Path, nom_original: str | None = None, ocr: bool = True,
                 enregistrer: bool = True, avant_id: int | None = None) -> dict:
        t0 = time.time()
        chemin = Path(chemin)
        nom_original = nom_original or chemin.name
        img = charger_image(chemin)
        empreinte = empreinte_fichier(chemin)

        # 1. Analyse visuelle par le CNN
        probas = self.classer(img)
        probas["type_cnn"] = probas["type"]
        qualite = max(probas["qualite"], key=probas["qualite"].get)

        # 2. Contrôle qualité OpenCV + fusion avec l'avis du CNN
        rapport = analyser_qualite(img)
        rejet, motifs_qualite = evaluer_qualite(qualite, probas["qualite"]["conforme"], rapport.alertes)

        # 3. OCR, classement par le texte (fusionné avec le CNN) et extraction
        texte, moteur_ocr, champs = "", None, extraire("")
        anomalies: list[str] = []
        if ocr and not rejet:
            texte, moteur_ocr = lire_texte(redresser(img))
            p_txt = probas_texte(scores_texte(texte, img.shape[0] / img.shape[1]))
            probas["type_texte"] = {t: round(p, 4) for t, p in p_txt.items()} if p_txt else None
            probas["type"] = fusionner(probas["type_cnn"], p_txt)
        type_doc = max(probas["type"], key=probas["type"].get)
        if texte:
            champs = extraire(texte)
            anomalies = verifier(champs, type_doc)

        # 4. Doublons et orientation
        doublon = self.base.doublon(empreinte, champs.numero, champs.emetteur, champs.montant_ttc, avant_id) \
            if enregistrer or avant_id else None
        decision, file_traitement, motifs = orienter(type_doc, probas["type"][type_doc], rejet,
                                                     motifs_qualite, anomalies, doublon)

        resultat = {
            "fichier_original": nom_original, "empreinte": empreinte,
            "type_doc": type_doc, "confiance_type": probas["type"][type_doc],
            "qualite": qualite, "confiance_qualite": probas["qualite"][qualite], "probas": probas,
            "mesures_qualite": rapport.dict(),
            **{k: v for k, v in champs.dict().items() if k != "anomalies"},
            "anomalies": anomalies, "texte_ocr": texte, "moteur_ocr": moteur_ocr,
            "decision": decision, "file_traitement": file_traitement, "motifs": motifs,
            "duree_ms": int((time.time() - t0) * 1000),
        }
        if enregistrer:
            dest = UPLOADS / f"{empreinte[:16]}{chemin.suffix.lower()}"
            if not dest.exists():
                shutil.copyfile(chemin, dest)
            resultat["chemin_stocke"] = dest.name
            resultat["id"] = self.base.inserer(resultat)
        return resultat
