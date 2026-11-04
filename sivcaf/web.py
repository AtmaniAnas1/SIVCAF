"""Application web (Flask) : dépôt, tableau de bord, consultation et correction."""
from __future__ import annotations

import csv
import io
import json
import tempfile
import threading
from pathlib import Path

from flask import (Flask, Response, abort, jsonify, redirect, render_template, request, send_from_directory,
                   url_for)
from werkzeug.utils import secure_filename

from .bd import Base
from .ocr import moteur as moteur_ocr
from .pays import PAYS as LISTE_PAYS
from .config import (CHEMIN_MODELE, EXTENSIONS_ACCEPTEES, FILES_TRAITEMENT, LIBELLES_QUALITES, LIBELLES_TYPES,
                     UPLOADS)

LIBELLES_DECISIONS = {"VALIDEE": "Validée", "A_VERIFIER": "À vérifier", "REJETEE": "Rejetée",
                      "DOUBLON": "Doublon", "NON_FACTURE": "Non-facture"}
CHAMPS_NUMERIQUES = {"montant_ht", "taux_tva", "montant_tva", "montant_ttc", "net_a_payer"}
SYMBOLES = {"EUR": "€", "MAD": "DH", "DZD": "DA", "TND": "DT", "CDF": "FC", "XOF": "FCFA", "XAF": "FCFA",
            "CHF": "CHF", "CAD": "$ CA"}


def creer_app(base: Base | None = None) -> Flask:
    app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))
    app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
    base = base or Base()
    etat = {"analyseur": None}
    verrou = threading.Lock()

    def analyseur():
        with verrou:
            if etat["analyseur"] is None:
                from .analyse import Analyseur
                etat["analyseur"] = Analyseur(base)
            return etat["analyseur"]

    @app.context_processor
    def contexte():
        fichier_metriques = CHEMIN_MODELE.with_suffix(".json")
        metriques = json.loads(fichier_metriques.read_text(encoding="utf-8")) if fichier_metriques.exists() else None
        moteur = moteur_ocr()[0] if moteur_ocr.cache_info().currsize else None  # sans forcer son chargement
        return {"TYPES": LIBELLES_TYPES, "QUALITES": LIBELLES_QUALITES, "DECISIONS": LIBELLES_DECISIONS,
                "PAYS": {p.code: p.nom for p in LISTE_PAYS}, "modele_present": CHEMIN_MODELE.exists(),
                "metriques": metriques, "moteur_ocr": (moteur or "EasyOCR").split(" ")[0],
                "version_css": int((Path(app.static_folder) / "app.css").stat().st_mtime)}  # invalide le cache

    @app.template_filter("montant")
    def montant(v, devise=None):
        if v is None:
            return "—"
        decimales = 0 if devise in ("XOF", "XAF") else 3 if devise == "TND" else 2
        texte = f"{v:,.{decimales}f}".replace(",", " ").replace(".", ",")
        return f"{texte} {SYMBOLES.get(devise, devise or '')}".strip()

    def analyser_fichiers(fichiers):
        resultats = []
        for f in fichiers:
            nom = f.filename or "document"
            ext = Path(nom).suffix.lower()
            if ext not in EXTENSIONS_ACCEPTEES:
                resultats.append({"fichier_original": nom, "erreur": f"Format non pris en charge ({ext})"})
                continue
            with tempfile.TemporaryDirectory() as tmp:
                chemin = Path(tmp) / (secure_filename(nom) or f"document{ext}")
                f.save(chemin)
                try:
                    r = analyseur().analyser(chemin, nom)
                    r.pop("texte_ocr", None)
                    resultats.append(r)
                except Exception as e:  # on remonte l'erreur sans interrompre le lot
                    resultats.append({"fichier_original": nom, "erreur": str(e)})
        return resultats

    @app.get("/")
    def accueil():
        filtres = {k: request.args.get(k) or None for k in ("decision", "type_doc", "recherche")}
        return render_template("index.html", factures=base.lister(**filtres), stats=base.statistiques(),
                               filtres=filtres)

    @app.post("/analyser")
    def analyser():
        resultats = analyser_fichiers(request.files.getlist("fichiers"))
        if request.accept_mimetypes.best == "application/json" or request.args.get("format") == "json":
            return jsonify(resultats)
        return redirect(url_for("accueil"))

    @app.get("/facture/<int:id_>")
    def detail(id_):
        f = base.obtenir(id_) or abort(404)
        return render_template("detail.html", f=f, files=sorted(set(FILES_TRAITEMENT.values())))

    @app.post("/facture/<int:id_>")
    def corriger(id_):
        f = base.obtenir(id_) or abort(404)
        maj = {}
        for k, v in request.form.items():
            v = v.strip() or None
            if k in CHAMPS_NUMERIQUES and v is not None:
                try:
                    v = float(v.replace(" ", "").replace(",", "."))
                except ValueError:
                    continue
            maj[k] = v
        action = request.form.get("action")
        if action == "valider":
            type_doc = maj.get("type_doc") or f["type_doc"]
            maj.update(decision="VALIDEE", statut="corrigee",
                       file_traitement=FILES_TRAITEMENT.get(type_doc, "Tri manuel (hors factures)"))
        elif action == "rejeter":
            maj.update(decision="REJETEE", statut="corrigee", file_traitement="Re-numérisation demandée")
        elif action == "traiter":
            maj["statut"] = "traitee"
        else:
            maj["statut"] = "corrigee"
        maj.pop("action", None)
        base.mettre_a_jour(id_, maj)
        return redirect(url_for("detail", id_=id_))

    @app.post("/facture/<int:id_>/reanalyser")
    def reanalyser(id_):
        base.obtenir(id_) or abort(404)
        analyseur().reanalyser(id_)
        return redirect(url_for("detail", id_=id_))

    @app.post("/facture/<int:id_>/supprimer")
    def supprimer(id_):
        f = base.supprimer(id_) or abort(404)
        if not base.doublon(f["empreinte"], None, None, None):  # fichier plus référencé
            (UPLOADS / f["chemin_stocke"]).unlink(missing_ok=True)
        return redirect(url_for("accueil"))

    @app.get("/fichiers/<path:nom>")
    def fichier(nom):
        return send_from_directory(UPLOADS, nom)

    @app.get("/export.csv")
    def export():
        colonnes = ["id", "date_import", "fichier_original", "type_doc", "confiance_type", "qualite", "decision",
                    "file_traitement", "statut", "numero", "date_facture", "emetteur", "pays", "devise", "siret",
                    "tva_intracom", "iban", "montant_ht", "taux_tva", "montant_tva", "montant_ttc", "net_a_payer"]
        tampon = io.StringIO()
        w = csv.writer(tampon, delimiter=";")
        w.writerow(colonnes)
        for f in base.lister(limite=100000):
            w.writerow([f.get(c) for c in colonnes])
        return Response("﻿" + tampon.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=factures.csv"})

    # --- API JSON -------------------------------------------------------
    @app.post("/api/analyser")
    def api_analyser():
        return jsonify(analyser_fichiers(request.files.getlist("fichiers")))

    @app.get("/api/factures")
    def api_lister():
        return jsonify(base.lister(request.args.get("decision"), request.args.get("type_doc"),
                                   request.args.get("recherche")))

    @app.get("/api/factures/<int:id_>")
    def api_detail(id_):
        return jsonify(base.obtenir(id_) or abort(404))

    @app.get("/api/statistiques")
    def api_stats():
        return jsonify(base.statistiques())

    return app
