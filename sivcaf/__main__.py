"""Interface en ligne de commande : python -m sivcaf <commande>."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from .config import DATASET, DONNEES, EXTENSIONS_ACCEPTEES, QUALITES, TYPES


def cmd_generer(a):
    from .synth import generer_dataset
    print(f"Génération de {a.n * len(TYPES)} images dans {a.dossier}…")
    generer_dataset(a.n, Path(a.dossier), a.graine)


def cmd_entrainer(a):
    from .entrainement import entrainer
    entrainer(Path(a.dossier), epoques=a.epoques)


def cmd_reentrainer(a):
    """Réentraîne le CNN sur le jeu synthétique + les vraies factures validées/corrigées dans l'application."""
    from .bd import Base
    from .entrainement import REELS, entrainer, preparer_reels
    if not preparer_reels(Base(), REELS, a.variantes):
        print("Aucun document réel validé ou corrigé : validez d'abord des factures dans l'application.")
        return
    entrainer([Path(a.dossier), REELS], epoques=a.epoques)


def cmd_preparer(a):
    cmd_generer(a)
    cmd_entrainer(a)


def cmd_exemples(a):
    """Crée quelques documents de test (jamais vus à l'entraînement) pour essayer l'application."""
    import cv2
    from .synth import generer_exemple
    dossier = Path(a.dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.graine)
    for i in range(a.n):
        t = rng.choice(TYPES)
        q = "conforme" if rng.random() < 0.65 else rng.choice(QUALITES[1:])
        img = generer_exemple(rng, t, q)
        cv2.imwrite(str(dossier / f"exemple_{i:02d}_{t}_{q}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"{a.n} documents de test écrits dans {dossier}")


def _fichiers(chemins):
    for c in map(Path, chemins):
        if c.is_dir():
            yield from sorted(p for p in c.iterdir() if p.suffix.lower() in EXTENSIONS_ACCEPTEES)
        else:
            yield c


def cmd_analyser(a):
    from .analyse import Analyseur
    an = Analyseur()
    for chemin in _fichiers(a.chemins):
        r = an.analyser(chemin, ocr=not a.sans_ocr, enregistrer=not a.sans_enregistrement)
        if a.json:
            r.pop("texte_ocr", None)
            print(json.dumps(r, ensure_ascii=False))
            continue
        print(f"\n{chemin.name}")
        print(f"  type     : {r['type_doc']} ({r['confiance_type']:.0%})   qualité : {r['qualite']} "
              f"({r['confiance_qualite']:.0%})")
        if r["montant_ttc"] is not None or r["numero"]:
            print(f"  extrait  : n° {r['numero']}  date {r['date_facture']}  TTC {r['montant_ttc']}  "
                  f"émetteur {r['emetteur']}")
        print(f"  décision : {r['decision']} -> {r['file_traitement']}   ({r['duree_ms']} ms)")
        for m in r["motifs"]:
            print(f"             - {m}")


def cmd_reanalyser(a):
    """Réanalyse les documents en base (sauf ceux corrigés à la main, sauf --tous)."""
    from .analyse import Analyseur
    an = Analyseur()
    for f in sorted(an.base.lister(limite=100000), key=lambda f: f["id"]):
        if f["statut"] == "corrigee" and not a.tous:
            continue
        r = an.reanalyser(f["id"])
        changement = "" if r["decision"] == f["decision"] else f"   (avant : {f['decision']})"
        print(f"#{f['id']:<4} {f['fichier_original'][:40]:40s} {r['decision']:11s} {r['type_doc']}{changement}")


def cmd_calibrer(a):
    """Mesure la pertinence des contrôles OpenCV sur le jeu étiqueté."""
    import csv
    from .pretraitement import charger_image
    from .qualite import analyser_qualite
    with open(Path(a.dossier) / "etiquettes.csv", encoding="utf-8") as fh:
        lignes = list(csv.DictReader(fh))
    random.Random(0).shuffle(lignes)
    stats = {q: [0, 0] for q in QUALITES}
    for l in lignes[:a.n]:
        r = analyser_qualite(charger_image(Path(a.dossier) / l["fichier"]))
        stats[l["qualite"]][0] += bool(r.alertes)
        stats[l["qualite"]][1] += 1
    for q, (alerte, n) in stats.items():
        print(f"{q:22s} {alerte:4d}/{n:<4d} avec au moins une alerte ({alerte / max(n, 1):.0%})")


def cmd_serveur(a):
    from .web import creer_app
    print(f"SIVCAF disponible sur http://{a.hote}:{a.port}")
    creer_app().run(host=a.hote, port=a.port, debug=False, threaded=True)


def main(argv=None):
    p = argparse.ArgumentParser(prog="sivcaf", description="Classement et vérification automatique de factures")
    s = p.add_subparsers(dest="commande", required=True)

    g = s.add_parser("generer", help="générer le jeu de données synthétique")
    g.add_argument("--n", type=int, default=400, help="images par type de document")
    g.add_argument("--dossier", default=str(DATASET))
    g.add_argument("--graine", type=int, default=42)
    g.set_defaults(f=cmd_generer)

    e = s.add_parser("entrainer", help="entraîner le CNN")
    e.add_argument("--dossier", default=str(DATASET))
    e.add_argument("--epoques", type=int, default=14)
    e.set_defaults(f=cmd_entrainer)

    re_ = s.add_parser("reentrainer", help="réentraîner avec les vraies factures validées dans l'application")
    re_.add_argument("--dossier", default=str(DATASET), help="jeu synthétique à conserver")
    re_.add_argument("--variantes", type=int, default=12, help="mises en scène simulées par document réel")
    re_.add_argument("--epoques", type=int, default=14)
    re_.set_defaults(f=cmd_reentrainer)

    pr = s.add_parser("preparer",help="générer le jeu puis entraîner (première installation)")
    pr.add_argument("--n", type=int, default=400)
    pr.add_argument("--dossier", default=str(DATASET))
    pr.add_argument("--graine", type=int, default=42)
    pr.add_argument("--epoques", type=int, default=14)
    pr.set_defaults(f=cmd_preparer)

    x = s.add_parser("exemples", help="créer des documents de test")
    x.add_argument("--n", type=int, default=20)
    x.add_argument("--dossier", default=str(DONNEES / "exemples"))
    x.add_argument("--graine", type=int, default=2026)
    x.set_defaults(f=cmd_exemples)

    an = s.add_parser("analyser", help="analyser des fichiers ou dossiers")
    an.add_argument("chemins", nargs="+")
    an.add_argument("--sans-ocr", action="store_true")
    an.add_argument("--sans-enregistrement", action="store_true", help="ne pas écrire en base")
    an.add_argument("--json", action="store_true")
    an.set_defaults(f=cmd_analyser)

    ra = s.add_parser("reanalyser", help="réanalyser les documents déjà en base")
    ra.add_argument("--tous", action="store_true", help="inclure les documents corrigés manuellement")
    ra.set_defaults(f=cmd_reanalyser)

    c = s.add_parser("calibrer",help="évaluer les contrôles qualité OpenCV")
    c.add_argument("--dossier", default=str(DATASET))
    c.add_argument("--n", type=int, default=600)
    c.set_defaults(f=cmd_calibrer)

    sv = s.add_parser("serveur", help="lancer l'application web")
    sv.add_argument("--hote", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=5000)
    sv.set_defaults(f=cmd_serveur)

    a = p.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
