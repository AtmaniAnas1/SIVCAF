"""Entraînement et évaluation du CNN sur le jeu de données (synthétique ou réel).

Le jeu de données est un dossier d'images accompagné d'un fichier
`etiquettes.csv` (colonnes : fichier, type, qualite). On peut y ajouter de
vraies factures étiquetées pour affiner le modèle.
"""
from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import CHEMIN_MODELE, DATASET, DONNEES, QUALITES, TYPES, UPLOADS

REELS = DONNEES / "reels"
from .modele import CNNFactures, sauvegarder, vers_tenseurs
from .pretraitement import charger_image, entrees_cnn


def preparer_reels(base, dossier: Path = REELS, variantes: int = 12, graine: int = 7, journal=print) -> int:
    """Construit un jeu d'entraînement à partir des documents réels validés ou corrigés dans l'application.

    Chaque document devient : l'original (conforme) + `variantes` mises en scène simulées
    (photo, scan, flou, mauvais cadrage, exposition) pour enseigner au CNN les vraies mises en page.
    """
    import cv2
    from PIL import Image
    from .synth import scene

    dossier.mkdir(parents=True, exist_ok=True)
    rng = random.Random(graine)
    fiables = [f for f in base.lister(limite=100000)
               if f["decision"] in ("VALIDEE", "NON_FACTURE") or f["statut"] in ("corrigee", "traitee")]
    fiables = [f for f in fiables if f["decision"] not in ("REJETEE", "DOUBLON") and f["type_doc"] in TYPES
               and not f["fichier_original"].startswith("exemple_")]  # les exemples synthétiques sont déjà au jeu
    lignes = []
    for f in fiables:
        try:
            img = charger_image(UPLOADS / f["chemin_stocke"])
        except (OSError, ValueError):
            continue
        cle = f["empreinte"][:12]
        cv2.imwrite(str(dossier / f"{cle}_orig.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        lignes.append((f"{cle}_orig.jpg", f["type_doc"], "conforme"))
        for k in range(variantes):
            q = "conforme" if rng.random() < 0.5 else rng.choice(QUALITES[1:])
            v = scene(rng, Image.fromarray(img), q, f["type_doc"] == "ticket_caisse")
            nom = f"{cle}_{k:02d}_{q}.jpg"
            cv2.imwrite(str(dossier / nom), cv2.cvtColor(v, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
            lignes.append((nom, f["type_doc"], q))
    with open(dossier / "etiquettes.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["fichier", "type", "qualite"])
        w.writerows(lignes)
    journal(f"{len(fiables)} document(s) réel(s) -> {len(lignes)} images dans {dossier}")
    return len(fiables)


def charger_dataset(dossier: Path, journal=print):
    with open(dossier / "etiquettes.csv", encoding="utf-8") as fh:
        lignes = list(csv.DictReader(fh))
    G = np.empty((len(lignes), 224, 224, 3), np.uint8)
    D = np.empty((len(lignes), 224, 224), np.uint8)
    yt = np.array([TYPES.index(l["type"]) for l in lignes])
    yq = np.array([QUALITES.index(l["qualite"]) for l in lignes])
    for i, l in enumerate(lignes):
        G[i], D[i] = entrees_cnn(charger_image(dossier / l["fichier"]))
        if (i + 1) % 500 == 0:
            journal(f"  {i + 1}/{len(lignes)} images prétraitées")
    return G, D, yt, yq


def augmenter(g: torch.Tensor, d: torch.Tensor):
    """Variations photométriques légères (n'altèrent pas l'étiquette de qualité)."""
    n = g.shape[0]
    gain = 1 + (torch.rand(n, 1, 1, 1) - 0.5) * 0.16
    biais = (torch.rand(n, 1, 1, 1) - 0.5) * 0.16
    g = g * gain + biais + torch.randn_like(g) * 0.03
    d = d * gain + biais + torch.randn_like(d) * 0.03
    return g, d


def matrice_confusion(vrai, pred, n):
    m = np.zeros((n, n), int)
    for a, b in zip(vrai, pred):
        m[a, b] += 1
    return m


def evaluer(modele, G, D, yt, yq, taille_lot=64):
    modele.eval()
    pt, pq = [], []
    with torch.no_grad():
        for i in range(0, len(G), taille_lot):
            g, d = vers_tenseurs(G[i:i + taille_lot], D[i:i + taille_lot])
            lt, lq = modele(g, d)
            pt.append(lt.argmax(1).numpy())
            pq.append(lq.argmax(1).numpy())
    pt, pq = np.concatenate(pt), np.concatenate(pq)
    return {
        "precision_type": float((pt == yt).mean()),
        "precision_qualite": float((pq == yq).mean()),
        "confusion_type": matrice_confusion(yt, pt, len(TYPES)).tolist(),
        "confusion_qualite": matrice_confusion(yq, pq, len(QUALITES)).tolist(),
    }


def entrainer(dossiers: Path | list[Path] = DATASET, epoques: int = 14, taille_lot: int = 32, lr: float = 2e-3,
              sortie: Path = CHEMIN_MODELE, graine: int = 0, journal=print) -> dict:
    torch.manual_seed(graine)
    t0 = time.time()
    dossiers = [Path(d) for d in (dossiers if isinstance(dossiers, (list, tuple)) else [dossiers])]
    parts = []
    for d in dossiers:
        journal(f"Prétraitement de {d}…")
        parts.append(charger_dataset(d, journal))
    G, D, yt, yq = (np.concatenate(x) for x in zip(*parts))
    idx = list(range(len(G)))
    random.Random(graine).shuffle(idx)
    n_val = n_test = max(1, len(idx) // 10)
    i_test, i_val, i_train = idx[:n_test], idx[n_test:n_test + n_val], idx[n_test + n_val:]
    journal(f"{len(i_train)} entraînement / {len(i_val)} validation / {len(i_test)} test")

    modele = CNNFactures()
    optim = torch.optim.AdamW(modele.parameters(), lr=lr, weight_decay=1e-4)
    pas_total = epoques * ((len(i_train) + taille_lot - 1) // taille_lot)
    planif = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=lr, total_steps=pas_total)
    perte = nn.CrossEntropyLoss(label_smoothing=0.05)
    meilleur, meilleur_etat = -1.0, None

    for ep in range(1, epoques + 1):
        modele.train()
        random.shuffle(i_train)
        cumul = 0.0
        for k in range(0, len(i_train), taille_lot):
            b = sorted(i_train[k:k + taille_lot])
            g, d = augmenter(*vers_tenseurs(G[b], D[b]))
            lt, lq = modele(g, d)
            l = perte(lt, torch.from_numpy(yt[b])) + perte(lq, torch.from_numpy(yq[b]))
            optim.zero_grad()
            l.backward()
            optim.step()
            planif.step()
            cumul += l.item() * len(b)
        v = evaluer(modele, G[i_val], D[i_val], yt[i_val], yq[i_val])
        score = v["precision_type"] + v["precision_qualite"]
        journal(f"Époque {ep:2d}/{epoques}  perte={cumul / len(i_train):.3f}  "
                f"val type={v['precision_type']:.3f}  val qualité={v['precision_qualite']:.3f}  "
                f"({time.time() - t0:.0f}s)")
        if score > meilleur:
            meilleur = score
            meilleur_etat = {k: t.clone() for k, t in modele.state_dict().items()}

    modele.load_state_dict(meilleur_etat)
    test = evaluer(modele, G[i_test], D[i_test], yt[i_test], yq[i_test])
    test.update({"n_images": len(G), "epoques": epoques, "duree_s": round(time.time() - t0)})
    sortie.parent.mkdir(parents=True, exist_ok=True)
    sauvegarder(modele, sortie, test)
    sortie.with_suffix(".json").write_text(json.dumps(test, indent=2, ensure_ascii=False), encoding="utf-8")
    journal(f"Test : type={test['precision_type']:.3f}  qualité={test['precision_qualite']:.3f}  -> {sortie}")
    return test
