"""Persistance des factures analysées (SQLite)."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .config import CHEMIN_BD

SCHEMA = """
CREATE TABLE IF NOT EXISTS factures (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    date_import        TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    fichier_original   TEXT NOT NULL,
    chemin_stocke      TEXT NOT NULL,
    empreinte          TEXT NOT NULL,
    -- Classification CNN
    type_doc           TEXT,
    confiance_type     REAL,
    qualite            TEXT,
    confiance_qualite  REAL,
    probas             TEXT,      -- JSON
    -- Contrôle qualité OpenCV
    mesures_qualite    TEXT,      -- JSON
    -- Données extraites par OCR
    numero             TEXT,
    date_facture       TEXT,
    emetteur           TEXT,
    siret              TEXT,
    tva_intracom       TEXT,
    iban               TEXT,
    montant_ht         REAL,
    taux_tva           REAL,
    montant_tva        REAL,
    montant_ttc        REAL,
    anomalies          TEXT,      -- JSON
    texte_ocr          TEXT,
    moteur_ocr         TEXT,
    -- Orientation
    decision           TEXT,      -- VALIDEE | A_VERIFIER | REJETEE | DOUBLON | NON_FACTURE
    file_traitement    TEXT,
    motifs             TEXT,      -- JSON
    statut             TEXT NOT NULL DEFAULT 'en_attente',  -- en_attente | traitee | corrigee
    duree_ms           INTEGER
);
CREATE INDEX IF NOT EXISTS idx_empreinte ON factures(empreinte);
CREATE INDEX IF NOT EXISTS idx_numero ON factures(numero, emetteur);
CREATE INDEX IF NOT EXISTS idx_decision ON factures(decision);
"""
# Colonnes ajoutées après la première version : créées à l'ouverture si absentes
MIGRATIONS = {"pays": "TEXT", "devise": "TEXT", "net_a_payer": "REAL"}
CHAMPS_JSON = {"probas", "mesures_qualite", "anomalies", "motifs"}
CHAMPS_MODIFIABLES = {"type_doc", "numero", "date_facture", "emetteur", "pays", "devise", "siret", "tva_intracom",
                      "iban", "montant_ht", "taux_tva", "montant_tva", "montant_ttc", "net_a_payer", "decision",
                      "file_traitement", "statut"}


class Base:
    def __init__(self, chemin: Path = CHEMIN_BD):
        self.chemin = str(chemin)
        with self._cnx() as c:
            c.executescript(SCHEMA)
            existantes = {r[1] for r in c.execute("PRAGMA table_info(factures)")}
            for col, typ in MIGRATIONS.items():
                if col not in existantes:
                    c.execute(f"ALTER TABLE factures ADD COLUMN {col} {typ}")

    @contextmanager
    def _cnx(self):
        cnx = sqlite3.connect(self.chemin)
        cnx.row_factory = sqlite3.Row
        try:
            yield cnx
            cnx.commit()
        finally:
            cnx.close()

    @staticmethod
    def _ligne(r: sqlite3.Row | None) -> dict | None:
        if r is None:
            return None
        d = dict(r)
        for k in CHAMPS_JSON:
            if d.get(k):
                d[k] = json.loads(d[k])
        return d

    def inserer(self, donnees: dict) -> int:
        d = {k: (json.dumps(v, ensure_ascii=False) if k in CHAMPS_JSON else v) for k, v in donnees.items()}
        cols = ", ".join(d)
        with self._cnx() as c:
            cur = c.execute(f"INSERT INTO factures ({cols}) VALUES ({', '.join('?' * len(d))})", list(d.values()))
            return cur.lastrowid

    def obtenir(self, id_: int) -> dict | None:
        with self._cnx() as c:
            return self._ligne(c.execute("SELECT * FROM factures WHERE id = ?", (id_,)).fetchone())

    def lister(self, decision: str | None = None, type_doc: str | None = None, recherche: str | None = None,
               limite: int = 500) -> list[dict]:
        sql, params = "SELECT * FROM factures WHERE 1=1", []
        if decision:
            sql += " AND decision = ?"; params.append(decision)
        if type_doc:
            sql += " AND type_doc = ?"; params.append(type_doc)
        if recherche:
            sql += " AND (fichier_original LIKE ? OR numero LIKE ? OR emetteur LIKE ?)"
            params += [f"%{recherche}%"] * 3
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limite)
        with self._cnx() as c:
            return [self._ligne(r) for r in c.execute(sql, params).fetchall()]

    def mettre_a_jour(self, id_: int, champs: dict):
        champs = {k: v for k, v in champs.items() if k in CHAMPS_MODIFIABLES}
        if not champs:
            return
        with self._cnx() as c:
            c.execute(f"UPDATE factures SET {', '.join(f'{k} = ?' for k in champs)} WHERE id = ?",
                      [*champs.values(), id_])

    def supprimer(self, id_: int) -> dict | None:
        ligne = self.obtenir(id_)
        with self._cnx() as c:
            c.execute("DELETE FROM factures WHERE id = ?", (id_,))
        return ligne

    def remplacer_analyse(self, id_: int, donnees: dict):
        """Réécrit le résultat d'analyse d'un document existant (réanalyse)."""
        d = {k: (json.dumps(v, ensure_ascii=False) if k in CHAMPS_JSON else v) for k, v in donnees.items()
             if k not in ("id", "date_import", "fichier_original", "chemin_stocke", "empreinte")}
        with self._cnx() as c:
            c.execute(f"UPDATE factures SET {', '.join(f'{k} = ?' for k in d)} WHERE id = ?", [*d.values(), id_])

    def doublon(self, empreinte: str, numero: str | None, emetteur: str | None,
                ttc: float | None, avant_id: int | None = None) -> dict | None:
        """Même fichier, ou même numéro chez le même émetteur pour le même montant.
        `avant_id` : ne compare qu'aux documents importés avant celui-ci (réanalyse)."""
        filtre, p = ("AND id < ?", [avant_id]) if avant_id else ("", [])
        with self._cnx() as c:
            r = c.execute(f"SELECT id, fichier_original FROM factures WHERE empreinte = ? {filtre} LIMIT 1",
                          [empreinte, *p]).fetchone()
            if r is None and numero and ttc is not None:
                r = c.execute("SELECT id, fichier_original FROM factures WHERE numero = ? AND "
                              "COALESCE(emetteur, '') = COALESCE(?, '') AND ABS(montant_ttc - ?) < 0.01 "
                              f"AND decision != 'REJETEE' {filtre} LIMIT 1", [numero, emetteur, ttc, *p]).fetchone()
            return dict(r) if r else None

    def statistiques(self) -> dict:
        with self._cnx() as c:
            total = c.execute("SELECT COUNT(*) FROM factures").fetchone()[0]
            par = lambda col: {r[0] or "—": r[1] for r in c.execute(
                f"SELECT {col}, COUNT(*) FROM factures GROUP BY {col} ORDER BY COUNT(*) DESC")}
            montant = {r[0]: round(r[1], 2) for r in c.execute(  # une somme par devise : jamais de mélange
                "SELECT devise, SUM(montant_ttc) FROM factures WHERE decision = 'VALIDEE' AND montant_ttc IS NOT NULL "
                "GROUP BY devise ORDER BY SUM(montant_ttc) DESC")}
            return {"total": total, "par_decision": par("decision"), "par_type": par("type_doc"), "par_pays": par("pays"),
                    "par_qualite": par("qualite"), "par_file": par("file_traitement"),
                    "montant_valide": montant}
