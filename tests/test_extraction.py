"""Tests d'extraction et de classement textuel sur des textes OCR de plusieurs pays.

Exécution : python -m pytest tests   (ou : python tests/test_extraction.py)
Les blocs d'une même ligne sont séparés par des tabulations, comme en sortie d'OCR.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sivcaf.classement_texte import probas_texte, scores_texte  # noqa: E402
from sivcaf.extraction import extraire, verifier  # noqa: E402

CAS = {
    "maroc_lydec": ("""Lydec\tFacture d'électricité et d'eau
Facture N° : 2024031245789\tDate de facture : 05/03/2024
Client : M. Karim Alaoui\tCasablanca
ICE : 001524789000063
Consommation électricité\t245 kWh\t312,45 DH
Consommation eau\t18 m3\t98,20 DH
Total HT\t410,65 DH
TVA 14 %\t57,49 DH
Total TTC\t468,14 DH
Net à payer\t468,14 DH""",
                    {"pays": "MA", "devise": "MAD", "numero": "2024031245789", "date_facture": "2024-03-05",
                     "montant_ht": 410.65, "montant_tva": 57.49, "montant_ttc": 468.14, "taux_tva": 14.0,
                     "siret": "ICE 001524789000063", "emetteur": "Lydec"}),
    "maroc_fournisseur": ("""ATLAS BUREAUTIQUE SARL\tFACTURE
Bd Zerktouni, Casablanca\tFacture N° FA-2025-0042
ICE : 002233445000012\tIF : 45123987\tRC : 112233
Date : 12 février 2025
Désignation\tQté\tP.U HT\tTotal HT
Imprimante laser\t2\t2 450,00\t4 900,00
Total HT\t4 900,00
TVA 20%\t980,00
Total TTC\t5 880,00 DH
RIB : 011 780 0000123456789012 34""",
                          {"pays": "MA", "numero": "FA-2025-0042", "date_facture": "2025-02-12",
                           "montant_ttc": 5880.0, "montant_ht": 4900.0, "emetteur": "ATLAS BUREAUTIQUE SARL",
                           "_type": "fournisseur"}),
    # Texte OCR réel d'une facture inwi (police fine : mots collés, « Dh » intercalé)
    "maroc_inwi_ocr": ("""inWi\tFACTURE
FACTUREN'\tF24121224239\tNEJMAT AL BAYDAE
Dotedefocturation:\t01/12/2024
Encil:\tscrviccclicntscotrcpriscs@inwimo\tMarcc
N? Compte facturation:\t10084483349\tFacture pour vos:
CASABLANCA
Abonnements\t16583
Communicotions\t000
Totol DhHT\t165,83
Monten: TVADn20%\t3317
Total DhTTC\t199,00
Montant Totalàpayer Dh TTC\t199,00
Reglement àeffectueràlordrede\tWANA CORPORATE""",
                       {"pays": "MA", "numero": "F24121224239", "date_facture": "2024-12-01", "montant_ht": 165.83,
                        "montant_tva": 33.17, "montant_ttc": 199.0, "taux_tva": 20.0, "emetteur": "INWI",
                        "anomalies": [], "_type": "telecom"}),
    "algerie_sonelgaz": ("""Sonelgaz\tSociété de Distribution de l'Electricité et du Gaz
Facture N° : 162204517832\tDate : 14/06/2023
Consommation électricité\t412 kWh
Montant HT (DA)\t2 315,40
TVA 19 %\t439,93
Montant TTC (DA)\t2 755,33
Net à payer (DA)\t2 755,33""",
                         {"pays": "DZ", "devise": "DZD", "montant_ttc": 2755.33, "taux_tva": 19.0,
                          "emetteur": "Sonelgaz", "_type": "electricite_gaz"}),
    "tunisie_telecom": ("""Tunisie Telecom\tFacture Mobile
Facture N° 7781234\tDate d'émission : 02/11/2024
Matricule fiscal : 0000123A/A/M/000
Forfait mobile 20 Go\t25,210 DT
Total HT\t21,185
TVA 19 %\t4,025
Total TTC\t25,210 DT""",
                        {"pays": "TN", "numero": "7781234", "date_facture": "2024-11-02", "montant_ttc": 25.21,
                         "montant_ht": 21.185, "montant_tva": 4.025, "anomalies": [], "_type": "telecom"}),
    "senegal_senelec": ("""SENELEC\tFacture d'électricité
Facture N° 4501238874\tDate de facturation : 10/01/2025
NINEA : 0012345 2G3
Consommation\t180 kWh\t21 600
TVA 18 %\t3 888
Montant TTC\t25 488 FCFA""",
                        {"pays": "SN", "devise": "XOF", "montant_ttc": 25488.0, "montant_tva": 3888.0,
                         "_type": "electricite_gaz"}),
    "france_edf": ("""EDF\tVotre facture d'électricité
Facture n° 10087654321\tDate : 15 mars 2024
PDL N° 12345678901234
Total HT\t85,40 €
TVA 20 %\t17,08 €
Total TTC\t102,48 €
EDF SA - SIRET 552 081 317 00048""",
                   {"pays": "FR", "devise": "EUR", "numero": "10087654321", "date_facture": "2024-03-15",
                    "montant_ttc": 102.48, "siret": "SIRET 55208131700048", "_type": "electricite_gaz"}),
    "rdc_regideso": ("""FACTURE DE CONSOMMATION D'EAU\tFacture : 0020035100263
REGIDESO KATANGA\tDate : 31/03/2020
Eau de 01 à 10 m3\t10,00\t503.2000\t5 032,00
TVA (16%)\t45.971,70\t0.1600\t7 355,47
Montant de la FACTURE\t53 327.17
TOTAL A PAYER\t193 532.82
Net à payer en FC\t193   532,82""",
                     {"pays": "CD", "devise": "CDF", "numero": "0020035100263", "montant_ttc": 53327.17,
                      "net_a_payer": 193532.82, "montant_tva": 7355.47, "montant_ht": 45971.7, "_type": "eau"}),
    "belgique": ("""Proximus\tFacture
Facture n° 512345678\tDate : 03.04.2024
TVA BE 0202.239.951
Total HTVA\t41,32 €
TVA 21 %\t8,68 €
Total TVAC\t50,00 €""",
                 {"pays": "BE", "date_facture": "2024-04-03", "tva_intracom": "BE0202.239.951",
                  "montant_ht": 41.32, "montant_ttc": 50.0, "anomalies": []}),
}


def test_cas():
    erreurs = []
    for nom, (texte, attendu) in CAS.items():
        c = extraire(texte)
        type_attendu = attendu.get("_type")
        if type_attendu:
            p = probas_texte(scores_texte(texte))
            obtenu = max(p, key=p.get) if p else None
            if obtenu != type_attendu:
                erreurs.append(f"{nom}: type {obtenu} != {type_attendu}")
        verifier(c, type_attendu or "fournisseur")
        for cle, val in attendu.items():
            if not cle.startswith("_") and getattr(c, cle) != val:
                erreurs.append(f"{nom}: {cle} = {getattr(c, cle)!r}, attendu {val!r}")
    assert not erreurs, "\n".join(erreurs)


if __name__ == "__main__":
    try:
        test_cas()
        print(f"OK : {len(CAS)} cas")
    except AssertionError as e:
        print(e)
        sys.exit(1)
