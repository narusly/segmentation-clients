"""Jeu « Default of Credit Card Clients » (UCI, CC BY 4.0) : téléchargement et indicateurs de comportement.

Yeh, I. C. & Lien, C. H. (2009). 30 000 porteurs de carte à Taïwan, d'avril à septembre 2005 (montants en NT$).
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "credit_card_clients.csv"
URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
MONTHS = 6
PAY = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]          # statut de remboursement, sept. → avril
BILL = [f"BILL_AMT{i}" for i in range(1, MONTHS + 1)]
PAID = [f"PAY_AMT{i}" for i in range(1, MONTHS + 1)]
TARGET = "default"

# Variables de comportement utilisées pour la segmentation (le défaut n'y entre pas : il sert à l'interpréter).
FEATURES = {
    "log_plafond": "Plafond de crédit (log)",
    "utilisation": "Taux d'utilisation du plafond",
    "taux_remboursement": "Part des factures remboursée",
    "part_mois_soldes": "Mois soldés intégralement",
    "part_mois_revolving": "Mois en crédit renouvelable",
    "part_mois_inactifs": "Mois sans utilisation",
    "mois_en_retard": "Mois en retard de paiement",
}


def download(path: Path = RAW) -> Path:
    """Télécharge l'archive UCI une fois et la convertit en CSV."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=120) as resp:
            archive = zipfile.ZipFile(io.BytesIO(resp.read()))
        xls = next(n for n in archive.namelist() if n.endswith(".xls"))
        frame = pd.read_excel(io.BytesIO(archive.read(xls)), header=1)
        frame.rename(columns={"default payment next month": TARGET}).to_csv(path, index=False)
    return path


def load() -> pd.DataFrame:
    return pd.read_csv(download())


def behaviour(df: pd.DataFrame) -> pd.DataFrame:
    """Indicateurs de comportement sur 6 mois.

    Codes PAY_x : −2 pas d'utilisation, −1 soldé intégralement, 0 crédit renouvelable (minimum payé),
    1 à 9 : retard de paiement en mois.
    """
    limit = df["LIMIT_BAL"].astype(float)
    bills = df[BILL].to_numpy(float)
    paid = df[PAID].to_numpy(float)
    status = df[PAY].to_numpy()
    billed = np.clip(bills, 0, None).sum(axis=1)
    out = pd.DataFrame(index=df.index)
    out["log_plafond"] = np.log(limit)
    out["utilisation"] = np.clip(np.clip(bills, 0, None).mean(axis=1) / limit, 0, 1.5)
    # Pas de facture sur la période : rien à rembourser, on considère le client à jour.
    out["taux_remboursement"] = np.where(billed > 0, np.clip(paid.sum(axis=1) / np.where(billed > 0, billed, 1), 0, 2), 1.0)
    out["part_mois_soldes"] = (status == -1).mean(axis=1)
    out["part_mois_revolving"] = (status == 0).mean(axis=1)
    out["part_mois_inactifs"] = (status == -2).mean(axis=1)
    out["mois_en_retard"] = (status >= 1).sum(axis=1).astype(float)
    return out[list(FEATURES)]
