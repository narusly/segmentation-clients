"""Segmentation K-means : choix de k, stabilité par bootstrap, nommage des segments à partir de leurs centres."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.preprocessing import StandardScaler


def standardize(X: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler().fit(X)
    return scaler.transform(X), scaler


def choose_k(Z: np.ndarray, ks=range(2, 9), sample: int = 6000, seed: int = 0) -> pd.DataFrame:
    """Critères internes pour chaque k. La silhouette (coûteuse) est calculée sur un échantillon."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Z), min(sample, len(Z)), replace=False)
    rows = []
    for k in ks:
        labels = KMeans(k, n_init=10, random_state=seed).fit_predict(Z)
        rows.append({"k": k, "inertie": KMeans(k, n_init=10, random_state=seed).fit(Z).inertia_,
                     "silhouette": silhouette_score(Z[idx], labels[idx]),
                     "calinski_harabasz": calinski_harabasz_score(Z, labels),
                     "davies_bouldin": davies_bouldin_score(Z, labels)})
    return pd.DataFrame(rows).set_index("k")


def fit(Z: np.ndarray, k: int, seed: int = 0) -> KMeans:
    return KMeans(k, n_init=20, random_state=seed).fit(Z)


def bootstrap_stability(Z: np.ndarray, reference: KMeans, n_boot: int = 20, seed: int = 1) -> np.ndarray:
    """ARI entre la segmentation de référence et celles refaites sur des rééchantillons bootstrap.

    Chaque modèle bootstrap est appliqué à l'ensemble des clients pour comparer les mêmes individus.
    """
    rng = np.random.default_rng(seed)
    ref = reference.predict(Z)
    scores = []
    for b in range(n_boot):
        idx = rng.choice(len(Z), len(Z), replace=True)
        model = KMeans(reference.n_clusters, n_init=5, random_state=b).fit(Z[idx])
        scores.append(adjusted_rand_score(ref, model.predict(Z)))
    return np.array(scores)


def name_segments(centers: pd.DataFrame) -> dict[int, str]:
    """Nomme les segments par règles sur leurs centres (variables d'origine), dans un ordre de priorité.

    Les règles ne dépendent pas du numéro arbitraire que K-means attribue à chaque segment.
    """
    rules = [
        ("mois_en_retard", "En difficulté de paiement"),
        ("part_mois_inactifs", "Peu actifs"),
        ("part_mois_soldes", "Transacteurs"),
        ("part_mois_revolving", "Revolvers"),
    ]
    names: dict[int, str] = {}
    remaining = centers.copy()
    for column, name in rules:
        if remaining.empty:
            break
        seg = int(remaining[column].idxmax())
        names[seg] = name
        remaining = remaining.drop(index=seg)
    for i, seg in enumerate(remaining.index, start=1):
        names[int(seg)] = f"Segment {i}"
    return names
