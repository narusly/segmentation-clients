"""Tests et tailles d'effet pour comparer les segments.

Avec 30 000 clients, presque toute différence est « significative » : on rapporte donc aussi la taille de
l'effet (V de Cramér, ε² de Kruskal-Wallis), qui dit si la différence compte vraiment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Intervalle de Wilson pour une proportion, fiable même près de 0 ou de 1."""
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


def cramers_v(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Test du χ² d'indépendance et V de Cramér (0 : aucun lien, 1 : lien parfait)."""
    table = pd.crosstab(x, y)
    chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
    n = table.to_numpy().sum()
    v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
    return float(v), float(p)


def kruskal_epsilon2(values: pd.Series, groups: pd.Series) -> tuple[float, float]:
    """Test de Kruskal-Wallis et taille d'effet ε² = H / (n − 1)."""
    samples = [values[groups == g].to_numpy() for g in groups.unique()]
    h, p = stats.kruskal(*samples)
    return float(h / (len(values) - 1)), float(p)
