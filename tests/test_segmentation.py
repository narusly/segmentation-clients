import numpy as np
import pandas as pd
import pytest

from segmentation.data import BILL, PAID, PAY, behaviour
from segmentation.model import bootstrap_stability, fit, name_segments, standardize
from segmentation.stats import cramers_v, kruskal_epsilon2, wilson_interval


def client(limit, bills, paid, status):
    row = {"LIMIT_BAL": limit}
    row.update(dict(zip(BILL, bills)))
    row.update(dict(zip(PAID, paid)))
    row.update(dict(zip(PAY, status)))
    return row


def test_behaviour_indicators():
    df = pd.DataFrame([
        client(100_000, [50_000] * 6, [5_000] * 6, [0] * 6),         # revolver : 10 % remboursé
        client(200_000, [0] * 6, [0] * 6, [-2] * 6),                  # inactif : aucune facture
        client(50_000, [40_000] * 6, [0] * 6, [2, 2, 1, 1, 0, 0]),    # quatre mois de retard
    ])
    b = behaviour(df)
    assert b.loc[0, "utilisation"] == pytest.approx(0.5)
    assert b.loc[0, "taux_remboursement"] == pytest.approx(0.1)
    assert b.loc[0, "part_mois_revolving"] == 1.0
    assert b.loc[1, "taux_remboursement"] == 1.0                       # rien à rembourser : à jour
    assert b.loc[1, "part_mois_inactifs"] == 1.0
    assert b.loc[2, "mois_en_retard"] == 4
    assert b.loc[0, "log_plafond"] == pytest.approx(np.log(100_000))


def test_name_segments_follows_profiles_not_label_numbers():
    centers = pd.DataFrame({
        "mois_en_retard": [0.3, 4.0, 0.2, 0.3],
        "part_mois_inactifs": [0.0, 0.0, 0.1, 0.9],
        "part_mois_soldes": [0.0, 0.0, 0.8, 0.1],
        "part_mois_revolving": [0.9, 0.3, 0.1, 0.0],
    }, index=[3, 0, 1, 2])
    assert name_segments(centers) == {0: "En difficulté de paiement", 2: "Peu actifs", 1: "Transacteurs",
                                      3: "Revolvers"}


def test_stability_on_separated_groups():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(np.vstack([rng.normal(c, 0.3, size=(300, 2)) for c in ((0, 0), (5, 0), (0, 5))]))
    Z, _ = standardize(X)
    ari = bootstrap_stability(Z, fit(Z, 3), n_boot=5)
    assert ari.min() > 0.99


def test_wilson_interval():
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3)
    assert hi == pytest.approx(0.5962, abs=1e-3)
    lo0, _ = wilson_interval(0, 100)
    assert lo0 == pytest.approx(0.0, abs=1e-12)


def test_cramers_v_extremes():
    x = pd.Series(["a", "b"] * 500)
    assert cramers_v(x, x)[0] == pytest.approx(1.0)
    y = pd.Series(np.random.default_rng(1).choice(["u", "v"], 1000))
    assert cramers_v(x, y)[0] < 0.1


def test_kruskal_effect_size():
    groups = pd.Series(["a"] * 200 + ["b"] * 200)
    same = pd.Series(np.random.default_rng(2).normal(size=400))
    shifted = pd.Series(np.r_[np.zeros(200), np.ones(200)] + np.random.default_rng(3).normal(0, 0.1, 400))
    assert kruskal_epsilon2(same, groups)[0] < 0.02
    assert kruskal_epsilon2(shifted, groups)[0] > 0.5
