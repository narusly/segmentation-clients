"""Analyse complète : `python -m segmentation.report` écrit `reports/resultats.md` et les figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from . import style
from .data import FEATURES, ROOT, TARGET, behaviour, load
from .model import bootstrap_stability, choose_k, fit, name_segments, standardize
from .stats import cramers_v, kruskal_epsilon2, wilson_interval

REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
K, SEED = 4, 0
ORDER = ["Transacteurs", "Revolvers", "Peu actifs", "En difficulté de paiement"]
# Divergente bleu ↔ rouge, milieu gris neutre : sous la moyenne / au-dessus de la moyenne.
DIVERGING = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#f2a7a6", "#c43c3b"])


def fr(x: float, decimals: int = 1) -> str:
    return f"{x:,.{decimals}f}".replace(",", " ").replace(".", ",")


def pct(x: float, decimals: int = 0) -> str:
    return f"{fr(100 * x, decimals)} %"


def fmt_p(p: float) -> str:
    """p-value lisible ; en dessous de la précision machine, on donne une borne."""
    if p < 1e-300:
        return "p < 10⁻³⁰⁰"
    mantissa, exponent = f"{p:.1e}".split("e")
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return f"p = {mantissa.replace('.', ',')} × 10{str(int(exponent)).translate(sup)}"


SHORT = {"log_plafond": "Plafond\nmédian (NT$)", "utilisation": "Utilisation\ndu plafond",
         "taux_remboursement": "Factures\nremboursées", "part_mois_soldes": "Mois soldés\nintégralement",
         "part_mois_revolving": "Mois en crédit\nrenouvelable", "part_mois_inactifs": "Mois sans\nutilisation",
         "mois_en_retard": "Mois en retard\nde paiement"}


def fig_choose_k(crit: pd.DataFrame, path: Path) -> None:
    panels = (("silhouette", "Silhouette (plus haut = mieux)"),
              ("calinski_harabasz", "Calinski-Harabasz (plus haut = mieux)"),
              ("davies_bouldin", "Davies-Bouldin (plus bas = mieux)"))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (col, title) in zip(axes, panels):
        ax.plot(crit.index, crit[col], color=style.SERIES[0], marker="o", markersize=4)
        ax.scatter([K], [crit.loc[K, col]], s=70, color=style.SERIES[0], edgecolors=style.SURFACE, linewidths=2,
                   zorder=3)
        below = col != "davies_bouldin"          # maximum en haut du panneau : étiquette dessous
        ax.annotate(f"k = {K}", (K, crit.loc[K, col]), xytext=(9, -14 if below else 8), textcoords="offset points",
                    fontsize=9, color=style.INK)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Nombre de segments k")
        ax.set_xticks(crit.index)
        ax.yaxis.set_major_formatter(lambda v, _: fr(v, 2) if v < 100 else fr(v, 0))
    fig.suptitle("Choix du nombre de segments : les trois critères désignent k = 4", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_profiles(profile: pd.DataFrame, z: pd.DataFrame, sizes: pd.Series, path: Path) -> None:
    """Carte de chaleur : couleur = écart à la moyenne (en écarts-types), texte = valeur réelle."""
    fmt = {"log_plafond": lambda v: f"{fr(np.exp(v) / 1000, 0)} k", "utilisation": pct,
           "taux_remboursement": pct, "part_mois_soldes": pct, "part_mois_revolving": pct,
           "part_mois_inactifs": pct, "mois_en_retard": lambda v: f"{fr(v)} mois"}
    fig, ax = plt.subplots(figsize=(12, 3.8))
    lim = float(np.abs(z.to_numpy()).max())
    ax.imshow(z.to_numpy(), cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
    for i, seg in enumerate(profile.index):
        for j, col in enumerate(profile.columns):
            strong = abs(z.loc[seg, col]) > 0.6 * lim
            ax.text(j, i, fmt[col](profile.loc[seg, col]), ha="center", va="center", fontsize=9,
                    color="#ffffff" if strong else style.INK)
    ax.set_xticks(range(len(profile.columns)), [SHORT[c] for c in profile.columns], fontsize=8.5)
    ax.set_yticks(range(len(profile.index)), [f"{s} ({pct(sizes[s])})" for s in profile.index])
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(profile.columns)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(profile.index)), minor=True)
    ax.grid(which="minor", color=style.SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.set_title("Profil des segments · bleu : sous la moyenne des clients, rouge : au-dessus")
    fig.savefig(path)
    plt.close(fig)


def fig_share(summary: pd.DataFrame, path: Path) -> None:
    """Part des clients et part des défauts par segment : deux séries, une seule échelle (en %)."""
    y = np.arange(len(summary))
    h = 0.36
    fig, ax = plt.subplots(figsize=(9, 3.9))
    ax.barh(y - h / 2 - 0.01, summary["part_clients"], height=h, color=style.SERIES[0], label="Part des clients")
    ax.barh(y + h / 2 + 0.01, summary["part_defauts"], height=h, color=style.SERIES[1], label="Part des défauts")
    for i, (_, row) in enumerate(summary.iterrows()):
        ax.annotate(pct(row["part_clients"]), (row["part_clients"], i - h / 2), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=style.INK_2)
        ax.annotate(f"{pct(row['part_defauts'])}  ·  taux de défaut {pct(row['taux_defaut'])}",
                    (row["part_defauts"], i + h / 2), xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=8.5, color=style.INK_2)
    ax.set_yticks(y, summary.index)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(lambda v, _: pct(v))
    ax.set_xlim(0, summary[["part_clients", "part_defauts"]].to_numpy().max() * 1.45)
    ax.grid(axis="y", visible=False)
    ax.set_title("Qui fait défaut ? Part de chaque segment dans les clients et dans les défauts")
    ax.legend(loc="lower right")
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    style.apply()
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = load()
    X = behaviour(df)
    Z, _ = standardize(X)

    crit = choose_k(Z, seed=SEED)
    model = fit(Z, K, seed=SEED)
    ari = bootstrap_stability(Z, model)

    labels = pd.Series(model.labels_, index=X.index)
    centers = X.groupby(labels).mean()
    names = name_segments(centers)
    segment = labels.map(names)

    profile = X.groupby(segment).mean().loc[ORDER]
    profile["log_plafond"] = np.log(df["LIMIT_BAL"].groupby(segment).median().loc[ORDER])
    z = (profile - X.mean()) / X.std()
    sizes = segment.value_counts(normalize=True)

    defaults = df[TARGET]
    rows = []
    for seg in ORDER:
        mask = segment == seg
        k, n = int(defaults[mask].sum()), int(mask.sum())
        lo, hi = wilson_interval(k, n)
        rows.append({"segment": seg, "clients": n, "part_clients": n / len(df), "défauts": k,
                     "part_defauts": k / defaults.sum(), "taux_defaut": k / n, "ic_bas": lo, "ic_haut": hi,
                     "âge médian": float(df.loc[mask, "AGE"].median())})
    summary = pd.DataFrame(rows).set_index("segment")

    v, p_chi2 = cramers_v(segment, defaults)
    eps_age, p_age = kruskal_epsilon2(df["AGE"], segment)
    eps_feat = {c: kruskal_epsilon2(X[c], segment)[0] for c in X.columns}

    fig_choose_k(crit, FIGURES / "choix_k.png")
    fig_profiles(profile, z, sizes, FIGURES / "profils.png")
    fig_share(summary, FIGURES / "defauts_par_segment.png")

    crit_md = crit.assign(inertie=crit["inertie"].map(lambda x: fr(x, 0)),
                          silhouette=crit["silhouette"].map(lambda x: fr(x, 3)),
                          calinski_harabasz=crit["calinski_harabasz"].map(lambda x: fr(x, 0)),
                          davies_bouldin=crit["davies_bouldin"].map(lambda x: fr(x, 3)))
    seg_md = pd.DataFrame({
        "clients": summary["clients"].map(lambda x: fr(x, 0)),
        "part des clients": summary["part_clients"].map(pct),
        "taux de défaut [IC 95 %]": [f"{pct(r.taux_defaut, 1)} [{pct(r.ic_bas, 1)} ; {pct(r.ic_haut, 1)}]"
                                     for r in summary.itertuples()],
        "part des défauts": summary["part_defauts"].map(pct),
        "âge médian": summary["âge médian"].map(lambda x: fr(x, 0)),
    })
    lines = [
        "# Résultats",
        "",
        f"{fr(len(df), 0)} porteurs de carte, taux de défaut global {pct(defaults.mean(), 1)}. "
        f"Segmentation sur {len(FEATURES)} indicateurs de comportement standardisés ; le défaut n'est utilisé "
        "qu'après coup, pour interpréter les segments.",
        "",
        "## Choix de k",
        "",
        crit_md.to_markdown(),
        "",
        f"Stabilité (k = {K}, 20 rééchantillons bootstrap) : ARI moyen {fr(ari.mean(), 3)}, minimum {fr(ari.min(), 3)}.",
        "",
        "## Segments",
        "",
        seg_md.to_markdown(),
        "",
        "## Liens statistiques",
        "",
        f"- Segment × défaut : test du χ², {fmt_p(p_chi2)}, V de Cramér = {fr(v, 3)}.",
        f"- Âge selon le segment : Kruskal-Wallis, {fmt_p(p_age)}, ε² = {fr(eps_age, 4)} "
        "(différence significative mais négligeable).",
        "- Taille d'effet ε² des variables de segmentation : "
        + ", ".join(f"{FEATURES[c].lower()} {fr(e, 2)}" for c, e in sorted(eps_feat.items(), key=lambda t: -t[1])) + ".",
        "",
    ]
    (REPORTS / "resultats.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
