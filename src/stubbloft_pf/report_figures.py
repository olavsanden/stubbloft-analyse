# -*- coding: utf-8 -*-
"""
report_figures.py
-----------------
Kuraterte, rapportklare figurer og tabeller for bacheloroppgaven.

Figurer lagres i figures/main/ (PNG) og figures/main_pdf/ (PDF).
Tabeller lagres i output/tables/.

Beregningslogikk, PF/LF-definisjon og scenarioverdier er ikke endret her.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from .plots import (
    dose_percent_to_pf,
    pf_to_dose_percent,
    nice_number_label,
    setup_pf_axis_with_dose_axis,
)

# ── Konsistente farger ─────────────────────────────────────────────────────
C = {
    "stubb":      "#C0392B",   # mørk rød — stubbloft bevart
    "utskiftet":  "#2471A3",   # mørk blå — full utskifting
    "ref":        "#7F8C8D",   # grå — referanse lett
    "band_stubb": "#F5CBA7",   # lys rød — sensitivitetsbånd stubb
    "band_utsk":  "#AED6F1",   # lys blå — sensitivitetsbånd utskiftet
    "ground":     "#E67E22",   # oransje — ground shine
    "roof":       "#1A7D3A",   # mørk grønn — roof shine
}

# ── Fontstørrelser ─────────────────────────────────────────────────────────
FS = {"title": 14, "label": 12, "tick": 11, "legend": 10, "annot": 9}

# ── Etasjerekkefølge ───────────────────────────────────────────────────────
FLOOR_ORDER = ["Kjeller", "1. etasje", "2. etasje", "3. etasje", "4. etasje", "5. etasje"]

# ── Scenario-navn ──────────────────────────────────────────────────────────
SC_MAIN = "Kjellervindu 40x60 cm"
LABEL_STUBB = "Stubbloft bevart"
LABEL_UTSK  = "Full utskifting"


# ── Hjelpefunksjoner ───────────────────────────────────────────────────────

def _save(fig: plt.Figure, png_path: Path, pdf_dir: Optional[Path] = None, dpi: int = 300) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    if pdf_dir is not None:
        pdf_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = pdf_dir / png_path.with_suffix(".pdf").name
        fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def _floor_positions(df_sub: pd.DataFrame):
    """Returnerer (y_positions, labels) sortert Kjeller–5.etasje."""
    order = {lbl: i for i, lbl in enumerate(FLOOR_ORDER)}
    df_sorted = df_sub.copy()
    df_sorted["_ord"] = df_sorted["label"].map(order)
    df_sorted = df_sorted.sort_values("_ord").reset_index(drop=True)
    y = np.arange(len(df_sorted))
    return y, df_sorted["label"].tolist(), df_sorted


def _filter(df: pd.DataFrame, building_contains: str, scenario: str = SC_MAIN) -> pd.DataFrame:
    return df[(df["scenario"] == scenario) & df["building"].str.contains(building_contains, regex=False)]


def _pf_xaxis(ax: plt.Axes, pf_max: float) -> None:
    ax.set_xscale("log")
    candidates = [1, 2, 5, 10, 20, 50, 100, 200]
    x_right = next((t for t in candidates if t >= pf_max * 1.1), 200)
    ticks = [t for t in candidates if 1 <= t <= x_right]
    ax.set_xlim(1, x_right)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], fontsize=FS["tick"])
    ax.set_xlabel("Beskyttelsesfaktor PF [-]", fontsize=FS["label"])


def _style_ax(ax: plt.Axes) -> None:
    ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
    ax.tick_params(labelsize=FS["tick"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Figur 01 + 02: PF-profil per fasade ───────────────────────────────────

def fig_pf_profile(
    df: pd.DataFrame,
    facade: str,
    fig_num: str,
    main_dir: Path,
    pdf_dir: Optional[Path],
) -> None:
    """PF per etasje — lokal log-skala. Band = spenn over alle scenarios."""

    bld_stubb = f"{facade}bygg | stubbloft bevart"
    bld_utsk  = f"{facade}bygg | full utskifting + mineralull"
    scenarios_all = df["scenario"].unique()

    # Samle PF-spenn over alle scenarios
    pf_stubb_all, pf_utsk_all = [], []
    for sc in scenarios_all:
        s = _filter(df, f"{facade}bygg | stubbloft", sc)
        u = _filter(df, f"{facade}bygg | full utskifting", sc)
        if not s.empty and not u.empty:
            _, _, s = _floor_positions(s)
            _, _, u = _floor_positions(u)
            pf_stubb_all.append(s["PF"].values)
            pf_utsk_all.append(u["PF"].values)

    # Hent main-scenario
    d_stubb = _filter(df, f"{facade}bygg | stubbloft")
    d_utsk  = _filter(df, f"{facade}bygg | full utskifting")
    y, labels, d_stubb = _floor_positions(d_stubb)
    _,       _, d_utsk  = _floor_positions(d_utsk)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Sensitivitetsbånd (alle kjellervindu-scenarioer)
    if len(pf_stubb_all) > 1:
        arr_s = np.array(pf_stubb_all)
        arr_u = np.array(pf_utsk_all)
        ax.fill_betweenx(y, arr_s.min(axis=0), arr_s.max(axis=0),
                         alpha=0.25, color=C["band_stubb"], zorder=1)
        ax.fill_betweenx(y, arr_u.min(axis=0), arr_u.max(axis=0),
                         alpha=0.25, color=C["band_utsk"], zorder=1)

    # Hovedlinjer
    ax.plot(d_stubb["PF"].values, y, color=C["stubb"], linewidth=2.5,
            marker="o", markersize=7, label=LABEL_STUBB, zorder=3)
    ax.plot(d_utsk["PF"].values,  y, color=C["utskiftet"], linewidth=2.5,
            marker="s", markersize=7, label=LABEL_UTSK,  zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=FS["tick"])
    ax.set_ylabel("Etasje", fontsize=FS["label"])
    pf_max = max(d_stubb["PF"].max(), d_utsk["PF"].max())
    _pf_xaxis(ax, pf_max)
    _style_ax(ax)
    ax.legend(fontsize=FS["legend"], framealpha=0.9, loc="lower right")
    ax.set_title(f"PF per etasje — {facade}bygg", fontsize=FS["title"], pad=10)

    fig.tight_layout()
    fname = f"{fig_num}_pf_profile_{facade.lower()}bygg.png"
    _save(fig, main_dir / fname, pdf_dir)
    print(f"  ✓ {fname}")


# ── Figur 03: PF-ratio ─────────────────────────────────────────────────────

def fig_pf_ratio(
    df: pd.DataFrame,
    facade: str,
    fig_num: str,
    main_dir: Path,
    pdf_dir: Optional[Path],
) -> None:
    """PF-ratio = PF_stubb / PF_utskiftet per etasje."""

    d_stubb = _filter(df, f"{facade}bygg | stubbloft")
    d_utsk  = _filter(df, f"{facade}bygg | full utskifting")
    y, labels, d_stubb = _floor_positions(d_stubb)
    _,       _, d_utsk  = _floor_positions(d_utsk)

    ratio = d_stubb["PF"].values / d_utsk["PF"].values

    fig, ax = plt.subplots(figsize=(7, 6))

    colors = [C["stubb"] if r > 2 else C["utskiftet"] if r < 1.5 else "#888" for r in ratio]
    bars = ax.barh(y, ratio, color=colors, alpha=0.85, height=0.6)

    # Annotasjoner
    for i, (r, bar) in enumerate(zip(ratio, bars)):
        ax.text(r + 0.1, i, f"{r:.1f}×", va="center", fontsize=FS["annot"],
                color="#333", fontweight="bold")

    ax.axvline(1.0, color="k", linewidth=1.2, linestyle="--", label="ratio = 1")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=FS["tick"])
    ax.set_ylabel("Etasje", fontsize=FS["label"])
    ax.set_xlabel("PF_stubbloft / PF_utskiftet", fontsize=FS["label"])
    ax.set_xlim(0, ratio.max() * 1.25)
    _style_ax(ax)
    ax.legend(fontsize=FS["legend"])
    ax.set_title(f"PF-ratio: stubbloft bevart / full utskifting — {facade}bygg",
                 fontsize=FS["title"], pad=10)

    fig.tight_layout()
    fname = f"{fig_num}_pf_ratio_{facade.lower()}bygg.png"
    _save(fig, main_dir / fname, pdf_dir)
    print(f"  ✓ {fname}")


# ── Figur 04: Ground/Roof-andeler ─────────────────────────────────────────

def fig_ground_roof(
    df: pd.DataFrame,
    facade: str,
    fig_num: str,
    main_dir: Path,
    pdf_dir: Optional[Path],
) -> None:
    """Stablet horisontal søyleplott: ground og roof shine per etasje."""

    d_stubb = _filter(df, f"{facade}bygg | stubbloft")
    d_utsk  = _filter(df, f"{facade}bygg | full utskifting")
    y, labels, d_stubb = _floor_positions(d_stubb)
    _,       _, d_utsk  = _floor_positions(d_utsk)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)

    for ax, d, title in [
        (axes[0], d_stubb, f"Stubbloft bevart ({SC_MAIN})"),
        (axes[1], d_utsk,  f"Full utskifting ({SC_MAIN})"),
    ]:
        g_pct = 100 * d["ground_frac"].values
        r_pct = 100 * d["roof_frac"].values
        ax.barh(y, g_pct, color=C["ground"], label="Ground shine", height=0.6)
        ax.barh(y, r_pct, left=g_pct, color=C["roof"], label="Roof shine", height=0.6)
        ax.axvline(50, color="#555", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Andel av innendørs dose [%]", fontsize=FS["label"])
        ax.set_title(title, fontsize=FS["title"] - 1, pad=8)
        _style_ax(ax)
        ax.tick_params(labelsize=FS["tick"])

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=FS["tick"])
    axes[0].set_ylabel("Etasje", fontsize=FS["label"])
    axes[0].legend(fontsize=FS["legend"], loc="lower right")

    fig.suptitle(f"Ground/roof shine-andeler per etasje — {facade}bygg",
                 fontsize=FS["title"], y=1.02)
    fig.tight_layout()
    fname = f"{fig_num}_ground_roof_{facade.lower()}bygg.png"
    _save(fig, main_dir / fname, pdf_dir)
    print(f"  ✓ {fname}")


# ── Figur 05: Mekanismeoversikt ────────────────────────────────────────────

def fig_mechanism(
    df: pd.DataFrame,
    facade: str,
    fig_num: str,
    main_dir: Path,
    pdf_dir: Optional[Path],
) -> None:
    """PF-ratio med ground/roof-andel som sektordiagram per etasje."""

    d_stubb = _filter(df, f"{facade}bygg | stubbloft")
    d_utsk  = _filter(df, f"{facade}bygg | full utskifting")
    y, labels, d_stubb = _floor_positions(d_stubb)
    _,       _, d_utsk  = _floor_positions(d_utsk)

    ratio = d_stubb["PF"].values / d_utsk["PF"].values
    g_pct = 100 * d_stubb["ground_frac"].values
    r_pct = 100 * d_stubb["roof_frac"].values

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5.5))

    # Venstre: PF-ratio
    bar_colors = [C["stubb"] if r > 3 else "#B7B7B7" for r in ratio]
    ax_left.barh(y, ratio, color=bar_colors, alpha=0.85, height=0.6)
    for i, r in enumerate(ratio):
        ax_left.text(r + 0.15, i, f"{r:.1f}×", va="center",
                     fontsize=FS["annot"], fontweight="bold", color="#333")
    ax_left.axvline(1.0, color="k", linewidth=1, linestyle="--")
    ax_left.set_yticks(y)
    ax_left.set_yticklabels(labels, fontsize=FS["tick"])
    ax_left.set_ylabel("Etasje", fontsize=FS["label"])
    ax_left.set_xlabel("PF-ratio (stubb / utskiftet)", fontsize=FS["label"])
    ax_left.set_xlim(0, ratio.max() * 1.3)
    ax_left.set_title("PF-gevinst ved å bevare stubbloft", fontsize=FS["title"] - 1, pad=8)
    _style_ax(ax_left)

    # Høyre: stablede ground/roof-andeler for stubbloft
    ax_right.barh(y, g_pct, color=C["ground"], label="Ground shine", height=0.6, alpha=0.85)
    ax_right.barh(y, r_pct, left=g_pct, color=C["roof"], label="Roof shine", height=0.6, alpha=0.85)
    ax_right.axvline(50, color="#555", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_right.set_xlim(0, 100)
    ax_right.set_xlabel("Andel av innendørs dose [%]", fontsize=FS["label"])
    ax_right.set_title(f"Kilde til innendørs dose — stubb ({SC_MAIN})", fontsize=FS["title"] - 1, pad=8)
    ax_right.legend(fontsize=FS["legend"], loc="lower right")
    ax_right.tick_params(labelsize=FS["tick"])
    _style_ax(ax_right)

    fig.suptitle(f"PF-gevinst og strålingskilde per etasje — {facade}bygg",
                 fontsize=FS["title"], y=1.02)
    fig.tight_layout()
    fname = f"{fig_num}_mechanism_{facade.lower()}bygg.png"
    _save(fig, main_dir / fname, pdf_dir)
    print(f"  ✓ {fname}")


# ── Figur 06: Kjelleråpnings-sensitivitet ─────────────────────────────────

def fig_kjeller_sensitivity(
    df: pd.DataFrame,
    facade: str,
    fig_num: str,
    main_dir: Path,
    pdf_dir: Optional[Path],
) -> None:
    """Kjeller-PF for ulike åpningsscenarioer, stubb vs utskiftet."""

    sc_labels = {
        "Kjellervindu 40x60 cm": "Kjellervindu\n40×60 cm",
        "Kjellervindu 60x30 cm": "Kjellervindu\n60×30 cm",
        "Luftespalte 15x15 cm":  "Luftespalte\n15×15 cm",
    }
    scenarios = list(sc_labels.keys())

    pf_stubb, pf_utsk = [], []
    for sc in scenarios:
        row_s = df[(df["scenario"] == sc) & df["building"].str.contains(f"{facade}bygg | stubbloft", regex=False) & (df["label"] == "Kjeller")]
        row_u = df[(df["scenario"] == sc) & df["building"].str.contains(f"{facade}bygg | full utskifting", regex=False) & (df["label"] == "Kjeller")]
        pf_stubb.append(float(row_s["PF"].iloc[0]) if len(row_s) else 0)
        pf_utsk.append(float(row_u["PF"].iloc[0]) if len(row_u) else 0)

    x = np.arange(len(scenarios))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, pf_stubb, width, color=C["stubb"], label=LABEL_STUBB, alpha=0.85)
    ax.bar(x + width/2, pf_utsk,  width, color=C["utskiftet"], label=LABEL_UTSK,  alpha=0.85)

    for i, (s, u) in enumerate(zip(pf_stubb, pf_utsk)):
        ax.text(i - width/2, s + 1, f"{s:.0f}", ha="center", va="bottom",
                fontsize=FS["annot"], color=C["stubb"], fontweight="bold")
        ax.text(i + width/2, u + 1, f"{u:.0f}", ha="center", va="bottom",
                fontsize=FS["annot"], color=C["utskiftet"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([sc_labels[s] for s in scenarios], fontsize=FS["tick"])
    ax.set_ylabel("Kjeller-PF [-]", fontsize=FS["label"])
    ax.set_title(f"Kjeller-PF per kjelleråpning — {facade}bygg", fontsize=FS["title"], pad=10)
    ax.legend(fontsize=FS["legend"])
    _style_ax(ax)
    fig.tight_layout()

    fname = f"{fig_num}_kjeller_sensitivity_{facade.lower()}bygg.png"
    _save(fig, main_dir / fname, pdf_dir)
    print(f"  ✓ {fname}")


# ── Tabeller ───────────────────────────────────────────────────────────────

def _make_tables(df: pd.DataFrame, output_dir: Path) -> None:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    from .scenarios import M_STUBB_BEVART, M_FULL_UTSKIFTING
    from .physics import light_reference_floor_mass_kg_m2

    # ── Tabell 1: scenario_summary ──────────────────────────────────────
    rows_t1 = [
        {"byggtype": "Murbygg", "etasjeskiller": "Stubbloft bevart",          "M_floor": M_STUBB_BEVART,          "M_wall": 450, "M_foundation": 650, "rolle": "Hoved"},
        {"byggtype": "Murbygg", "etasjeskiller": "Full utskifting + mineralull","M_floor": M_FULL_UTSKIFTING,       "M_wall": 450, "M_foundation": 650, "rolle": "Hoved"},
        {"byggtype": "Murbygg", "etasjeskiller": "Referanse lett",             "M_floor": round(light_reference_floor_mass_kg_m2(), 1), "M_wall": 450, "M_foundation": 650, "rolle": "Referanse"},
        {"byggtype": "Trebygg", "etasjeskiller": "Stubbloft bevart",          "M_floor": M_STUBB_BEVART,          "M_wall": 80,  "M_foundation": 350, "rolle": "Hoved"},
        {"byggtype": "Trebygg", "etasjeskiller": "Full utskifting + mineralull","M_floor": M_FULL_UTSKIFTING,       "M_wall": 80,  "M_foundation": 350, "rolle": "Hoved"},
        {"byggtype": "Trebygg", "etasjeskiller": "Referanse lett",             "M_floor": round(light_reference_floor_mass_kg_m2(), 1), "M_wall": 80,  "M_foundation": 350, "rolle": "Referanse"},
    ]
    pd.DataFrame(rows_t1).to_csv(tables_dir / "tabell1_scenario_summary.csv", index=False)

    # ── Tabell 2: main_findings_murbygg ────────────────────────────────
    sc = SC_MAIN
    mec = {
        "Kjeller":    "Kjelleråpning dominerer",
        "1. etasje":  "Ground shine dominerer (fasade)",
        "2. etasje":  "Overgang ground/roof",
        "3. etasje":  "Roof shine dominerer",
        "4. etasje":  "Roof shine, 1 skiller over",
        "5. etasje":  "Kun tak, lite følsom for M_floor",
    }
    ds = df[(df["scenario"] == sc) & df["building"].str.contains("Murbygg | stubbloft", regex=False)]
    du = df[(df["scenario"] == sc) & df["building"].str.contains("Murbygg | full utskifting", regex=False)]
    rows_t2 = []
    for lbl in FLOOR_ORDER:
        rs = ds[ds["label"] == lbl]
        ru = du[du["label"] == lbl]
        if rs.empty or ru.empty:
            continue
        rs, ru = rs.iloc[0], ru.iloc[0]
        rows_t2.append({
            "etasje":           lbl,
            "PF_stubbloft":     round(rs["PF"], 2),
            "PF_utskifting":    round(ru["PF"], 2),
            "ratio":            round(rs["PF"] / ru["PF"], 2),
            "Ground%_stubb":    round(100*rs["ground_frac"], 1),
            "Roof%_stubb":      round(100*rs["roof_frac"], 1),
            "Ground%_utsk":     round(100*ru["ground_frac"], 1),
            "Roof%_utsk":       round(100*ru["roof_frac"], 1),
            "mekanisme":        mec.get(lbl, ""),
        })
    pd.DataFrame(rows_t2).to_csv(tables_dir / "tabell2_main_findings_murbygg.csv", index=False)

    # Lag LaTeX-versjon av tabell 2
    _write_latex_table2(pd.DataFrame(rows_t2), tables_dir / "tabell2_main_findings_murbygg.tex")

    # ── Tabell 3: trend_summary (Markdown) ─────────────────────────────
    trend_md = """# Trendoppsummering — stubbloft vs. full utskifting

| Observasjon | Etasje(r) | Kommentar |
|---|---|---|
| Stubbloft bevart gir høyere PF | 1.–5. et., kjeller | Størst effekt i 2.–4. etasje |
| Størst prosentvis gevinst | 2.–3. etasje | PF-ratio 3–4× for Murbygg |
| 5. etasje lite følsom for M_floor | 5. etasje | Kun tak over, ingen skiller |
| Kjeller styres av kjelleråpning | Kjeller | PF-range stor mellom scenarioer |
| Roof shine dominerer 3.–5. et. | 3.–5. etasje | Øvre dekker viktigst |
| Ground shine dominerer 1.–2. et. | 1.–2. etasje | Fasaden viktigst her |
| Trebygg: mer ground shine | Alle | Lettere fasade slipper mer inn |
| Murbygg: øvre dekker særlig kritiske | 2.–4. etasje | Fasaden skjermer 1. etasje godt |
"""
    (tables_dir / "tabell3_trend_summary.md").write_text(trend_md, encoding="utf-8")

    # ── Tabell 4: sensitivity_summary ──────────────────────────────────
    rows_t4 = [
        {"sensitivitet": "Stubbloft-totalmasse (132–202 kg/m²)", "påvirker_mest": "2.–4. etasje", "endrer_konklusjon": "Nei", "kommentar": "Spenn ±20% i PF, ikke retningsendring"},
        {"sensitivitet": "Rehabilitert-masse (49–76 kg/m²)",     "påvirker_mest": "2.–4. etasje", "endrer_konklusjon": "Nei", "kommentar": "Liten effekt, PF nær uendret"},
        {"sensitivitet": "Kjelleråpningstype",                    "påvirker_mest": "Kjeller",       "endrer_konklusjon": "Ja for kjeller", "kommentar": "PF varierer sterkt (7–47 Murbygg)"},
        {"sensitivitet": "Ground radius (30–100 m)",              "påvirker_mest": "Alle",           "endrer_konklusjon": "Nei", "kommentar": "Marginal effekt"},
        {"sensitivitet": "Kjellerdekke-modus",                    "påvirker_mest": "Kjeller",        "endrer_konklusjon": "Nei for lett bygg", "kommentar": "Viktig for Murbygg kjeller-PF"},
        {"sensitivitet": "Fasadetype (mur vs tre)",               "påvirker_mest": "1.–2. etasje",  "endrer_konklusjon": "Nei", "kommentar": "Rollen til ground/roof endres, ikke trenden"},
    ]
    pd.DataFrame(rows_t4).to_csv(tables_dir / "tabell4_sensitivity_summary.csv", index=False)

    print(f"  ✓ Tabeller: tabell1–4 i {tables_dir}/")


def _write_latex_table2(df: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Nøkkeltall — Murbygg, Kjellervindu 40×60 cm}",
        r"\label{tab:main_findings_murbygg}",
        r"\small",
        r"\begin{tabular}{lrrrrrrrl}",
        r"\toprule",
        r"Etasje & PF stubb & PF utsk & Ratio & G\% stubb & R\% stubb & G\% utsk & R\% utsk & Mekanisme \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{row['etasje']} & {row['PF_stubbloft']:.1f} & {row['PF_utskifting']:.1f} & "
            f"{row['ratio']:.2f} & {row['Ground%_stubb']:.0f} & {row['Roof%_stubb']:.0f} & "
            f"{row['Ground%_utsk']:.0f} & {row['Roof%_utsk']:.0f} & "
            + row['mekanisme'].replace("%", r"\%") + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════
# LUFTESPALTE-SETT  ·  nye rapportklare hovedfigurer  ·  01–06
# ═══════════════════════════════════════════════════════════════════════════

_SC_LUF  = "Luftespalte 15x15 cm"
_BW_KW   = dict(alpha=0.13, color="#AED6F1", zorder=0, label="Bedwell PF 5–10")
_FLOOR_ORDER = ["Kjeller", "1. etasje", "2. etasje", "3. etasje", "4. etasje", "5. etasje"]


def _luf_data(df: pd.DataFrame, facade: str, btype: str, scenario: str = _SC_LUF) -> pd.DataFrame:
    """Henter én bygg-type for ett scenario, sortert nedenfra (Kjeller) og opp."""
    key = f"{facade}bygg | {btype}"
    sub = df[(df["scenario"] == scenario) & df["building"].str.contains(key, regex=False)].copy()
    order = {lbl: i for i, lbl in enumerate(_FLOOR_ORDER)}
    sub["_ord"] = sub["label"].map(order)
    return sub.sort_values("_ord").reset_index(drop=True)


def _floor_y(d: pd.DataFrame):
    return np.arange(len(d)), d["label"].tolist()


def _pf_floor_axis(ax: plt.Axes, all_pf: list, fontsize: int = 12) -> None:
    """Setter opp dual PF/dose x-akse (log), Bedwell-band og grid."""
    ax.axvspan(5, 10, **_BW_KW)
    setup_pf_axis_with_dose_axis(ax, all_pf, xlabel_fontsize=fontsize)
    ax.grid(True, which="both", alpha=0.22, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _legend_safe(ax: plt.Axes) -> None:
    """Legend plassert der kurvene ikke er (øvre høyre er tom for profil-plot)."""
    ax.legend(fontsize=FS["legend"], framealpha=0.92, loc="upper right",
              edgecolor="#ccc", handlelength=1.8)


# ── Felles tegne-kjerne ────────────────────────────────────────────────────

def _draw_profile(
    ax: plt.Axes,
    y: np.ndarray,
    labels: list,
    lines: list,          # [(pf_arr, color, lw, ms, marker, label), ...]
    bands: list = (),     # [(pf_min, pf_max, color), ...]
    all_pf: list = None,
) -> None:
    if all_pf is None:
        all_pf = [v for pf, *_ in lines for v in pf]
        for lo, hi, _ in bands:
            all_pf.extend(hi)

    _pf_floor_axis(ax, all_pf, fontsize=FS["label"])

    for lo, hi, col in bands:
        ax.fill_betweenx(y, lo, hi, alpha=0.20, color=col, zorder=1)

    for pf, col, lw, ms, mk, lbl in lines:
        ax.plot(pf, y, color=col, linewidth=lw, marker=mk,
                markersize=ms, label=lbl, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=FS["tick"])
    ax.set_ylabel("Etasje", fontsize=FS["label"])
    ax.tick_params(axis="x", labelsize=FS["tick"])


# ── Figur 01 / 02: Ren profil, ingen bånd ─────────────────────────────────

def _fig_luf_profile(df: pd.DataFrame, facade: str, fig_num: str,
                     main_dir: Path, dpi: int = 300) -> None:
    sc_slug = "luftespalte_15x15_cm"
    fname   = f"{fig_num}_pf_profile_{facade.lower()}bygg_{sc_slug}.png"

    ds = _luf_data(df, facade, "stubbloft bevart")
    du = _luf_data(df, facade, "full utskifting")
    y, labels = _floor_y(ds)

    fig, ax = plt.subplots(figsize=(9.0, 6.5))

    _draw_profile(
        ax, y, labels,
        lines=[
            (ds["PF"].values, C["stubb"],     2.5, 7, "o", "Stubbloft bevart"),
            (du["PF"].values, C["utskiftet"], 2.5, 7, "s", "Full utskifting + mineralull"),
        ],
    )
    _legend_safe(ax)
    fig.tight_layout()
    _save(fig, main_dir / fname, dpi=dpi)
    print(f"  ✓ {fname}")


# ── M_floor-sensitivitetsbånd (kjernes ikke kjelleråpning) ───────────────

def _compute_mass_band(
    base: object,
    facade: str,
    M_lo: float,
    M_hi: float,
    M_wall: float,
    M_found: float,
) -> tuple:
    """
    Kjører modellen med M_lo og M_hi for M_floor_kg_m2 (Luftespalte-scenariet).
    Returnerer (pf_min, pf_max) per etasje, sortert Kjeller–5.etasje.
    Båndet representerer usikkerhet i etasjeskillermassen, ikke kjelleråpning.
    """
    import warnings
    from dataclasses import replace as dc_replace
    from .calculations import calculate_rows_for_points, rows_to_dataframe
    from .geometry import default_five_points_for_building
    from .scenarios import make_scenarios

    points = default_five_points_for_building(base)
    sc_template = next(s for s in make_scenarios(base) if s.name == _SC_LUF)
    order = {lbl: i for i, lbl in enumerate(_FLOOR_ORDER)}

    pf_arrays = []
    for M in (M_lo, M_hi):
        b = dc_replace(sc_template, M_floor_kg_m2=M,
                       M_wall_kg_m2=M_wall, M_foundation_wall_kg_m2=M_found,
                       name="__tmp__")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rows = calculate_rows_for_points(_SC_LUF, "5 pt", points, b)
        df_tmp = rows_to_dataframe(rows)
        df_tmp["_ord"] = df_tmp["label"].map(order)
        df_tmp = df_tmp.sort_values("_ord").reset_index(drop=True)
        pf_arrays.append(df_tmp["PF"].values)

    lo, hi = pf_arrays[0], pf_arrays[1]
    return np.minimum(lo, hi), np.maximum(lo, hi)


# ── Figur 03 / 04 (og clean-variantene 07 / 08) ──────────────────────────

def _fig_luf_band(
    df: pd.DataFrame,
    facade: str,
    fig_num: str,
    main_dir: Path,
    dpi: int = 300,
    base: object = None,
    clean: bool = False,
) -> None:
    """
    PF-profil med sensitivitetsbånd basert på M_floor-spenn.
    clean=True: ingen tittel — klar for direkte innliming i Overleaf.
    Båndet er IKKE basert på kjelleråpningstype, men på etasjeskillermasse:
      stubbloft 132–230 kg/m²,  utskifting 49–76 kg/m².
    """
    sc_slug = "luftespalte_15x15_cm"
    suffix  = "_clean" if clean else ""
    fname   = f"{fig_num}_pf_band_{facade.lower()}bygg_{sc_slug}{suffix}.png"

    M_wall  = 450.0 if facade == "Mur" else 80.0
    M_found = 650.0 if facade == "Mur" else 350.0

    ds = _luf_data(df, facade, "stubbloft bevart")
    du = _luf_data(df, facade, "full utskifting")
    y, labels = _floor_y(ds)

    # Sensitivitetsbånd: M_floor-spenn, ikke kjelleråpningstype
    if base is not None:
        s_min, s_max = _compute_mass_band(base, facade, 132.0, 230.0, M_wall, M_found)
        u_min, u_max = _compute_mass_band(base, facade,  49.0,  76.0, M_wall, M_found)
    else:
        s_min = s_max = ds["PF"].values
        u_min = u_max = du["PF"].values

    all_pf = list(s_max) + list(u_max)
    fig, ax = plt.subplots(figsize=(9.0, 6.5))

    # Akseoppsett: Bedwell + log PF + dose% (øverst)
    ax.axvspan(5, 10, **_BW_KW)
    setup_pf_axis_with_dose_axis(ax, all_pf, xlabel_fontsize=FS["label"])
    ax.grid(True, which="both", alpha=0.22, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Sensitivitetsbånd (legges bak kurvene)
    ax.fill_betweenx(y, s_min, s_max, alpha=0.20, color=C["stubb"],
                     label="Etasjeskiller 132–230 kg/m²", zorder=1)
    ax.fill_betweenx(y, u_min, u_max, alpha=0.20, color=C["utskiftet"],
                     label="Etasjeskiller 49–76 kg/m²", zorder=1)

    # Hovedkurver
    ax.plot(ds["PF"].values, y, color=C["stubb"],     linewidth=2.5, marker="o",
            markersize=7, label="Stubbloft bevart",             zorder=3)
    ax.plot(du["PF"].values, y, color=C["utskiftet"], linewidth=2.5, marker="s",
            markersize=7, label="Full utskifting + mineralull", zorder=3)

    # Y-akse
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=FS["tick"])
    ax.set_ylabel("Etasje", fontsize=FS["label"])
    ax.tick_params(axis="x", labelsize=FS["tick"])

    # Legend: ønsket rekkefølge
    _order = {
        "Bedwell PF 5–10":                0,
        "Stubbloft bevart":                1,
        "Etasjeskiller 132–230 kg/m²":    2,
        "Full utskifting + mineralull":    3,
        "Etasjeskiller 49–76 kg/m²":      4,
    }
    handles, hlabels = ax.get_legend_handles_labels()
    pairs = sorted(zip(hlabels, handles), key=lambda t: _order.get(t[0], 99))
    lbl_sorted, hdl_sorted = zip(*pairs)
    ax.legend(hdl_sorted, lbl_sorted, fontsize=FS["legend"],
              framealpha=0.92, loc="upper right", edgecolor="#ccc")

    # Tittel kun for ikke-clean variant
    if not clean:
        ax.set_title(
            f"PF per etasje — {facade}bygg | Luftespalte 15×15 cm",
            fontsize=FS["title"], pad=10,
        )

    fig.tight_layout()
    _save(fig, main_dir / fname, dpi=dpi)
    print(f"  ✓ {fname}")


# ── Figur 05 / 06: Enkleste og reneste versjon ────────────────────────────

def _fig_luf_simple(df: pd.DataFrame, facade: str, fig_num: str,
                    main_dir: Path, dpi: int = 300) -> None:
    """Renest mulige figur — stor font, ingen ekstra elementer."""
    sc_slug = "luftespalte_15x15_cm"
    fname   = f"{fig_num}_pf_simple_{facade.lower()}bygg_{sc_slug}.png"

    ds = _luf_data(df, facade, "stubbloft bevart")
    du = _luf_data(df, facade, "full utskifting")
    y, labels = _floor_y(ds)

    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    _draw_profile(
        ax, y, labels,
        lines=[
            (ds["PF"].values, C["stubb"],     3.0, 8, "o", "Stubbloft bevart"),
            (du["PF"].values, C["utskiftet"], 3.0, 8, "s", "Full utskifting + mineralull"),
        ],
    )

    # Legend i trygt hjørne, ingen ramme
    ax.legend(fontsize=FS["legend"] + 1, frameon=False, loc="upper right",
              handlelength=1.8)

    fig.tight_layout()
    _save(fig, main_dir / fname, dpi=dpi)
    print(f"  ✓ {fname}")


# ── Koordinert kall ────────────────────────────────────────────────────────

def create_luftespalte_figures(
    df: pd.DataFrame,
    main_dir: Path,
    dpi: int = 300,
    base: object = None,
) -> None:
    """
    Genererer figurene 01–08 basert på Luftespalte 15×15 cm-scenariet.

    01–02: Ren profil (ingen bånd)
    03–04: Profil med M_floor-sensitivitetsbånd
    05–06: Enkleste versjon (stor font, ingen ramme på legend)
    07–08: Ren variant uten tittel, klar for Overleaf (med M_floor-bånd)
    """
    _fig_luf_profile(df, "Mur", "01", main_dir, dpi)
    _fig_luf_profile(df, "Tre", "02", main_dir, dpi)
    _fig_luf_band(df, "Mur", "03", main_dir, dpi, base=base, clean=False)
    _fig_luf_band(df, "Tre", "04", main_dir, dpi, base=base, clean=False)
    _fig_luf_simple(df, "Mur", "05", main_dir, dpi)
    _fig_luf_simple(df, "Tre", "06", main_dir, dpi)
    _fig_luf_band(df, "Mur", "07", main_dir, dpi, base=base, clean=True)
    _fig_luf_band(df, "Tre", "08", main_dir, dpi, base=base, clean=True)


# ── Hoved-inngangspunkt ────────────────────────────────────────────────────

def create_report_figures(
    df: pd.DataFrame,
    figures_dir: Path,
    output_dir: Path,
    export_pdf: bool = False,
    base: object = None,
) -> None:
    """
    Lag 6 rapportklare hovedfigurer (PNG, 300 dpi) og 4 tabeller.

    Figurer → figures/main/ (PNG alltid)
    PDF     → figures/main_pdf/ (kun hvis export_pdf=True)
    Tabeller → output/tables/

    Standard er export_pdf=False. PNG-filene er 300 dpi og egner seg
    direkte for import i Overleaf.
    """
    main_dir = figures_dir / "main"
    main_dir.mkdir(parents=True, exist_ok=True)

    pdf_dir: Optional[Path] = None
    if export_pdf:
        pdf_dir = figures_dir / "main_pdf"
        pdf_dir.mkdir(parents=True, exist_ok=True)

    print("\nRapportfigurer:")

    # ── Luftespalte-sett 01–08 (nye rapportklare figurer) ───────────────
    create_luftespalte_figures(df, main_dir, dpi=300, base=base)

    # ── Eldre figurer (beholdes) ─────────────────────────────────────────
    fig_pf_profile(df, "Mur", "01_old", main_dir, pdf_dir)
    fig_pf_profile(df, "Tre", "02_old", main_dir, pdf_dir)

    # ── PF-ratio ─────────────────────────────────────────────────────────
    fig_pf_ratio(df, "Mur", "03", main_dir, pdf_dir)

    # ── Ground/Roof-andeler ──────────────────────────────────────────────
    fig_ground_roof(df, "Mur", "04", main_dir, pdf_dir)

    # ── Mekanismeplot ─────────────────────────────────────────────────────
    fig_mechanism(df, "Mur", "05", main_dir, pdf_dir)

    # ── Kjelleråpnings-sensitivitet ──────────────────────────────────────
    fig_kjeller_sensitivity(df, "Mur", "06", main_dir, pdf_dir)

    # ── Tabeller ─────────────────────────────────────────────────────────
    print("\nTabeller:")
    _make_tables(df, output_dir)
