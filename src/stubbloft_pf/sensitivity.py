# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import replace
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .calculations import calculate_rows_for_points, rows_to_dataframe
from .config import Building, ResultRow
from .geometry import default_five_points_for_building
from .physics import stubbloft_floor_mass_from_clay_kg_m2
from .scenarios import M_STUBB_BEVART, M_FULL_UTSKIFTING
from .plots import finish_figure, setup_pf_axis_with_dose_axis, short_building_label
from .scenarios import make_buildings

def run_ground_radius_sensitivity(
    base: Building,
    scenario_template: Building,
    radii_m: Iterable[float] = (30.0, 60.0, 100.0),
) -> pd.DataFrame:
    # Kjører enkel sensitivitetsanalyse for ground_radius_m.

    all_rows: List[ResultRow] = []

    for radius in radii_m:
        scenario = replace(scenario_template, ground_radius_m=radius)
        buildings = make_buildings(scenario)
        points = default_five_points_for_building(scenario)
        point_name = "5 målepunkter"

        for b in buildings:
            rows = calculate_rows_for_points(
                scenario=f"{scenario.name} | R={radius:.0f} m",
                point_set_name=point_name,
                points=points,
                b=b,
            )
            all_rows.extend(rows)

    return rows_to_dataframe(all_rows)


def plot_ground_radius_sensitivity(
    sensitivity_df: pd.DataFrame,
    building_contains: str = "Murbygg",
) -> None:
    # Plotter PF som spenn over ulike ground shine-radier for valgt bygningsvariant.

    df = sensitivity_df[sensitivity_df["building"].str.contains(building_contains, regex=False)].copy()
    if df.empty:
        raise ValueError(f"Fant ingen bygningsrad som inneholder {building_contains!r}")

    # Hent radius ut av scenario-teksten "... | R=60 m".
    df["R_m"] = df["scenario"].str.extract(r"R=(\d+(?:\.\d+)?) m").astype(float)

    plt.figure(figsize=(9, 6))
    for label, sub in df.groupby("label"):
        sub = sub.sort_values("R_m")
        plt.plot(sub["R_m"], sub["PF"], marker="o", linewidth=2, label=label)

    plt.xlabel("Ground shine-radius [m]")
    plt.ylabel("Protection Factor PF [-]")
    plt.title(f"Sensitivitet for ground shine-radius\n{building_contains}")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.show()


def run_clay_thickness_sensitivity(
    base: Building,
    scenario_template: Building,
    clay_thicknesses_m: Iterable[float] = (0.06, 0.08, 0.10),
    clay_density_kg_m3: float = 1400.0,
    other_floor_mass_kg_m2: float = 90.0,
) -> pd.DataFrame:
    # Sensitivitet for stubbloftsleire 6-10 cm.

    # Dette brukes fordi tegningssnitt og litteratur ikke nødvendigvis gir ett entydig tall.
    # Resultatet bør tolkes som spenn for tungt historisk etasjeskiller.

    all_rows: List[ResultRow] = []
    points = default_five_points_for_building(base)

    for t in clay_thicknesses_m:
        M_floor = stubbloft_floor_mass_from_clay_kg_m2(
            t, clay_density_kg_m3=clay_density_kg_m3, other_floor_mass_kg_m2=other_floor_mass_kg_m2
        )
        scenario = replace(scenario_template, M_floor_kg_m2=M_floor)
        b = replace(
            scenario,
            name=f"Murbygg | stubbloft {100*t:.0f} cm | dekke {M_floor:.0f}",
            M_wall_kg_m2=450.0,
            M_foundation_wall_kg_m2=650.0,
        )
        rows = calculate_rows_for_points(
            scenario=f"{scenario_template.name} | leire {100*t:.0f} cm",
            point_set_name="5 målepunkter",
            points=points,
            b=b,
        )
        all_rows.extend(rows)

    return rows_to_dataframe(all_rows)


def plot_clay_thickness_sensitivity(clay_df: pd.DataFrame) -> None:
    # Plotter PF mot tykkelse stubbloftsleire for hver etasje.

    df = clay_df.copy()
    df["leire_cm"] = df["scenario"].str.extract(r"leire (\d+(?:\.\d+)?) cm").astype(float)

    plt.figure(figsize=(9, 6))
    for label, sub in df.groupby("label"):
        sub = sub.sort_values("leire_cm")
        plt.plot(sub["leire_cm"], sub["PF"], marker="o", linewidth=2, label=label)

    plt.xlabel("Stubbloftsleire [cm]")
    plt.ylabel("Protection Factor PF [-]")
    plt.title("Sensitivitet for tykkelse av stubbloftsleire")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.show()


def run_basement_ceiling_sensitivity(
    base: Building,
    scenario_templates: Sequence[Building],
    heavy_mass_kg_m2: float = 300.0,
) -> pd.DataFrame:
    # Sammenligner to kjellerdekkemoduser for alle scenarioer og bygningsvarianter.
    #
    # "same_as_floor": kjellerdekket bruker M_floor_kg_m2 (eksisterende modell).
    # "heavy_basement_ceiling": kjellerdekket bruker heavy_mass_kg_m2.
    #
    # Resultatet viser modellspenn for PF i kjeller og 1. etasje knyttet til
    # usikkerhet i kjellerdekkets konstruksjon.

    all_rows: List[ResultRow] = []
    points = default_five_points_for_building(base)

    modes = [
        ("same_as_floor", base.M_floor_kg_m2),
        ("heavy_basement_ceiling", heavy_mass_kg_m2),
    ]

    for mode_name, _ in modes:
        for scenario_template in scenario_templates:
            scenario = replace(
                scenario_template,
                basement_ceiling_mode=mode_name,
                M_basement_ceiling_kg_m2=heavy_mass_kg_m2,
            )
            buildings = make_buildings(scenario)
            for b in buildings:
                rows = calculate_rows_for_points(
                    scenario=f"{scenario_template.name} | kjellerdekke:{mode_name}",
                    point_set_name="5 målepunkter",
                    points=points,
                    b=b,
                )
                for r in rows:
                    all_rows.append(r)

    df = rows_to_dataframe(all_rows)

    # Trekk ut kjellerdekke-modus fra scenario-feltet for enkel filtrering.
    df["kjellerdekke_mode"] = df["scenario"].str.extract(r"kjellerdekke:(\S+)$")
    df["base_scenario"] = df["scenario"].str.replace(r" \| kjellerdekke:\S+$", "", regex=True)

    return df


def plot_basement_ceiling_sensitivity(
    df: pd.DataFrame,
    heavy_mass_kg_m2: float = 300.0,
    save_path=None,
    show: bool = False,
) -> None:
    # Gruppert søyleplott: kjeller-PF for same_as_floor vs. heavy_basement_ceiling.
    # Én subplot per scenario. X-akse: bygningstype. Y-akse: PF (log-skala).

    kjeller = df[df["label"] == "Kjeller"].copy()
    kjeller["building_short"] = kjeller["building"].apply(short_building_label)

    scenarios = list(dict.fromkeys(kjeller["base_scenario"].tolist()))
    n = len(scenarios)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5.5), sharey=False)
    if n == 1:
        axes = [axes]

    mode_style = {
        "same_as_floor":           ("steelblue",  "Same as floor (M_floor)"),
        "heavy_basement_ceiling":  ("darkorange", f"Tungt kjellerdekke ({heavy_mass_kg_m2:.0f} kg/m²)"),
    }
    modes = list(mode_style.keys())
    bar_width = 0.35

    building_order = [
        "Mur | stubbloft", "Mur | utskiftet", "Mur | lett ref.",
        "Tre | stubbloft", "Tre | utskiftet", "Tre | lett ref.",
    ]

    for ax, scenario in zip(axes, scenarios):
        sub = kjeller[kjeller["base_scenario"] == scenario]
        x = np.arange(len(building_order))

        for mi, mode in enumerate(modes):
            color, label = mode_style[mode]
            mode_sub = sub[sub["kjellerdekke_mode"] == mode]
            pf_vals = []
            for bname in building_order:
                row = mode_sub[mode_sub["building_short"] == bname]
                pf_vals.append(float(row["PF"].values[0]) if len(row) > 0 else np.nan)

            offset = (mi - 0.5) * bar_width
            ax.bar(x + offset, pf_vals, bar_width, label=label, color=color, alpha=0.85)

        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(
            ["Mur\nstubb", "Mur\nutskiftet", "Mur\nlett ref.",
             "Tre\nstubb", "Tre\nutskiftet", "Tre\nlett ref."],
            fontsize=9,
        )
        ax.set_ylabel("Kjeller-PF [-]")
        ax.set_title(scenario, fontsize=11, weight="bold")
        ax.grid(axis="y", alpha=0.3, which="both")
        ax.legend(fontsize=8)

    fig.suptitle(
        f"Sensitivitet: kjellerdekke\nKjeller-PF — same as floor vs. tungt ({heavy_mass_kg_m2:.0f} kg/m²)",
        fontsize=13,
        weight="bold",
    )
    fig.tight_layout()
    finish_figure(fig, save_path=save_path, show=show)


def run_floor_mass_sensitivity(
    base: Building,
    scenario_templates: Sequence[Building],
    stubb_masses_kg_m2: Tuple[float, ...] = (132.0, M_STUBB_BEVART, 202.0),
    rehab_masses_kg_m2: Tuple[float, ...] = (49.0, M_FULL_UTSKIFTING, 76.0),
) -> pd.DataFrame:
    # Sensitivitetsanalyse for total etasjeskillermasse.
    #
    # Tester to spenn for begge fasadetyper (mur og tre):
    #
    # Stubbloft-spenn  (132 – 145 – 202 kg/m²):
    #   132 = tegningsbasert lav (8 cm leire, c/c 600 mm)
    #   145 = representativ midtverdi (standard for make_buildings)
    #   202 = litteraturbasert/stor oppbygning (stubbloft_floor_mass_from_clay_kg_m2(0.08))
    #
    # Rehabilitert-spenn (49 – 62 – 76 kg/m²):
    #   49  = enkel oppbygning (gulv + himling lett)
    #   62  = standard midtverdi (make_buildings)
    #   76  = tung oppbygning

    all_rows: List[ResultRow] = []
    points = default_five_points_for_building(base)

    for scenario_template in scenario_templates:
        for M_floor in stubb_masses_kg_m2:
            for M_wall, M_found, fasade in [(450.0, 650.0, "Mur"), (80.0, 350.0, "Tre")]:
                b = replace(
                    scenario_template,
                    name=f"{fasade}bygg | stubbloft {M_floor:.0f} kg/m²",
                    M_floor_kg_m2=M_floor,
                    M_wall_kg_m2=M_wall,
                    M_foundation_wall_kg_m2=M_found,
                )
                rows = calculate_rows_for_points(
                    scenario=f"{scenario_template.name} | stubb {M_floor:.0f}",
                    point_set_name="5 målepunkter",
                    points=points,
                    b=b,
                )
                all_rows.extend(rows)

        for M_floor in rehab_masses_kg_m2:
            for M_wall, M_found, fasade in [(450.0, 650.0, "Mur"), (80.0, 350.0, "Tre")]:
                b = replace(
                    scenario_template,
                    name=f"{fasade}bygg | utskiftet {M_floor:.0f} kg/m²",
                    M_floor_kg_m2=M_floor,
                    M_wall_kg_m2=M_wall,
                    M_foundation_wall_kg_m2=M_found,
                )
                rows = calculate_rows_for_points(
                    scenario=f"{scenario_template.name} | rehab {M_floor:.0f}",
                    point_set_name="5 målepunkter",
                    points=points,
                    b=b,
                )
                all_rows.extend(rows)

    return rows_to_dataframe(all_rows)


# ─────────────────────────────────────────────────────────────────────────────
# Etasjeskiller-prioriteringsanalyse: per-dekke sensitivitet
# ─────────────────────────────────────────────────────────────────────────────

_DECK_NAMES = {0: "K–1", 1: "1–2", 2: "2–3", 3: "3–4", 4: "4–5"}
_DECK_Z     = {0: "z=0 m", 1: "z=3 m", 2: "z=6 m", 3: "z=9 m", 4: "z=12 m"}

_FACADE_CONFIGS_DECK = [
    ("Murbygg", 450.0, 650.0),
    ("Trebygg",  80.0, 350.0),
]


def run_floor_deck_importance(
    base: Building,
    scenario_template: Building,
    stubb_mass: float = M_STUBB_BEVART,
    utsk_mass:  float = M_FULL_UTSKIFTING,
) -> pd.DataFrame:
    # Analyserer PF-bidraget fra individuelle etasjeskiller for Murbygg og Trebygg.
    #
    # For hvert etasjeskiller k (0 = kjellerdekke … floors-1 = toppetasjedekke):
    #   "remove_one":   start med alle dekker = stubbloft, bytt dekke k til utskifting.
    #   "preserve_one": start med alle dekker = utskifting, bevar dekke k som stubbloft.
    # Baselines: "all_stubb" og "all_utsk".

    points  = default_five_points_for_building(scenario_template)
    n_decks = base.floors
    records: List = []

    def _make(facade_lbl, M_wall, M_fwall, masses):
        return replace(
            scenario_template,
            name=facade_lbl,
            M_floor_kg_m2=stubb_mass,
            M_wall_kg_m2=M_wall,
            M_foundation_wall_kg_m2=M_fwall,
            floor_masses_kg_m2=tuple(masses),
        )

    def _record(rows, facade, analysis, deck_k):
        for r in rows:
            records.append({
                "facade":      facade,
                "analysis":    analysis,
                "deck_k":      deck_k,
                "deck_label":  _DECK_NAMES.get(deck_k, str(deck_k)),
                "deck_z":      _DECK_Z.get(deck_k, ""),
                "floor_index": r.floor,
                "floor_label": r.label,
                "z_m":         r.z_m,
                "PF":          r.PF,
                "Dg_in":       r.Dg_in,
                "Dr_in":       r.Dr_in,
                "Dg_ref":      r.Dg_ref,
                "Dr_ref":      r.Dr_ref,
                "ground_pct":  100.0 * r.ground_frac,
                "roof_pct":    100.0 * r.roof_frac,
            })

    sn = scenario_template.name

    for facade_lbl, M_wall, M_fwall in _FACADE_CONFIGS_DECK:
        _record(
            calculate_rows_for_points(sn, "5 pts", points,
                _make(facade_lbl, M_wall, M_fwall, [stubb_mass] * n_decks)),
            facade_lbl, "all_stubb", -1,
        )
        _record(
            calculate_rows_for_points(sn, "5 pts", points,
                _make(facade_lbl, M_wall, M_fwall, [utsk_mass] * n_decks)),
            facade_lbl, "all_utsk", -1,
        )
        for k in range(n_decks):
            masses = [stubb_mass] * n_decks
            masses[k] = utsk_mass
            _record(
                calculate_rows_for_points(sn, "5 pts", points,
                    _make(facade_lbl, M_wall, M_fwall, masses)),
                facade_lbl, "remove_one", k,
            )
        for k in range(n_decks):
            masses = [utsk_mass] * n_decks
            masses[k] = stubb_mass
            _record(
                calculate_rows_for_points(sn, "5 pts", points,
                    _make(facade_lbl, M_wall, M_fwall, masses)),
                facade_lbl, "preserve_one", k,
            )

    return pd.DataFrame(records)


def _deck_pivot(
    df: pd.DataFrame,
    facade: str,
    analysis: str,
    focus_labels: List[str],
    baseline_analysis: str,
) -> Tuple[np.ndarray, np.ndarray]:
    base_sub = df[(df.facade == facade) & (df.analysis == baseline_analysis)]
    n_decks  = int(df["deck_k"].max()) + 1
    matrix   = np.zeros((len(focus_labels), n_decks))
    baseline = np.full(len(focus_labels), np.nan)

    for i, fl in enumerate(focus_labels):
        row = base_sub[base_sub.floor_label == fl]
        if len(row):
            baseline[i] = float(row["PF"].values[0])
        for k in range(n_decks):
            sub = df[(df.facade == facade) & (df.analysis == analysis) &
                     (df.deck_k == k) & (df.floor_label == fl)]
            if len(sub):
                matrix[i, k] = float(sub["PF"].values[0])

    return matrix, baseline


def plot_deck_importance_heatmap(
    df: pd.DataFrame,
    facade_label: str,
    save_path=None,
    show: bool = False,
) -> None:
    # Heatmap: prosentvis PF-tap (venstre) og PF-gevinst (høyre) per etasje × dekke.

    focus   = ["Kjeller", "2. etasje", "3. etasje", "4. etasje"]
    n_decks = int(df["deck_k"].max()) + 1
    deck_xlbls = [f"{_DECK_NAMES[k]}\n({_DECK_Z[k]})" for k in range(n_decks)]

    matrix_r,  pf_stubb = _deck_pivot(df, facade_label, "remove_one",   focus, "all_stubb")
    matrix_p,  pf_utsk  = _deck_pivot(df, facade_label, "preserve_one", focus, "all_utsk")

    with np.errstate(divide="ignore", invalid="ignore"):
        tap_pct  = 100.0 * (pf_stubb[:, None] - matrix_r) / np.where(pf_stubb[:, None] > 0, pf_stubb[:, None], np.nan)
        gain_pct = 100.0 * (matrix_p - pf_utsk[:, None])  / np.where(pf_utsk[:, None]  > 0, pf_utsk[:, None],  np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, mat, cmap, cb_lbl, title in zip(
        axes,
        [tap_pct, gain_pct],
        ["Reds", "Greens"],
        ["PF-tap [%]", "PF-gevinst [%]"],
        [
            "PF-tap [%] ved fjerning av ett dekke\n(start: alle stubbloft bevart)",
            "PF-gevinst [%] ved bevaring av ett dekke\n(start: alle full utskifting)",
        ],
    ):
        vmax = float(np.nanmax(np.abs(mat))) * 1.05 if np.any(np.isfinite(mat)) else 1.0
        im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax, label=cb_lbl, shrink=0.85)
        ax.set_xticks(range(n_decks))
        ax.set_xticklabels(deck_xlbls, fontsize=10)
        ax.set_yticks(range(len(focus)))
        ax.set_yticklabels(focus, fontsize=11)
        ax.set_xlabel("Etasjeskiller")
        ax.set_title(f"{facade_label}\n{title}", fontsize=11, weight="bold")

        for i in range(len(focus)):
            for j in range(n_decks):
                val = mat[i, j]
                if np.isfinite(val):
                    txt_color = "white" if val > 0.65 * vmax else "black"
                    ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                            fontsize=9, color=txt_color)

    fig.tight_layout()
    finish_figure(fig, save_path=save_path, show=show)


def plot_deck_importance_lines(
    df: pd.DataFrame,
    facade_label: str,
    save_path=None,
    show: bool = False,
) -> None:
    # Linjeplott: PF-tap (venstre) og PF-gevinst (høyre) per dekke, linjer per etasje.

    focus   = ["Kjeller", "2. etasje", "3. etasje", "4. etasje"]
    colors  = ["#9467BD", "#FF7F0E", "#2CA02C", "#D62728"]
    n_decks = int(df["deck_k"].max()) + 1
    x = np.arange(n_decks)
    deck_xlbls = [f"{_DECK_NAMES[k]}\n({_DECK_Z[k]})" for k in range(n_decks)]

    matrix_r,  pf_stubb = _deck_pivot(df, facade_label, "remove_one",   focus, "all_stubb")
    matrix_p,  pf_utsk  = _deck_pivot(df, facade_label, "preserve_one", focus, "all_utsk")

    with np.errstate(divide="ignore", invalid="ignore"):
        tap_pct  = 100.0 * (pf_stubb[:, None] - matrix_r) / np.where(pf_stubb[:, None] > 0, pf_stubb[:, None], np.nan)
        gain_pct = 100.0 * (matrix_p - pf_utsk[:, None])  / np.where(pf_utsk[:, None]  > 0, pf_utsk[:, None],  np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, mat, ylabel, title in zip(
        axes,
        [tap_pct, gain_pct],
        ["PF-tap [%]", "PF-gevinst [%]"],
        [
            "PF-tap ved fjerning av ett dekke\n(start: alle stubbloft bevart)",
            "PF-gevinst ved bevaring av ett dekke\n(start: alle full utskifting)",
        ],
    ):
        for i, (fl, color) in enumerate(zip(focus, colors)):
            ax.plot(x, mat[i], marker="o", linewidth=2.0, color=color, label=fl)
            for xi, val in enumerate(mat[i]):
                if np.isfinite(val) and val > 0.3:
                    ax.annotate(f"{val:.1f}%", xy=(xi, val),
                                xytext=(0, 6), textcoords="offset points",
                                ha="center", fontsize=7.5, color=color)

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(deck_xlbls, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Etasjeskiller")
        ax.set_title(f"{facade_label} — {title}", fontsize=11, weight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc="upper right")

    fig.tight_layout()
    finish_figure(fig, save_path=save_path, show=show)


# ─────────────────────────────────────────────────────────────────────────────
# Mekanismeanalyse: ground radius vs. PF-ratio og ground/roof-regime
# ─────────────────────────────────────────────────────────────────────────────

def _extract_radius(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["R_m"] = df["scenario"].str.extract(r"R=(\d+(?:\.\d+)?) m").astype(float)
    return df


def _merge_stubb_utsk(df: pd.DataFrame, facade: str) -> pd.DataFrame:
    # Fletter stubbloft- og utskiftingsrader på (R_m, label) for én fasadetype.
    fl = facade.lower()
    s = df[df["building"].str.lower().str.contains(fl) &
           df["building"].str.lower().str.contains("stubbloft") &
           ~df["building"].str.lower().str.contains("referanse")].copy()
    u = df[df["building"].str.lower().str.contains(fl) &
           df["building"].str.lower().str.contains("utskifting")].copy()
    return s.merge(u, on=["R_m", "label", "z_m"], suffixes=("_s", "_u"))


def plot_pf_ratio_by_radius(
    df: pd.DataFrame,
    facade_label: str,
    save_path=None,
    show: bool = False,
) -> None:
    # PF-ratio (stubbloft / utskifting) per etasje for ulike ground radius.
    # Kjeller ekskluderes (dominert av åpningstype, ikke etasjeskillermasse).

    df = _extract_radius(df)
    merged = _merge_stubb_utsk(df, facade_label)
    merged = merged[merged["label"] != "Kjeller"]
    merged["ratio"] = merged["PF_s"] / merged["PF_u"].clip(lower=1e-9)

    floor_order = (
        merged[["label", "z_m"]].drop_duplicates("label")
        .sort_values("z_m")["label"].tolist()
    )
    y_map = {lbl: i for i, lbl in enumerate(floor_order)}

    radii   = sorted(merged["R_m"].unique())
    palette = plt.cm.plasma(np.linspace(0.1, 0.85, len(radii)))

    fig, ax = plt.subplots(figsize=(8, 6))

    for r, color in zip(radii, palette):
        sub = (merged[merged["R_m"] == r]
               .set_index("label").reindex(floor_order).reset_index())
        y_vals = [y_map[lbl] for lbl in sub["label"]]
        ratio_vals = sub["ratio"].values
        ax.plot(ratio_vals, y_vals, marker="o", linewidth=2.0, color=color,
                label=f"R = {r:.0f} m")
        if abs(r - 60.0) < 1:
            for yv, rv in zip(y_vals, ratio_vals):
                if np.isfinite(rv):
                    ax.text(rv + 0.06, yv + 0.07, f"{rv:.1f}×",
                            fontsize=8, color=color, va="bottom")

    ax.axvline(1.0, color="black", linewidth=1.5, linestyle="--", alpha=0.7,
               label="Ratio = 1 (ingen forskjell)")

    max_ratio = merged["ratio"].max()
    ax.set_xlim(0, max(max_ratio * 1.15, 2.0))
    ax.set_yticks(range(len(floor_order)))
    ax.set_yticklabels(floor_order, fontsize=11)
    ax.set_ylabel("Etasje")
    ax.set_xlabel("PF_stubbloft / PF_utskifting [-]")
    ax.set_title(
        f"PF-ratio per etasje — effekt av ground radius\n{facade_label} | 1.–5. etasje",
        fontsize=12, weight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout()
    finish_figure(fig, save_path=save_path, show=show)


def plot_ground_pct_by_radius(
    df: pd.DataFrame,
    facade_label: str,
    save_path=None,
    show: bool = False,
) -> None:
    # Ground%-andel per etasje for stubbloft og utskifting ved ulike ground radius.
    # Vertikal linje ved 50 % markerer overgangen mellom ground- og roof-dominert regime.

    df = _extract_radius(df)
    merged = _merge_stubb_utsk(df, facade_label)
    merged = merged[merged["label"] != "Kjeller"]
    merged["gnd_pct_s"] = 100.0 * merged["ground_frac_s"]
    merged["gnd_pct_u"] = 100.0 * merged["ground_frac_u"]

    floor_order = (
        merged[["label", "z_m"]].drop_duplicates("label")
        .sort_values("z_m")["label"].tolist()
    )
    y_map = {lbl: i for i, lbl in enumerate(floor_order)}

    radii   = sorted(merged["R_m"].unique())
    palette = plt.cm.plasma(np.linspace(0.1, 0.85, len(radii)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)

    for ax, col, concept in zip(
        axes,
        ["gnd_pct_s", "gnd_pct_u"],
        ["Stubbloft bevart (145 kg/m²)", "Full utskifting + mineralull (62 kg/m²)"],
    ):
        for r, color in zip(radii, palette):
            sub = (merged[merged["R_m"] == r]
                   .set_index("label").reindex(floor_order).reset_index())
            y_vals = [y_map[lbl] for lbl in sub["label"]]
            ax.plot(sub[col].values, y_vals, marker="o", linewidth=2.0, color=color,
                    label=f"R = {r:.0f} m")

        ax.axvline(50, color="gray", linewidth=1.0, linestyle="--", alpha=0.6,
                   label="50 % grense")
        ax.set_xlim(0, 105)
        ax.set_xlabel("Ground shine-andel [%]")
        ax.set_yticks(range(len(floor_order)))
        ax.set_yticklabels(floor_order, fontsize=11)
        ax.set_title(concept, fontsize=11, weight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="upper right")

    fig.suptitle(
        f"Ground%-andel per etasje ved ulike ground radius\n{facade_label} | 1.–5. etasje",
        fontsize=12, weight="bold",
    )
    fig.tight_layout()
    finish_figure(fig, save_path=save_path, show=show)


def plot_roof_model_comparison(
    df: pd.DataFrame,
    facade_label: str,
    save_path=None,
    show: bool = False,
) -> None:
    # PF per etasje for flat tak vs. saltak, for stubbloft bevart og full utskifting.
    # Farge = etasjeskillerkonsept, linjetype = takmodell.

    sub = df[
        df["building"].str.contains(facade_label, case=False, regex=False)
        & ~df["building"].str.contains("referanse", case=False, regex=False)
    ].copy()

    if sub.empty:
        return

    sub["model_label"] = sub["scenario"].str.split("|").str[-1].str.strip()
    sub["concept"] = np.where(
        sub["building"].str.contains("stubbloft", case=False),
        "stubbloft",
        "utskifting",
    )

    floor_labels = (
        sub[["label", "z_m"]]
        .drop_duplicates("label")
        .sort_values("z_m")["label"]
        .tolist()
    )
    y_map = {label: i for i, label in enumerate(floor_labels)}
    y = np.arange(len(floor_labels))

    concept_colors   = {"stubbloft": "#D62728", "utskifting": "#1F77B4"}
    model_linestyles = {"Flatt tak": "solid", "Saltak 35°": "dashed", "Saltak 45°": "dotted", "Saltak 50°": (0, (3, 1, 1, 1))}
    model_lw         = {"Flatt tak": 2.5,    "Saltak 35°": 2.0,      "Saltak 45°": 1.8,      "Saltak 50°": 1.6}
    concept_display  = {"stubbloft": "Stubbloft bevart", "utskifting": "Full utskifting"}

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    all_pf = []

    for (concept, model_label), grp in sub.groupby(["concept", "model_label"]):
        grp = grp.sort_values("z_m").reset_index(drop=True)
        pf_vals = grp["PF"].values
        y_vals  = [y_map[lbl] for lbl in grp["label"]]

        color = concept_colors.get(concept, "gray")
        ls    = model_linestyles.get(model_label, "solid")
        lw    = model_lw.get(model_label, 2.0)
        label = f"{concept_display.get(concept, concept)} | {model_label}"

        ax.plot(pf_vals, y_vals, marker="o", linewidth=lw, linestyle=ls, color=color, label=label)
        all_pf.extend(pf_vals.tolist())

    setup_pf_axis_with_dose_axis(ax, all_pf, xlabel_fontsize=11)

    ax.set_yticks(y)
    ax.set_yticklabels(floor_labels, fontsize=11)
    ax.set_ylabel("Etasje")
    ax.set_title(
        f"Takmodellsensitivitet: flat vs. saltak\n{facade_label}",
        fontsize=15,
        weight="bold",
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

    fig.subplots_adjust(left=0.14, right=0.98, top=0.84, bottom=0.13)
    finish_figure(fig, save_path=save_path, show=show)


def run_roof_model_comparison(
    base: Building,
    scenario_template: Building,
    roof_models: Sequence[Tuple[str, float]] = (
        ("flat", 0.0),
        ("sloped", 35.0),
        ("sloped", 45.0),
        ("sloped", 50.0),
    ),
) -> pd.DataFrame:
    # Sammenligner flatt tak med idealisert saltak.
    # Standardmodellen bruker sloped 45° (historisk tegningsgrunnlag).
    # Flat tak beholdes som referanse/sensitivitet.

    all_rows: List[ResultRow] = []
    points = default_five_points_for_building(base)

    for model, pitch in roof_models:
        scenario = replace(scenario_template, roof_model=model, roof_pitch_deg=pitch)
        buildings = make_buildings(scenario)
        model_name = "Flatt tak" if model == "flat" else f"Saltak {pitch:.0f}°"
        for b in buildings:
            rows = calculate_rows_for_points(
                scenario=f"{scenario_template.name} | {model_name}",
                point_set_name="5 målepunkter",
                points=points,
                b=b,
            )
            all_rows.extend(rows)

    return rows_to_dataframe(all_rows)


# ─────────────────────────────────────────────────────────────────────────────
# Dekke-kombinasjonsanalyse: alle 2^5 = 32 kombinasjoner
# ─────────────────────────────────────────────────────────────────────────────

_COMB_FOCUS_FLOORS = ["Kjeller", "2. etasje", "3. etasje", "4. etasje"]
_COMB_FOCUS_COLS   = ["Kjeller", "2F", "3F", "4F"]
_COMB_FOCUS_MAP    = dict(zip(_COMB_FOCUS_FLOORS, _COMB_FOCUS_COLS))
_COMB_DECK_SHORT   = ["K–1", "1–2", "2–3", "3–4", "4–5"]
_COMB_DECK_Z       = ["z=0m", "z=3m", "z=6m", "z=9m", "z=12m"]


def run_deck_combination_analysis(
    base: Building,
    scenario_template: Building,
    stubb_mass: float = M_STUBB_BEVART,
    utsk_mass:  float = M_FULL_UTSKIFTING,
) -> pd.DataFrame:
    # Alle 2^5 = 32 kombinasjoner av stubbloft/utskiftet for 5 dekker.
    # Rapporterer PF, LF og sammensatte mål for fokus-etasjene.
    # Kombinasjons-ID er en 5-bits bitmask: bit k=1 betyr dekke k er stubbloft.

    import itertools as _it
    points  = default_five_points_for_building(scenario_template)
    n_decks = base.floors
    sn      = scenario_template.name
    records: List = []

    def _make(fl, Mw, Mfw, masses):
        return replace(
            scenario_template, name=fl,
            M_floor_kg_m2=stubb_mass, M_wall_kg_m2=Mw, M_foundation_wall_kg_m2=Mfw,
            floor_masses_kg_m2=tuple(masses),
        )

    for fl_lbl, Mw, Mfw in _FACADE_CONFIGS_DECK:
        for combo in _it.product([0, 1], repeat=n_decks):
            masses   = [stubb_mass if c else utsk_mass for c in combo]
            n_stubb  = sum(combo)
            combo_id = int("".join(str(c) for c in combo), 2)

            rows_out   = calculate_rows_for_points(sn, "5 pts", points, _make(fl_lbl, Mw, Mfw, masses))
            row_by_lbl = {r.label: r for r in rows_out}

            rec: dict = {
                "facade":      fl_lbl,
                "combo_id":    combo_id,
                "n_stubb":     n_stubb,
                "combo_label": "".join("S" if c else "-" for c in combo),
            }
            for k, c in enumerate(combo):
                rec[f"d{k}"] = c   # 1 = stubbloft, 0 = utskifting

            pf_list, lf_list = [], []
            for fl, col in _COMB_FOCUS_MAP.items():
                r  = row_by_lbl.get(fl)
                pf = r.PF if r else np.nan
                lf = r.LF if r else np.nan
                rec[f"PF_{col}"] = pf
                rec[f"LF_{col}"] = lf
                pf_list.append(pf)
                lf_list.append(lf)

            pf_arr = np.array([v for v in pf_list if np.isfinite(v)])
            lf_arr = np.array([v for v in lf_list if np.isfinite(v)])
            rec["mean_LF_focus"] = float(np.mean(lf_arr)) if len(lf_arr) else np.nan
            rec["min_PF_focus"]  = float(np.min(pf_arr))  if len(pf_arr) else np.nan
            rec["mean_PF_focus"] = float(np.mean(pf_arr)) if len(pf_arr) else np.nan
            records.append(rec)

    return pd.DataFrame(records)


def _shapley_values(sub: pd.DataFrame, value_col: str, n: int = 5) -> np.ndarray:
    # Shapley-verdier: gjennomsnittlig marginalbidrag per dekke over alle subset.
    # Positiv verdi = dekket øker value_col (høyere = bedre antatt).

    import itertools as _it, math as _m
    lookup = {
        tuple(int(row[f"d{k}"]) for k in range(n)): float(row[value_col])
        for _, row in sub.iterrows()
    }
    phi = np.zeros(n)
    for k in range(n):
        rest = [i for i in range(n) if i != k]
        for r in range(n):
            w = _m.factorial(r) * _m.factorial(n - r - 1) / _m.factorial(n)
            for S in _it.combinations(rest, r):
                base_c = [0] * n
                for i in S:
                    base_c[i] = 1
                with_k = base_c.copy()
                with_k[k] = 1
                v0 = lookup.get(tuple(base_c), np.nan)
                v1 = lookup.get(tuple(with_k),    np.nan)
                if np.isfinite(v0) and np.isfinite(v1):
                    phi[k] += w * (v1 - v0)
    return phi


def plot_deck_combo_heatmap(
    df: pd.DataFrame,
    facade_label: str,
    save_path=None,
    show: bool = False,
) -> None:
    # Alle 32 kombinasjoner sortert etter (n_stubb, mean_LF_focus).
    # Venstre: binær dekkekonfigurasjon (grønn=stubb, rød=utsk).
    # Høyre: LF-verdier per fokus-etasje (lavere = bedre).

    sub = (df[df.facade == facade_label]
           .sort_values(["n_stubb", "mean_LF_focus"])
           .reset_index(drop=True))
    n   = len(sub)

    deck_mat = sub[[f"d{k}" for k in range(5)]].values.astype(float)
    lf_mat   = sub[[f"LF_{c}" for c in _COMB_FOCUS_COLS]].values.astype(float)

    fig = plt.figure(figsize=(13, max(8, 0.38 * n + 2.0)))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.4], wspace=0.04)
    ax_d  = fig.add_subplot(gs[0, 0])
    ax_lf = fig.add_subplot(gs[0, 1], sharey=ax_d)

    ax_d.imshow(deck_mat, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax_d.set_xticks(range(5))
    ax_d.set_xticklabels(
        [f"{s}\n({z})" for s, z in zip(_COMB_DECK_SHORT, _COMB_DECK_Z)], fontsize=8,
    )
    ax_d.set_yticks(range(n))
    ax_d.set_yticklabels(
        [f"n={int(r.n_stubb)}  {r.combo_label}" for _, r in sub.iterrows()],
        fontsize=7.5, family="monospace",
    )
    ax_d.set_title("Konfigurasjon\n(grønn=stubb, rød=utsk)", fontsize=9, weight="bold")
    for i in range(n):
        for j in range(5):
            ax_d.text(j, i, "S" if deck_mat[i, j] else "·",
                      ha="center", va="center", fontsize=7.5, color="black")

    vmin_lf, vmax_lf = float(np.nanmin(lf_mat)), float(np.nanmax(lf_mat))
    im = ax_lf.imshow(lf_mat, cmap="YlOrRd", aspect="auto", vmin=vmin_lf, vmax=vmax_lf)
    ax_lf.set_xticks(range(4))
    ax_lf.set_xticklabels(_COMB_FOCUS_FLOORS, fontsize=9, rotation=20, ha="right")
    ax_lf.set_yticks([])
    ax_lf.set_title("LF = 1/PF per fokus-etasje\n(lavere = bedre)", fontsize=9, weight="bold")
    plt.colorbar(im, ax=ax_lf, label="LF", shrink=0.85, pad=0.02)
    for i in range(n):
        for j in range(4):
            v = lf_mat[i, j]
            if np.isfinite(v):
                ax_lf.text(j, i, f"{v:.3f}", ha="center", va="center",
                           fontsize=7, color="white" if v > 0.65 * vmax_lf else "black")

    fig.suptitle(
        f"Alle 32 kombinasjoner — {facade_label}\n"
        "Sortert: antall stubb-dekker (primær), mean_LF_focus (sekundær)",
        fontsize=11, weight="bold",
    )
    fig.tight_layout()
    finish_figure(fig, save_path=save_path, show=show)


def plot_best_combo_by_n(
    df: pd.DataFrame,
    facade_label: str,
    save_path=None,
    show: bool = False,
) -> None:
    # Beste og verste kombinasjon per antall stubb-dekker (N = 0..5).
    # Venstre: mean_LF_focus (lavere = bedre).
    # Høyre: min_PF_focus (høyere = bedre).

    sub = df[df.facade == facade_label].copy()
    ns  = sorted(sub["n_stubb"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, col, ylabel, ascending, title in zip(
        axes,
        ["mean_LF_focus",                       "min_PF_focus"],
        ["Gjennomsnittlig LF for fokus-etasjer", "Min PF blant fokus-etasjer"],
        [True,                                    False],
        ["mean_LF_focus (lavere = bedre)",         "min_PF_focus (høyere = bedre)"],
    ):
        best_v, worst_v, best_lbls = [], [], []
        for ns_ in ns:
            grp = sub[sub["n_stubb"] == ns_].sort_values(col, ascending=ascending)
            best_v.append(float(grp.iloc[0][col]))
            worst_v.append(float(grp.iloc[-1][col]))
            best_lbls.append(grp.iloc[0]["combo_label"])

        ax.fill_between(ns, best_v, worst_v, alpha=0.15, color="steelblue",
                        label="Spenn (beste–verste)")
        ax.plot(ns, best_v,  "o-",  color="steelblue", linewidth=2.5, label="Beste kombinasjon")
        ax.plot(ns, worst_v, "s--", color="gray",      linewidth=1.5, alpha=0.7, label="Verste kombinasjon")
        for ns_, bv, lbl in zip(ns, best_v, best_lbls):
            ax.annotate(lbl, xy=(ns_, bv), xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=8.5, color="steelblue", family="monospace")

        ax.set_xticks(ns)
        ax.set_xlabel("Antall stubbloft-dekker bevart (N)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{facade_label} — {title}", fontsize=11, weight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.tight_layout()
    finish_figure(fig, save_path=save_path, show=show)


def plot_deck_pair_heatmap(
    df: pd.DataFrame,
    facade_label: str,
    mode: str = "preserve",
    save_path=None,
    show: bool = False,
) -> None:
    # Symmetrisk 5×5-matrise for par-kombinasjoner.
    # mode="preserve": N=2 stubb — hvilken 2-dekke-kombinasjon gir lavest mean_LF?
    # mode="remove":   2 dekker fjernet fra all-stubb — hvilken par-fjerning gir størst tap?

    sub = df[df.facade == facade_label].copy()
    n   = 5
    mat = np.full((n, n), np.nan)

    if mode == "preserve":
        pool = sub[sub["n_stubb"] == 2]
        for _, row in pool.iterrows():
            stubb = [k for k in range(n) if int(row[f"d{k}"]) == 1]
            if len(stubb) == 2:
                i, j = stubb
                mat[i, j] = mat[j, i] = float(row["mean_LF_focus"])
        cmap     = "YlOrRd"
        val_lbl  = "mean_LF_focus"
        title    = "mean_LF_focus — lavest = beste 2-dekke-kombinasjon\n(N=2 dekker bevart som stubbloft)"
        fmt_fn   = lambda v: f"{v:.4f}"
    else:
        all_v = float(sub[sub["n_stubb"] == 5]["mean_LF_focus"].values[0])
        pool  = sub[sub["n_stubb"] == 3]
        for _, row in pool.iterrows():
            utsk = [k for k in range(n) if int(row[f"d{k}"]) == 0]
            if len(utsk) == 2:
                i, j = utsk
                mat[i, j] = mat[j, i] = float(row["mean_LF_focus"]) - all_v
        cmap     = "Reds"
        val_lbl  = "Δ mean_LF (tap)"
        title    = "Økning i mean_LF — størst = verst å miste\n(2 dekker fjernet fra all-stubbloft)"
        fmt_fn   = lambda v: f"+{v:.4f}"

    tick_lbls = [f"{s}\n({z})" for s, z in zip(_COMB_DECK_SHORT, _COMB_DECK_Z)]
    vmax_val  = float(np.nanmax(mat))

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(mat, cmap=cmap, aspect="auto",
                   vmin=float(np.nanmin(mat)), vmax=vmax_val)
    plt.colorbar(im, ax=ax, label=val_lbl, shrink=0.88)
    ax.set_xticks(range(n))
    ax.set_xticklabels(tick_lbls, fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels(tick_lbls, fontsize=10)
    ax.set_title(f"{facade_label}\n{title}", fontsize=10, weight="bold")

    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            if np.isfinite(v):
                txt_color = "white" if v > 0.65 * vmax_val else "black"
                ax.text(j, i, fmt_fn(v), ha="center", va="center",
                        fontsize=9, color=txt_color)

    fig.tight_layout()
    finish_figure(fig, save_path=save_path, show=show)


def plot_deck_shapley(
    df: pd.DataFrame,
    facade_label: str,
    save_path=None,
    show: bool = False,
) -> None:
    # Shapley-verdier: gjennomsnittlig marginalbidrag per dekke over alle 32 kombinasjoner.
    # Positiv φ = dekket bidrar positivt til beskyttelse (øker PF, reduserer LF).

    sub = df[df.facade == facade_label].copy()
    sub["_neg_lf"] = -sub["mean_LF_focus"]   # høyere = bedre for Shapley

    phi_pf = _shapley_values(sub, "min_PF_focus")
    phi_lf = _shapley_values(sub, "_neg_lf")

    x      = np.arange(5)
    xlbls  = [f"{s}\n({z})" for s, z in zip(_COMB_DECK_SHORT, _COMB_DECK_Z)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, phi, ylabel, title in zip(
        axes,
        [phi_pf, phi_lf],
        ["Shapley φ  [ΔPF_min]",               "Shapley φ  [Δ(−LF_mean)]"],
        ["Bidrag til min_PF_focus\n(svakeste etasje)",  "Bidrag til −mean_LF_focus\n(gjennomsnittlig dose)"],
    ):
        bar_colors = ["#4CAF50" if v >= 0 else "#F44336" for v in phi]
        ax.bar(x, phi, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.8)
        scale = float(np.max(np.abs(phi))) if np.any(phi != 0) else 1.0
        for xi, v in enumerate(phi):
            ax.text(xi, v + 0.04 * np.sign(v + 1e-12) * scale,
                    f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(xlbls, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"{facade_label} — {title}", fontsize=10, weight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Shapley-verdier: gjennomsnittlig marginalbidrag per dekke\n{facade_label}",
        fontsize=12, weight="bold",
    )
    fig.tight_layout()
    finish_figure(fig, save_path=save_path, show=show)


# ─────────────────────────────────────────────────────────────────────────────
# Robusthetssjekk: tre målsett (A / B / C)
# ─────────────────────────────────────────────────────────────────────────────
# Målsett A: 2.–4. etasje  — oppholdsetasjer der M_floor-effekten er størst
# Målsett B: Kjeller       — SIP-tilfelle; vurderes separat
# Målsett C: Kjeller + 2.–4. etasje — samlet aktuelle oppholdssoner
#
# Begrunnet eksklusjon:
#   1. etasje: sterkt ground-shine-dominert gjennom fasade/vinduer → lav PF
#              uansett M_floor; inkludering vrir aggregatmål mot fasadeeffekter
#   5. etasje: roof-shine-dominert, 0 etasjeskiller over mottakerpunktet →
#              PF nesten uavhengig av M_floor; inkludering fortynner signalet
# ─────────────────────────────────────────────────────────────────────────────

def _add_focus_metrics(df: pd.DataFrame) -> pd.DataFrame:
    # Beregner mean_LF og min_PF for målsett A, B og C.
    df = df.copy()
    df["mean_LF_A"] = df[["LF_2F", "LF_3F", "LF_4F"]].mean(axis=1)
    df["min_PF_A"]  = df[["PF_2F", "PF_3F", "PF_4F"]].min(axis=1)
    df["mean_LF_B"] = df["LF_Kjeller"]
    df["min_PF_B"]  = df["PF_Kjeller"]
    df["mean_LF_C"] = df[["LF_Kjeller", "LF_2F", "LF_3F", "LF_4F"]].mean(axis=1)
    df["min_PF_C"]  = df[["PF_Kjeller", "PF_2F", "PF_3F", "PF_4F"]].min(axis=1)
    return df


def plot_multifocus_shapley(
    df: pd.DataFrame,
    facade_label: str,
    save_path=None,
    show: bool = False,
) -> None:
    # Shapley-verdier for min_PF under tre målsett.
    # Positiv φ: dekket øker min PF for valgte fokus-etasjer.
    # Sammenligner om dekke-rangeringen endres med målsettet.

    df  = _add_focus_metrics(df)
    sub = df[df.facade == facade_label].copy()
    sub["_neg_lf_A"] = -sub["mean_LF_A"]
    sub["_neg_lf_B"] = -sub["mean_LF_B"]
    sub["_neg_lf_C"] = -sub["mean_LF_C"]

    sets = [
        ("min_PF_A",   "Målsett A: 2.–4. etasje\n(oppholdsetasjer, M_floor-effekten størst)"),
        ("min_PF_B",   "Målsett B: Kjeller/SIP\n(alle dekker forventes lik vekt)"),
        ("min_PF_C",   "Målsett C: Kjeller + 2.–4. etasje\n(samlet aktuelle oppholdssoner)"),
    ]
    x     = np.arange(5)
    xlbls = [f"{s}\n({z})" for s, z in zip(_COMB_DECK_SHORT, _COMB_DECK_Z)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

    for ax, (col, title) in zip(axes, sets):
        phi = _shapley_values(sub, col)
        bar_colors = ["#4CAF50" if v >= 0 else "#F44336" for v in phi]
        ax.bar(x, phi, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.8)
        scale = float(np.max(np.abs(phi))) if np.any(phi != 0) else 1.0
        for xi, v in enumerate(phi):
            ax.text(xi, v + 0.04 * np.sign(v + 1e-12) * scale,
                    f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(xlbls, fontsize=10)
        ax.set_ylabel("Shapley φ [Δ min_PF]", fontsize=10)
        ax.set_title(f"{facade_label} — {title}", fontsize=10, weight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Shapley-verdier per dekke for tre målsett\n"
        "φ > 0: dekket øker min_PF for valgt fokusgruppe",
        fontsize=11, weight="bold",
    )
    fig.tight_layout()
    finish_figure(fig, save_path=save_path, show=show)


def plot_multifocus_best_by_n(
    df: pd.DataFrame,
    facade_label: str,
    save_path=None,
    show: bool = False,
) -> None:
    # Beste og verste kombinasjon per N for Målsett A (venstre) og B (høyre).
    # Viser at dekke-rangeringen kan avvike mellom oppholdsetasjer og kjeller.

    df  = _add_focus_metrics(df)
    sub = df[df.facade == facade_label].copy()
    ns  = sorted(sub["n_stubb"].unique())

    configs = [
        ("mean_LF_A", True,
         "Gjennomsnittlig LF, 2.–4. etasje",
         "Målsett A — 2.–4. etasje (lavere = bedre)\nOppholdsetasjer: M_floor-effekten domineres av øvre dekker"),
        ("mean_LF_B", True,
         "LF Kjeller",
         "Målsett B — Kjeller/SIP (lavere = bedre)\nAlle dekker bidrar tilnærmet likt"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, (col, ascending, ylabel, title) in zip(axes, configs):
        best_v, worst_v, best_lbls = [], [], []
        for ns_ in ns:
            grp = sub[sub["n_stubb"] == ns_].sort_values(col, ascending=ascending)
            best_v.append(float(grp.iloc[0][col]))
            worst_v.append(float(grp.iloc[-1][col]))
            best_lbls.append(grp.iloc[0]["combo_label"])

        ax.fill_between(ns, best_v, worst_v, alpha=0.15, color="steelblue", label="Spenn (beste–verste)")
        ax.plot(ns, best_v,  "o-",  color="steelblue", linewidth=2.5, label="Beste kombinasjon")
        ax.plot(ns, worst_v, "s--", color="gray",      linewidth=1.5, alpha=0.7, label="Verste kombinasjon")

        for ns_, bv, lbl in zip(ns, best_v, best_lbls):
            ax.annotate(lbl, xy=(ns_, bv), xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=8.5, color="steelblue", family="monospace")

        ax.set_xticks(ns)
        ax.set_xlabel("Antall stubbloft-dekker bevart (N)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{facade_label} — {title}", fontsize=10, weight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.tight_layout()
    finish_figure(fig, save_path=save_path, show=show)


def plot_deck_pair_heatmap_setA(
    df: pd.DataFrame,
    facade_label: str,
    save_path=None,
    show: bool = False,
) -> None:
    # Par-heatmap for Målsett A (2.–4. etasje): lavest mean_LF_A = beste 2-dekke-par.

    df  = _add_focus_metrics(df)
    sub = df[df.facade == facade_label].copy()
    n   = 5
    mat = np.full((n, n), np.nan)

    for _, row in sub[sub["n_stubb"] == 2].iterrows():
        stubb = [k for k in range(n) if int(row[f"d{k}"]) == 1]
        if len(stubb) == 2:
            i, j = stubb
            mat[i, j] = mat[j, i] = float(row["mean_LF_A"])

    tick_lbls = [f"{s}\n({z})" for s, z in zip(_COMB_DECK_SHORT, _COMB_DECK_Z)]
    vmax_val  = float(np.nanmax(mat))
    vmin_val  = float(np.nanmin(mat))

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=vmin_val, vmax=vmax_val)
    plt.colorbar(im, ax=ax, label="mean_LF  2.–4. etasje  (lavere = bedre)", shrink=0.88)
    ax.set_xticks(range(n))
    ax.set_xticklabels(tick_lbls, fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels(tick_lbls, fontsize=10)
    ax.set_title(
        f"{facade_label} — Målsett A: 2.–4. etasje\n"
        "mean_LF lavest = beste 2-dekke-kombinasjon  (N=2 bevart som stubbloft)",
        fontsize=10, weight="bold",
    )
    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.4f}", ha="center", va="center",
                        fontsize=9, color="white" if v > 0.65 * vmax_val else "black")
    fig.tight_layout()
    finish_figure(fig, save_path=save_path, show=show)
