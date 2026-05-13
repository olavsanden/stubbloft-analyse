# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .calculations import rows_to_plot_df
from .config import ResultRow


def finish_figure(
    fig: plt.Figure,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> None:
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_pf_profiles(
    data: Dict[str, List[ResultRow]],
    scenario_name: str,
    point_name: str,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> None:
    # PF per etasje med to x-akser:
    # nederst vises beskyttelsesfaktor PF, øverst vises tilsvarende doseprosent.
    # Y-aksen viser etasjenavn (ikke meter) for bedre lesbarhet.

    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    first_df = rows_to_plot_df(next(iter(data.values()))).sort_values("z").reset_index(drop=True)
    floor_labels = first_df["Etasje"].tolist()
    y = np.arange(len(floor_labels))

    all_pf_values = []

    for name, rows in data.items():
        df = rows_to_plot_df(rows).sort_values("z").reset_index(drop=True)
        all_pf_values.extend(df["PF"].values.tolist())
        ax.plot(df["PF"].values, y, marker="o", linewidth=2.5, label=short_building_label(name))

    ax.axvspan(5, 10, alpha=0.15, label="PF 5–10")
    setup_pf_axis_with_dose_axis(ax, all_pf_values, xlabel_fontsize=11)

    ax.set_yticks(y)
    ax.set_yticklabels(floor_labels, fontsize=11)
    ax.set_ylabel("Etasje")
    ax.set_title(
        f"Beskyttelse per etasje\n{scenario_name} | {point_name}",
        fontsize=15,
        weight="bold",
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    fig.subplots_adjust(left=0.14, right=0.98, top=0.84, bottom=0.13)
    finish_figure(fig, save_path=save_path, show=show)


def short_building_label(name: str) -> str:
    # Kortere etiketter for figurer — matcher de seks scenariene i make_buildings.

    n = name.lower()
    if "murbygg" in n and "stubbloft" in n:     return "Mur | stubbloft"
    if "murbygg" in n and "utskifting" in n:    return "Mur | utskiftet"
    if "murbygg" in n and "referanse" in n:     return "Mur | lett ref."
    if "trebygg" in n and "stubbloft" in n:     return "Tre | stubbloft"
    if "trebygg" in n and "utskifting" in n:    return "Tre | utskiftet"
    if "trebygg" in n and "referanse" in n:     return "Tre | lett ref."
    return name


def pf_to_dose_percent(pf):
    # Omregning fra PF til innendørs dose i prosent av utendørs dose.

    pf = np.asarray(pf, dtype=float)
    return 100.0 / np.maximum(pf, 1e-12)


def dose_percent_to_pf(dose_percent):
    # Omregning fra innendørs doseprosent tilbake til PF.

    dose_percent = np.asarray(dose_percent, dtype=float)
    return 100.0 / np.maximum(dose_percent, 1e-12)


def nice_number_label(value):
    # Gir korte og lesbare akseetiketter som 1, 2, 5, 10, 0.5 osv.

    value = float(value)
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:g}"


def setup_pf_axis_with_dose_axis(ax, pf_values, xlabel_fontsize=12):
    # Setter nederste x-akse til PF og øverste x-akse til tilsvarende doseprosent.
    # PF-aksen beholdes logaritmisk, men tick-etikettene vises som vanlige tall.

    pf_values = np.asarray(pf_values, dtype=float)
    pf_values = pf_values[np.isfinite(pf_values) & (pf_values > 0)]

    if pf_values.size == 0:
        pf_max = 10.0
    else:
        pf_max = max(10.0, float(np.nanmax(pf_values)) * 1.15)

    pf_tick_candidates = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    x_left = 1.0
    x_right = next((t for t in pf_tick_candidates if t >= pf_max), None)

    if x_right is None:
        x_right = 10 ** np.ceil(np.log10(pf_max))

    pf_ticks = [t for t in pf_tick_candidates if x_left <= t <= x_right]

    ax.set_xscale("log")
    ax.set_xlim(x_left, x_right)
    ax.set_xticks(pf_ticks)
    ax.set_xticklabels([nice_number_label(t) for t in pf_ticks])
    ax.set_xlabel(
        "Beskyttelsesfaktor PF [-] (høyere = bedre skjerming)",
        fontsize=xlabel_fontsize,
    )

    secax = ax.secondary_xaxis(
        "top",
        functions=(pf_to_dose_percent, dose_percent_to_pf),
    )

    dose_tick_candidates = [100, 50, 20, 10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.02]
    dose_ticks = []
    for d in dose_tick_candidates:
        pf_at_tick = float(dose_percent_to_pf(d))
        if x_left <= pf_at_tick <= x_right:
            dose_ticks.append(d)

    secax.set_xticks(dose_ticks)
    secax.set_xticklabels([f"{nice_number_label(d)} %" for d in dose_ticks])
    secax.set_xlabel(
        "Innendørs dose [% av utendørs dose] (lavere = bedre)",
        fontsize=xlabel_fontsize,
    )

    return secax


def plot_pf_profile_with_building_section(
    data: Dict[str, List[ResultRow]],
    scenario_name: str,
    point_name: str,
    building_image_path: Optional[str] = None,
    use_short_labels: bool = True,
    show_height_in_labels: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> None:
    # Lager PF-profil med byggsnitt/illustrasjon til venstre og PF-kurver til høyre.

    # Hvis building_image_path er None, tegnes et enkelt skjematisk snitt i stedet.
    # Y-aksen vises som etasjer, ikke meter. Meter kan legges i etikettene med
    # show_height_in_labels=True.

    first_df = rows_to_plot_df(next(iter(data.values()))).sort_values("z").reset_index(drop=True)
    floor_labels = first_df["Etasje"].tolist()
    if show_height_in_labels:
        floor_labels = [f"{e} ({z:.1f} m)" for e, z in zip(first_df["Etasje"], first_df["z"])]

    y = np.arange(len(floor_labels))

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.55], wspace=0.04)

    ax_img = fig.add_subplot(gs[0, 0])

    if building_image_path:
        img = plt.imread(building_image_path)
        ax_img.imshow(img, extent=[0, 1, -0.5, len(floor_labels) - 0.5], aspect="auto")
        ax_img.set_xlim(0, 1)
    else:
        # Enkel skjematisk bygning dersom man ikke har et snittbilde klart.
        ax_img.set_xlim(0, 1)
        ax_img.plot([0.25, 0.25, 0.75, 0.75, 0.25], [-0.4, len(y)-0.1, len(y)-0.1, -0.4, -0.4], linewidth=2)
        for yi in y:
            ax_img.plot([0.25, 0.75], [yi, yi], linewidth=1.5)
        ax_img.plot([0.25, 0.50, 0.75], [len(y)-0.1, len(y)+0.45, len(y)-0.1], linewidth=2)

    ax_img.set_ylim(-0.5, len(floor_labels) - 0.5)
    ax_img.set_xticks([])
    ax_img.set_yticks(y)
    ax_img.set_yticklabels(floor_labels, fontsize=12)
    ax_img.set_title("Byggesnitt og etasjer", fontsize=16, weight="bold", pad=12)

    for yi in y:
        ax_img.axhline(yi, linestyle="--", linewidth=1, alpha=0.35)

    for spine in ["top", "right", "bottom"]:
        ax_img.spines[spine].set_visible(False)

    ax = fig.add_subplot(gs[0, 1], sharey=ax_img)

    all_pf_values = []

    for name, rows in data.items():
        df = rows_to_plot_df(rows).sort_values("z").reset_index(drop=True)
        all_pf_values.extend(df["PF"].values.tolist())
        label = short_building_label(name) if use_short_labels else name
        ax.plot(df["PF"].values, y, marker="o", linewidth=2.5, markersize=7, label=label)

    ax.axvspan(5, 10, alpha=0.12, label="PF 5–10")
    setup_pf_axis_with_dose_axis(ax, all_pf_values, xlabel_fontsize=11)

    ax.set_yticks(y)
    ax.set_yticklabels(floor_labels, fontsize=12)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_title(
        f"Beskyttelse per etasje\n{scenario_name} | {point_name}",
        fontsize=16,
        weight="bold",
        pad=14,
    )
    ax.legend(fontsize=10, loc="best", frameon=True)

    fig.subplots_adjust(left=0.05, right=0.98, top=0.84, bottom=0.12, wspace=0.04)
    finish_figure(fig, save_path=save_path, show=show)


def plot_location_factor(
    data: Dict[str, List[ResultRow]],
    scenario_name: str,
    point_name: str,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> None:
    # LF per etasje.

    plt.figure(figsize=(9, 6))

    for name, rows in data.items():
        df = rows_to_plot_df(rows)
        plt.plot(df["LF"], df["z"], marker="o", linewidth=2.5, label=name)

    plt.axvline(0.15, linestyle="--", linewidth=2, label="LF = 0,15")
    plt.axvspan(0.10, 0.20, alpha=0.15, label="LF 0,10–0,20")
    plt.xlabel("Location Factor LF = 1/PF [-]")
    plt.ylabel("Høyde målepunkt z [m]")
    plt.title(f"Location factor per etasje\n{scenario_name} | {point_name}")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    finish_figure(plt.gcf(), save_path=save_path, show=show)


def plot_pf_heatmap(
    data: Dict[str, List[ResultRow]],
    scenario_name: str,
    point_name: str,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> None:
    # Heatmap av PF.

    labels = list(data.keys())
    short_labels = [short_building_label(name) for name in labels]
    floors = rows_to_plot_df(data[labels[0]])["Etasje"].tolist()
    M = np.array([rows_to_plot_df(data[name])["PF"].values for name in labels])

    plt.figure(figsize=(9, 4.5))
    im = plt.imshow(np.log10(M), aspect="auto")

    plt.colorbar(im, label="log10(PF)")
    plt.yticks(np.arange(len(labels)), short_labels, fontsize=10)
    plt.xticks(np.arange(len(floors)), floors, rotation=30)
    plt.title(f"PF-heatmap, log10-skala\n{scenario_name} | {point_name}")

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            plt.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    finish_figure(plt.gcf(), save_path=save_path, show=show)


def plot_indoor_dose_components(
    data: Dict[str, List[ResultRow]],
    scenario_name: str,
    point_name: str,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> None:
    # Stablet søyleplott: ground og roof shine som andel av utendørs dose [%].
    # Søylehøyde = LF * 100 = total innendørs dose i % av utendørs dose.
    # Ground-andel = LF * Ground%,  Roof-andel = LF * Roof%.

    floors = rows_to_plot_df(next(iter(data.values())))["Etasje"].tolist()
    x = np.arange(len(floors))
    width = 0.25

    plt.figure(figsize=(12, 5.5))

    for i, (name, rows) in enumerate(data.items()):
        df = rows_to_plot_df(rows)
        ground_pct = df["LF"] * df["Ground%"]
        roof_pct   = df["LF"] * df["Roof%"]

        offset = (i - 1) * width
        label_short = short_building_label(name)

        plt.bar(x + offset, ground_pct, width, label=f"{label_short} – ground shine")
        plt.bar(x + offset, roof_pct,   width, bottom=ground_pct, label=f"{label_short} – roof shine")

    plt.xticks(x, floors, rotation=30)
    plt.ylabel("Innendørs dose [% av utendørs dose]")
    plt.title(f"Fordeling av innendørs dosebidrag\n{scenario_name} | {point_name}")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    finish_figure(plt.gcf(), save_path=save_path, show=show)


def plot_floor_crossings(
    data: Dict[str, List[ResultRow]],
    scenario_name: str,
    point_name: str,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> None:
    # Plotter gjennomsnittlig antall kryssede etasjeskiller.

    plt.figure(figsize=(9, 5))

    for name, rows in data.items():
        df = rows_to_plot_df(rows)
        plt.plot(df["Cross"], df["z"], marker="o", linewidth=2.5, label=name)

    plt.xlabel("Gjennomsnittlig antall kryssede etasjeskiller")
    plt.ylabel("Høyde målepunkt z [m]")
    plt.title(f"Geometrisk skjerming fra etasjeskiller\n{scenario_name} | {point_name}")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    finish_figure(plt.gcf(), save_path=save_path, show=show)


def plot_ground_roof_share_by_floor(
    data: Dict[str, List[ResultRow]],
    scenario_name: str,
    point_name: str,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> None:
    # Horisontalt stablet søyleplott: andel av innendørs dose fra ground og roof shine.
    # Y-akse: etasjenavn (nedenfra og opp). X-akse: 0–100 % av innendørs dose.
    # Én subplot per bygningstype.

    n = len(data)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6), sharey=True)
    if n == 1:
        axes = [axes]

    c_ground = "#2196F3"
    c_roof   = "#FF9800"

    for ax, (name, rows) in zip(axes, data.items()):
        df = rows_to_plot_df(rows).sort_values("z").reset_index(drop=True)
        y = np.arange(len(df))

        ax.barh(y, df["Ground%"], color=c_ground, label="Ground shine")
        ax.barh(y, df["Roof%"],   left=df["Ground%"], color=c_roof, label="Roof shine")

        ax.set_xlim(0, 100)
        ax.set_xlabel("Andel av innendørs dose [%]")
        ax.set_yticks(y)
        ax.set_yticklabels(df["Etasje"].tolist(), fontsize=11)
        ax.set_title(short_building_label(name), fontsize=12, weight="bold")
        ax.axvline(50, linestyle="--", linewidth=0.8, alpha=0.45, color="gray")
        ax.grid(axis="x", alpha=0.3)

        if ax is axes[0]:
            ax.legend(fontsize=9, loc="lower right")

    fig.suptitle(
        f"Ground/roof shine-andeler per etasje\n{scenario_name} | {point_name}",
        fontsize=14,
        weight="bold",
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.84, bottom=0.12, wspace=0.06)
    finish_figure(fig, save_path=save_path, show=show)


def plot_pf_sensitivity_band_by_facade(
    facade_label: str,
    rows_stubb_main: List[ResultRow],
    rows_stubb_low: List[ResultRow],
    rows_stubb_high: List[ResultRow],
    rows_utsk_main: List[ResultRow],
    rows_utsk_low: List[ResultRow],
    rows_utsk_high: List[ResultRow],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> None:
    # PF per etasje med sensitivitetsbånd for de to etasjeskillerkonseptene:
    # stubbloft bevart og full utskifting + mineralull.
    # Båndet spenner mellom lav og høy M_floor fra parameterrommet.

    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    df_ref = rows_to_plot_df(rows_stubb_main).sort_values("z").reset_index(drop=True)
    floor_labels = df_ref["Etasje"].tolist()
    y = np.arange(len(floor_labels))

    def _pf(rows: List[ResultRow]) -> np.ndarray:
        return rows_to_plot_df(rows).sort_values("z").reset_index(drop=True)["PF"].values

    pf_sm = _pf(rows_stubb_main)
    pf_sl = _pf(rows_stubb_low)
    pf_sh = _pf(rows_stubb_high)
    pf_um = _pf(rows_utsk_main)
    pf_ul = _pf(rows_utsk_low)
    pf_uh = _pf(rows_utsk_high)

    c_stubb = "#D62728"
    c_utsk  = "#1F77B4"

    ax.fill_betweenx(
        y,
        np.minimum(pf_sl, pf_sh),
        np.maximum(pf_sl, pf_sh),
        alpha=0.18, color=c_stubb,
        label="Stubbloft – sensitivitetsbånd (132–230 kg/m²)",
    )
    ax.fill_betweenx(
        y,
        np.minimum(pf_ul, pf_uh),
        np.maximum(pf_ul, pf_uh),
        alpha=0.18, color=c_utsk,
        label="Utskifting – sensitivitetsbånd (49–76 kg/m²)",
    )

    ax.plot(pf_sm, y, marker="o", linewidth=2.5, color=c_stubb,
            label="Stubbloft bevart (145 kg/m²)")
    ax.plot(pf_um, y, marker="o", linewidth=2.5, color=c_utsk,
            label="Full utskifting + mineralull (62 kg/m²)")

    all_pf = np.concatenate([pf_sm, pf_sl, pf_sh, pf_um, pf_ul, pf_uh])
    setup_pf_axis_with_dose_axis(ax, all_pf, xlabel_fontsize=11)

    ax.set_yticks(y)
    ax.set_yticklabels(floor_labels, fontsize=11)
    ax.set_ylabel("Etasje")
    ax.set_title(
        f"PF per etasje med sensitivitetsbånd\n{facade_label}",
        fontsize=15,
        weight="bold",
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

    fig.subplots_adjust(left=0.14, right=0.98, top=0.84, bottom=0.13)
    finish_figure(fig, save_path=save_path, show=show)


def plot_skjermingsprofil(
    data: Dict[str, List[ResultRow]],
    scenario_name: str,
    point_name: str,
    building_name: str,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> None:
    # Polarplot for én bygningsvariant.

    rows = data[building_name]
    df = rows_to_plot_df(rows)

    theta = np.linspace(0, 2 * np.pi, len(df), endpoint=False)
    theta = np.r_[theta, theta[0]]

    pf = np.r_[df["PF"].values, df["PF"].values[0]]
    ground = np.r_[df["Ground%"].values, df["Ground%"].values[0]]
    roof = np.r_[df["Roof%"].values, df["Roof%"].values[0]]

    fig = plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)

    ax.plot(theta, pf / np.nanmax(pf), marker="o", linewidth=2.5, label="PF normalisert")
    ax.plot(theta, ground / 100, marker="o", linewidth=2, label="Ground shine")
    ax.plot(theta, roof / 100, marker="o", linewidth=2, label="Roof shine")

    ax.set_xticks(theta[:-1])
    ax.set_xticklabels(df["Etasje"].tolist())
    ax.set_title(f"Skjermingsprofil\n{scenario_name}\n{building_name}\n{point_name}", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15))

    plt.tight_layout()
    finish_figure(fig, save_path=save_path, show=show)


# ─────────────────────────────────────────────────────────────────────────────
# Nye rapportfigurer: sammenligning per scenario
# ─────────────────────────────────────────────────────────────────────────────

def _find_building(data: Dict[str, List[ResultRow]], facade: str, concept: str) -> Optional[str]:
    # Finn bygningsnavn i data-dict ved tekstsøk på fasade og konsept.
    facade_l, concept_l = facade.lower(), concept.lower()
    return next(
        (n for n in data if facade_l in n.lower() and concept_l in n.lower()),
        None,
    )


def plot_pf_band_per_scenario(
    facade_label: str,
    scenario_name: str,
    point_name: str,
    rows_stubb_main: List[ResultRow],
    rows_stubb_low: List[ResultRow],
    rows_stubb_high: List[ResultRow],
    rows_utsk_main: List[ResultRow],
    rows_utsk_low: List[ResultRow],
    rows_utsk_high: List[ResultRow],
    include_basement: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> None:
    # PF per etasje med sensitivitetsbånd. Kjelleråpningsscenario vises i tittel.
    # include_basement=False lager en profil for kun oppholdsetasjene (1.–5. etasje).

    def _df(rows: List[ResultRow]) -> pd.DataFrame:
        df = rows_to_plot_df(rows).sort_values("z").reset_index(drop=True)
        if not include_basement:
            df = df[df["Etasje"] != "Kjeller"].reset_index(drop=True)
        return df

    df_ref = _df(rows_stubb_main)
    if df_ref.empty:
        return

    floor_labels = df_ref["Etasje"].tolist()
    y = np.arange(len(floor_labels))

    def _pf(rows: List[ResultRow]) -> np.ndarray:
        return _df(rows)["PF"].values

    pf_sm = _pf(rows_stubb_main)
    pf_sl = _pf(rows_stubb_low)
    pf_sh = _pf(rows_stubb_high)
    pf_um = _pf(rows_utsk_main)
    pf_ul = _pf(rows_utsk_low)
    pf_uh = _pf(rows_utsk_high)

    c_stubb = "#D62728"
    c_utsk  = "#1F77B4"

    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    if include_basement:
        ax.axhspan(-0.5, 0.5, alpha=0.09, color="gray", zorder=0)
        ax.text(
            1.01, 0.5 / len(floor_labels),
            "Kjeller:\nsterkt avhengig\nav åpningstype",
            transform=ax.transAxes, fontsize=7.5, color="#666666",
            va="center", ha="left",
        )

    ax.fill_betweenx(y, np.minimum(pf_sl, pf_sh), np.maximum(pf_sl, pf_sh),
                     alpha=0.18, color=c_stubb, label="Stubbloft – sensitivitetsbånd (132–230 kg/m²)")
    ax.fill_betweenx(y, np.minimum(pf_ul, pf_uh), np.maximum(pf_ul, pf_uh),
                     alpha=0.18, color=c_utsk,  label="Utskifting – sensitivitetsbånd (49–76 kg/m²)")
    ax.plot(pf_sm, y, marker="o", linewidth=2.5, color=c_stubb,
            label="Stubbloft bevart (145 kg/m²)")
    ax.plot(pf_um, y, marker="o", linewidth=2.5, color=c_utsk,
            label="Full utskifting + mineralull (62 kg/m²)")

    all_pf = np.concatenate([pf_sm, pf_sl, pf_sh, pf_um, pf_ul, pf_uh])
    setup_pf_axis_with_dose_axis(ax, all_pf, xlabel_fontsize=11)

    ax.set_yticks(y)
    ax.set_yticklabels(floor_labels, fontsize=11)
    ax.set_ylabel("Etasje")
    ax.set_title(
        f"PF per etasje med sensitivitetsbånd\n{facade_label} | {scenario_name}",
        fontsize=12, weight="bold",
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

    fig.subplots_adjust(left=0.14, right=0.88, top=0.84, bottom=0.13)
    finish_figure(fig, save_path=save_path, show=show)


def plot_pf_bars(
    data: Dict[str, List[ResultRow]],
    scenario_name: str,
    facade_label: str,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> None:
    # Gruppert stolpediagram: PF per etasje for stubbloft vs. full utskifting.

    s_name = _find_building(data, facade_label, "stubbloft")
    u_name = _find_building(data, facade_label, "utskifting")
    if s_name is None or u_name is None:
        return

    df_s = rows_to_plot_df(data[s_name]).sort_values("z").reset_index(drop=True)
    df_u = rows_to_plot_df(data[u_name]).sort_values("z").reset_index(drop=True)
    floor_labels = df_s["Etasje"].tolist()

    x = np.arange(len(floor_labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - w / 2, df_s["PF"].values, w, color="#D62728", alpha=0.85,
           label="Stubbloft bevart (145 kg/m²)")
    ax.bar(x + w / 2, df_u["PF"].values, w, color="#1F77B4", alpha=0.85,
           label="Full utskifting + mineralull (62 kg/m²)")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(floor_labels, fontsize=11)
    ax.set_xlabel("Etasje")
    ax.set_ylabel("Beskyttelsesfaktor PF [-] (log-skala)")
    ax.set_title(
        f"PF per etasje\n{facade_label} | {scenario_name}",
        fontsize=13, weight="bold",
    )
    ax.grid(True, which="both", alpha=0.3, axis="y")
    ax.legend(fontsize=10)
    fig.tight_layout()
    finish_figure(fig, save_path=save_path, show=show)


def plot_pf_ratio(
    data: Dict[str, List[ResultRow]],
    scenario_name: str,
    facade_label: str,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> None:
    # Horisontalt stolpediagram: PF_stubbloft / PF_utskifting per etasje.
    # Verdier > 1 viser at stubbloft gir bedre skjerming.

    s_name = _find_building(data, facade_label, "stubbloft")
    u_name = _find_building(data, facade_label, "utskifting")
    if s_name is None or u_name is None:
        return

    df_s = rows_to_plot_df(data[s_name]).sort_values("z").reset_index(drop=True)
    df_u = rows_to_plot_df(data[u_name]).sort_values("z").reset_index(drop=True)
    floor_labels = df_s["Etasje"].tolist()
    y = np.arange(len(floor_labels))
    ratio = df_s["PF"].values / np.maximum(df_u["PF"].values, 1e-12)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.barh(y, ratio, color="#2CA02C", alpha=0.82)
    ax.axvline(1.0, color="black", linewidth=1.5, linestyle="--", label="Ratio = 1")
    for i, r in enumerate(ratio):
        ax.text(r + 0.05, i, f"{r:.1f}×", va="center", fontsize=10)

    x_max = max(ratio.max() * 1.18, 2.0)
    ax.set_xlim(0, x_max)
    ax.set_yticks(y)
    ax.set_yticklabels(floor_labels, fontsize=11)
    ax.set_xlabel("PF_stubbloft / PF_utskifting [-]")
    ax.set_ylabel("Etasje")
    ax.set_title(
        f"PF-ratio: stubbloft / utskifting\n{facade_label} | {scenario_name}",
        fontsize=13, weight="bold",
    )
    ax.grid(True, alpha=0.3, axis="x")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    finish_figure(fig, save_path=save_path, show=show)


def plot_pf_reduction_pct(
    data: Dict[str, List[ResultRow]],
    scenario_name: str,
    facade_label: str,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> None:
    # Horisontalt stolpediagram: prosentvis PF-reduksjon ved full utskifting.
    # Verdi = 100 × (1 − PF_utskifting / PF_stubbloft).

    s_name = _find_building(data, facade_label, "stubbloft")
    u_name = _find_building(data, facade_label, "utskifting")
    if s_name is None or u_name is None:
        return

    df_s = rows_to_plot_df(data[s_name]).sort_values("z").reset_index(drop=True)
    df_u = rows_to_plot_df(data[u_name]).sort_values("z").reset_index(drop=True)
    floor_labels = df_s["Etasje"].tolist()
    y = np.arange(len(floor_labels))
    reduction = 100.0 * (1.0 - df_u["PF"].values / np.maximum(df_s["PF"].values, 1e-12))

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.barh(y, reduction, color="#FF7F0E", alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    for i, r in enumerate(reduction):
        ax.text(max(r + 0.5, 1.0), i, f"{r:.0f} %", va="center", fontsize=10)

    ax.set_xlim(0, max(reduction.max() * 1.18, 10))
    ax.set_yticks(y)
    ax.set_yticklabels(floor_labels, fontsize=11)
    ax.set_xlabel("PF-reduksjon ved utskifting av stubbloft [%]")
    ax.set_ylabel("Etasje")
    ax.set_title(
        f"PF-reduksjon: stubbloft → utskifting\n{facade_label} | {scenario_name}",
        fontsize=13, weight="bold",
    )
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    finish_figure(fig, save_path=save_path, show=show)


def plot_ground_roof_main_buildings(
    data: Dict[str, List[ResultRow]],
    scenario_name: str,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> None:
    # 2×2 subplot: ground/roof shine-andeler per etasje for de fire hovedvariantene.

    targets = [
        ("murbygg", "stubbloft"),
        ("murbygg", "utskifting"),
        ("trebygg", "stubbloft"),
        ("trebygg", "utskifting"),
    ]
    names = [_find_building(data, f, c) for f, c in targets]
    if any(n is None for n in names):
        return

    c_ground = "#2196F3"
    c_roof   = "#FF9800"

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=False)
    axes = axes.flatten()

    for ax, name in zip(axes, names):
        df = rows_to_plot_df(data[name]).sort_values("z").reset_index(drop=True)
        y = np.arange(len(df))
        ax.barh(y, df["Ground%"], color=c_ground, label="Ground shine")
        ax.barh(y, df["Roof%"], left=df["Ground%"], color=c_roof, label="Roof shine")
        ax.set_xlim(0, 100)
        ax.set_xlabel("Andel av innendørs dose [%]")
        ax.set_yticks(y)
        ax.set_yticklabels(df["Etasje"].tolist(), fontsize=10)
        ax.set_title(short_building_label(name), fontsize=11, weight="bold")
        ax.axvline(50, linestyle="--", linewidth=0.8, alpha=0.4, color="gray")
        ax.grid(axis="x", alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(
        f"Ground/Roof shine-andeler per etasje\n{scenario_name}",
        fontsize=13, weight="bold",
    )
    fig.tight_layout()
    finish_figure(fig, save_path=save_path, show=show)


def plot_basement_opening_sensitivity(
    all_scenario_data: Dict[str, Dict[str, List[ResultRow]]],
    facade_label: str,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> None:
    # Stolpediagram: kjeller-PF for stubbloft og utskifting på tvers av åpningsscenarioer.

    scenario_names = list(all_scenario_data.keys())
    pf_stubb, pf_utsk = [], []

    for sname in scenario_names:
        d = all_scenario_data[sname]
        sn = _find_building(d, facade_label, "stubbloft")
        un = _find_building(d, facade_label, "utskifting")
        def _kjeller_pf(name):
            if name is None:
                return np.nan
            df = rows_to_plot_df(d[name])
            row = df[df["Etasje"] == "Kjeller"]["PF"]
            return float(row.values[0]) if len(row) > 0 else np.nan
        pf_stubb.append(_kjeller_pf(sn))
        pf_utsk.append(_kjeller_pf(un))

    short_labels = []
    for s in scenario_names:
        sl = s.lower()
        if "40" in sl and ("60" in sl or "x60" in sl):
            short_labels.append("Kjellervindu\n40×60 cm")
        elif "60" in sl and ("30" in sl or "x30" in sl):
            short_labels.append("Kjellervindu\n60×30 cm")
        elif "luftespalte" in sl or "15" in sl:
            short_labels.append("Luftespalte\n15×15 cm")
        else:
            short_labels.append(s)

    x = np.arange(len(scenario_names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.bar(x - w / 2, pf_stubb, w, color="#D62728", alpha=0.85,
           label="Stubbloft bevart (145 kg/m²)")
    ax.bar(x + w / 2, pf_utsk,  w, color="#1F77B4", alpha=0.85,
           label="Full utskifting + mineralull (62 kg/m²)")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=11)
    ax.set_xlabel("Kjelleråpningsscenario")
    ax.set_ylabel("Kjeller-PF [-] (log-skala)")
    ax.set_title(
        f"Kjellerplan: PF per åpningsscenario\n{facade_label}",
        fontsize=13, weight="bold",
    )
    ax.grid(True, which="both", alpha=0.3, axis="y")
    ax.legend(fontsize=10)
    fig.tight_layout()
    finish_figure(fig, save_path=save_path, show=show)
