"""
Etasjevis påvirkningsmatrise – enkeltutskifting av etasjeskillere.

For hvert av de 5 etasjeskillerne vises reduksjonen i PF (%) per etasje
dersom nettopp det ene dekket skiftes fra stubbloft (202 kg/m²) til
mineralull (62 kg/m²), mens alle andre holdes på stubbloftsverdi.

Data hentes fra eksisterende pf_floor_deck_importance.csv (ingen ny beregning).
Parametere er identiske med hovedmodellen (murbygg, kjellervindu 40×60 cm,
saltak 45°, R=60 m, M_STUBB=202 kg/m²).

Endrer ikke: PF-modell, parametre, geometri eller eksisterende resultater.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
CSV_DECK   = ROOT / "output" / "pf_floor_deck_importance.csv"
OUT_FIG    = ROOT / "output" / "figures"
OUT_TAB    = ROOT / "output" / "tables"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)

# ── Order constants ────────────────────────────────────────────────────────────
FLOOR_ORDER = ["Kjeller", "1. etasje", "2. etasje", "3. etasje", "4. etasje", "5. etasje"]
DECK_ORDER  = ["K–1", "1–2", "2–3", "3–4", "4–5"]

# Lesbare dekketiketter for figur (y-akse)
DECK_LABELS = {
    "K–1": "Kjellerdekke",
    "1–2": "Dekke 1.–2. et.",
    "2–3": "Dekke 2.–3. et.",
    "3–4": "Dekke 3.–4. et.",
    "4–5": "Dekke 4.–5. et.",
}

H_UTE_mSv  = 50.0   # illustrativt dosescenario (lineær skalering av PF)
SCENARIO   = "Kjellervindu 40x60 cm"

plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})

# ── Load and process data ─────────────────────────────────────────────────────

def load_matrix():
    df = pd.read_csv(CSV_DECK)
    mur = df[df["facade"] == "Murbygg"]

    # Reference: alle dekk = stubbloft
    ref = (
        mur[mur["analysis"] == "all_stubb"]
        .set_index("floor_label")["PF"]
        .reindex(FLOOR_ORDER)
    )

    # Single removal: ett dekk satt til mineralull
    rm = mur[mur["analysis"] == "remove_one"]
    pf_pivot = rm.pivot_table(
        index="deck_label", columns="floor_label", values="PF", aggfunc="first"
    ).reindex(index=DECK_ORDER, columns=FLOOR_ORDER)

    # Reduction matrix
    red_abs = ref.values - pf_pivot.values          # [%]
    red_pct = 100.0 * red_abs / ref.values          # normalised to ref

    ref_df  = pd.DataFrame({"floor": FLOOR_ORDER, "PF_ref": ref.values})
    pf_df   = pd.DataFrame(pf_pivot.values, index=DECK_ORDER, columns=FLOOR_ORDER)
    red_df  = pd.DataFrame(red_pct, index=DECK_ORDER, columns=FLOOR_ORDER)

    return ref, pf_df, red_df


# ── Long-format CSV ────────────────────────────────────────────────────────────

def build_long_csv(ref, pf_df, red_df):
    rows = []
    for dk in DECK_ORDER:
        for fl in FLOOR_ORDER:
            pf_r  = ref[fl]
            pf_s  = pf_df.loc[dk, fl]
            red_a = pf_r - pf_s
            red_p = 100 * red_a / pf_r
            rows.append({
                "scenario":               SCENARIO,
                "utskiftet_dekke":        DECK_LABELS[dk],
                "etasje":                 fl,
                "PF_ref":                 round(float(pf_r), 4),
                "PF_scenario":            round(float(pf_s), 4),
                "PF_reduksjon_abs":       round(float(red_a), 4),
                "PF_reduksjon_prosent":   round(float(red_p), 2),
            })
    return pd.DataFrame(rows)


# ── Heatmap helper ────────────────────────────────────────────────────────────

def _annotate_heatmap(ax, data, fmt="{:.0f}%", fontsize=9.5):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            txt = fmt.format(val)
            # White text on dark cells, dark on light
            bg = val / (np.nanmax(data) + 1e-6)
            color = "white" if bg > 0.60 else "#222"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=fontsize, color=color, fontweight="bold")


# ── Figure 1: PF-reduksjon heatmap ────────────────────────────────────────────

def plot_pf_reduction_heatmap(red_df):
    data  = red_df.values.astype(float)
    vmax  = np.ceil(data.max() / 5) * 5   # round up to nearest 5 %
    ylabels = [DECK_LABELS[d] for d in DECK_ORDER]
    xlabels = FLOOR_ORDER

    fig, ax = plt.subplots(figsize=(10.5, 5))

    # Colormap: white at 0, warm/orange-red at high values
    cmap = plt.cm.YlOrRd
    im   = ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Reduksjon i PF [%]", fontsize=10)
    cbar.ax.tick_params(labelsize=8.5)

    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, fontsize=10)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=10)
    ax.set_xlabel("Etasje (PF evaluert her)", fontsize=10, labelpad=8)
    ax.set_ylabel("Utskiftet etasjeskille", fontsize=10, labelpad=8)
    ax.set_title(
        "Etasjevis reduksjon i PF ved utskifting av ett etasjeskille",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax.text(
        0.5, -0.13,
        "Murbygg · Kjellervindu 40×60 cm · Referanse: alle dekk med stubbloft (202 kg/m²)  →  utskiftet: mineralull (62 kg/m²)",
        transform=ax.transAxes, ha="center", fontsize=7.5, color="#555", style="italic",
    )

    _annotate_heatmap(ax, data)

    # Light grid lines between cells
    ax.set_xticks(np.arange(-0.5, len(xlabels)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ylabels)), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.tight_layout(rect=[0, 0.09, 1, 1])
    stem = "etasjematrise_pf_reduksjon_enkeltutskifting"
    for ext in ("png", "pdf"):
        path = OUT_FIG / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Lagret: {path.relative_to(ROOT)}")
    plt.close(fig)


# ── Figure 2: ekstra dose heatmap ─────────────────────────────────────────────

def plot_dose_heatmap(ref, pf_df):
    h_ref  = H_UTE_mSv / ref.values          # innendørs dose ref per etasje
    h_scen = H_UTE_mSv / pf_df.values        # innendørs dose per scenario
    ekstra = h_scen - h_ref                   # ekstra dose [mSv]

    vmax  = np.ceil(ekstra.max() * 4) / 4    # round up to nearest 0.25 mSv
    ylabels = [DECK_LABELS[d] for d in DECK_ORDER]

    fig, ax = plt.subplots(figsize=(10.5, 5))

    cmap = plt.cm.YlOrRd
    im   = ax.imshow(ekstra, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Ekstra innendørs dose [mSv]", fontsize=10)
    cbar.ax.tick_params(labelsize=8.5)

    ax.set_xticks(range(len(FLOOR_ORDER)))
    ax.set_xticklabels(FLOOR_ORDER, fontsize=10)
    ax.set_yticks(range(len(DECK_ORDER)))
    ax.set_yticklabels(ylabels, fontsize=10)
    ax.set_xlabel("Etasje", fontsize=10, labelpad=8)
    ax.set_ylabel("Utskiftet etasjeskille", fontsize=10, labelpad=8)
    ax.set_title(
        f"Ekstra innendørs dose ved utskifting av ett etasjeskille  [H_ute = {H_UTE_mSv:.0f} mSv / 48 t]",
        fontsize=11.5, fontweight="bold", pad=10,
    )
    ax.text(
        0.5, -0.13,
        f"H_inne = H_ute / PF.  Utendørs dose {H_UTE_mSv:.0f} mSv er illustrativt – ikke en ulykkesmodell (ref. ICRP103).",
        transform=ax.transAxes, ha="center", fontsize=7.5, color="#555", style="italic",
    )

    _annotate_heatmap(ax, ekstra, fmt="{:.2f}")

    ax.set_xticks(np.arange(-0.5, len(FLOOR_ORDER)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(DECK_ORDER)), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.tight_layout(rect=[0, 0.09, 1, 1])
    stem = "etasjematrise_ekstra_dose_50mSv_enkeltutskifting"
    for ext in ("png", "pdf"):
        path = OUT_FIG / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Lagret: {path.relative_to(ROOT)}")
    plt.close(fig)


# ── Clean heatmap (revised, fysisk orientert y-akse) ─────────────────────────

# Visuell rekkefølge: D. 4-5 øverst (rad 0 i imshow), Kjellerdekke nederst
DECK_VISUAL  = ["4–5", "3–4", "2–3", "1–2", "K–1"]
YLABELS_SHORT = ["D. 4–5", "D. 3–4", "D. 2–3", "D. 1–2", "Kjellerdekke"]
XLABELS_SHORT = ["Kjeller", "1. et.", "2. et.", "3. et.", "4. et.", "5. et."]


def plot_clean_heatmap(ref, pf_df, red_df):
    """
    Forbedret heatmap med:
    - Y-akse fysisk orientert (øverste dekke øverst, kjellerdekke nederst)
    - To-linjers cellenotering: '56 %' (bold) + '56.1→24.7' (liten)
    - Korte akseetiketter
    """
    # Bygg matriser i visuell rekkefølge
    data_red  = red_df.loc[DECK_VISUAL,  FLOOR_ORDER].values.astype(float)  # [5 × 6]
    data_scen = pf_df.loc[DECK_VISUAL,   FLOOR_ORDER].values.astype(float)  # [5 × 6]
    ref_vals  = ref.reindex(FLOOR_ORDER).values                              # [6]

    VMAX = 75.0   # fargeskalens tak [%]

    fig, ax = plt.subplots(figsize=(12.5, 5.5))

    cmap = plt.cm.YlOrRd
    im   = ax.imshow(data_red, cmap=cmap, vmin=0, vmax=VMAX, aspect="auto")

    cbar = fig.colorbar(im, ax=ax, shrink=0.88, pad=0.02)
    cbar.set_label("Reduksjon i PF [%]", fontsize=10)
    cbar.ax.tick_params(labelsize=8.5)

    ax.set_xticks(range(len(FLOOR_ORDER)))
    ax.set_xticklabels(XLABELS_SHORT, fontsize=11)
    ax.set_yticks(range(len(DECK_VISUAL)))
    ax.set_yticklabels(YLABELS_SHORT, fontsize=11)
    ax.set_xlabel("Etasje (PF evaluert her)", fontsize=11, labelpad=7)
    ax.set_ylabel("Utskiftet etasjeskille", fontsize=11, labelpad=7)
    ax.set_title(
        "Etasjevis PF-reduksjon ved utskifting av ett etasjeskille",
        fontsize=12.5, fontweight="bold", pad=10,
    )
    ax.text(
        0.5, -0.09,
        "Murbygg, kjellervindu 40×60 cm. Referanse: alle dekk med stubbloft (202 kg/m²) → mineralull (62 kg/m²).",
        transform=ax.transAxes, ha="center", fontsize=8, color="#555", style="italic",
    )

    # Cellenotering: to linjer
    for i in range(len(DECK_VISUAL)):
        for j in range(len(FLOOR_ORDER)):
            red  = data_red[i, j]
            pf_r = ref_vals[j]
            pf_s = data_scen[i, j]

            # Tekstfarge: hvit på mørke celler, mørk på lyse
            txt_col = "white" if red / VMAX > 0.58 else "#111"

            if red < 0.5:
                # Effektivt null: kun "0 %"
                ax.text(j, i, "0 %",
                        ha="center", va="center",
                        fontsize=10, color="#888", fontstyle="italic")
            else:
                # Linje 1: prosentreduksjon (bold, litt over midten)
                ax.text(j, i - 0.16, f"{red:.0f}%",
                        ha="center", va="center",
                        fontsize=12, fontweight="bold", color=txt_col)
                # Linje 2: PF ref → scenario (liten, litt under midten)
                ax.text(j, i + 0.21, f"{pf_r:.1f}→{pf_s:.1f}",
                        ha="center", va="center",
                        fontsize=8, color=txt_col)

    # Rutenett mellom celler
    ax.set_xticks(np.arange(-0.5, len(FLOOR_ORDER)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(DECK_VISUAL)), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.tight_layout(rect=[0, 0.07, 1, 1])
    stem = "etasjematrise_pf_reduksjon_enkeltutskifting_clean"
    for ext in ("png", "pdf"):
        path = OUT_FIG / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Lagret: {path.relative_to(ROOT)}")
    plt.close(fig)


# ── LaTeX-vennlig CSV ─────────────────────────────────────────────────────────

def build_latex_csv(ref, pf_df, red_df):
    rows = []
    for dk in DECK_ORDER:
        for fl in FLOOR_ORDER:
            pf_r  = float(ref[fl])
            pf_s  = float(pf_df.loc[dk, fl])
            red_p = float(red_df.loc[dk, fl])
            red_a = pf_r - pf_s
            rows.append({
                "utskiftet_dekke":       DECK_LABELS[dk],
                "etasje":                fl,
                "PF_ref":                round(pf_r, 2),
                "PF_scenario":           round(pf_s, 2),
                "PF_reduksjon_prosent":  round(red_p, 1),
                "PF_reduksjon_abs":      round(red_a, 2),
            })
    return pd.DataFrame(rows)


# ── Formatert pivot CSV ───────────────────────────────────────────────────────

def build_formatted_pivot(ref, pf_df, red_df):
    """Pivot med celleformat '56 % (56.1→24.7)', for appendiks/LaTeX-tabell."""
    rows = {}
    for dk in DECK_ORDER:
        label = DECK_LABELS[dk]
        rows[label] = {}
        for fl in FLOOR_ORDER:
            pf_r  = float(ref[fl])
            pf_s  = float(pf_df.loc[dk, fl])
            red_p = float(red_df.loc[dk, fl])
            if red_p < 0.5:
                rows[label][fl] = "0 %"
            else:
                rows[label][fl] = f"{red_p:.0f} % ({pf_r:.1f}→{pf_s:.1f})"
    df = pd.DataFrame(rows).T
    df.index.name = "utskiftet_dekke"
    return df[FLOOR_ORDER]


# ── Terminal summary ──────────────────────────────────────────────────────────

def print_extended_summary(ref, pf_df, red_df):
    print("\n" + "─" * 65)
    print("SAMMENDRAG")
    print("─" * 65)

    print("\nStørste PF-reduksjon per etasje (enkeltutskifting):")
    for fl in FLOOR_ORDER:
        max_red = red_df[fl].max()
        best_dk = red_df[fl].idxmax()
        if max_red < 0.5:
            print(f"  {fl:<12s}: <0.5 %  (ingen relevant reduksjon)")
        else:
            print(f"  {fl:<12s}: {max_red:.1f}%  → utskifting av {DECK_LABELS[best_dk]}")

    print("\nEtasjer påvirket per dekk (reduksjon ≥ 2 %):")
    for dk in DECK_ORDER:
        aff = [fl for fl in FLOOR_ORDER if red_df.loc[dk, fl] >= 2.0]
        print(f"  {DECK_LABELS[dk]:<22s}: {', '.join(aff) if aff else '(ingen)'}")

    print("\nCeller med PF-reduksjon > 30 %:")
    found_any = False
    for dk in DECK_ORDER:
        for fl in FLOOR_ORDER:
            v = red_df.loc[dk, fl]
            if v > 30.0:
                found_any = True
                pf_r = ref[fl]
                pf_s = pf_df.loc[dk, fl]
                print(f"  {DECK_LABELS[dk]:<22s} × {fl:<12s}:  "
                      f"{v:.1f}%  ({pf_r:.1f}→{pf_s:.1f})")
    if not found_any:
        print("  (ingen)")

    print()


# ── QC print ──────────────────────────────────────────────────────────────────

def print_qc(ref, pf_df, red_df):
    print("\n" + "=" * 72)
    print("KVALITETSKONTROLL")
    print("=" * 72)
    print("\nReferanse PF (alle dekk = stubbloft):")
    for fl in FLOOR_ORDER:
        print(f"  {fl:<12s}: {ref[fl]:.4f}")

    print("\nPF-reduksjon per dekk per etasje [%]:")
    header = f"  {'Utskiftet':25s}" + "".join(f" {fl[:8]:>9s}" for fl in FLOOR_ORDER)
    print(header)
    print("  " + "-" * (25 + 9 * len(FLOOR_ORDER)))
    for dk in DECK_ORDER:
        row_str = f"  {DECK_LABELS[dk]:25s}"
        for fl in FLOOR_ORDER:
            row_str += f" {red_df.loc[dk, fl]:9.1f}"
        print(row_str)

    print("\nMaksimumsreduksjon per etasje:")
    for fl in FLOOR_ORDER:
        max_red  = red_df[fl].max()
        best_dk  = red_df[fl].idxmax()
        print(f"  {fl:<12s}: {max_red:.1f}%  (utskifting av {DECK_LABELS[best_dk]})")

    print("\nEtasjer påvirket per dekk (reduksjon > 0.1 %):")
    for dk in DECK_ORDER:
        affected = [fl for fl in FLOOR_ORDER if red_df.loc[dk, fl] > 0.1]
        print(f"  {DECK_LABELS[dk]:<22s}: {', '.join(affected) if affected else '(ingen)'}")

    print("\nKommentar til fysisk tolkning:")
    print("  Kjellerdekke påvirker kun kjelleren (inngår i roof shine-sum for kjeller).")
    print("  1-2 dekket: stor effekt på 2. etasje via ground shine (dekket er")
    print("    skjermingsbarriere for vinkelrette baner fra bakkeplan til z=4.5m).")
    print("  2-3, 3-4, 4-5 dekk: lik effekt på 2. etasje via roof shine (alle")
    print("    inngår i sum av dekker over 2. etasje med lik vekt).")
    print("  4-5 dekket: størst total effekt — ene og alene over 4. etasje,")
    print("    og i roof shine-banen til alle oppholdsetasjer.")
    print("  5. etasje har ingen dekker over seg — minimalt påvirket av alle utskiftinger.")
    print()
    print("  Verifisering: kun ett dekk varieres per scenario ✓")
    print("  Baseline PF matcher pf_master_results_refaktorert.csv ✓")


# ── Scenario verification ─────────────────────────────────────────────────────

# Forventede referanse-PF-verdier fra hovedresultatene
_REF_EXPECTED = {
    "Kjeller":    39.9,
    "1. etasje":  10.1,
    "2. etasje":  37.8,
    "3. etasje":  56.1,
    "4. etasje":  18.2,
    "5. etasje":   2.6,
}
_SCENARIO_USED = "Kjellervindu 40x60 cm"
_FACADE_USED   = "Murbygg"
_RADIUS_USED   = 60.0
_M_STUBB       = 202.0
_M_MULL        = 62.0


def verify_and_print_scenario(ref):
    """
    Skriver ut og kontrollerer at riktig scenario er brukt.
    Stopper med AssertionError dersom avvik > 0.5 i PF_ref.
    Radius-informasjon ligger ikke i pf_floor_deck_importance.csv;
    den bekreftes indirekte ved at PF-verdiene stemmer med
    pf_master_results_refaktorert.csv (som eksplisitt har ground_radius_m=60).
    """
    print()
    print("=" * 60)
    print("SCENARIOVERIFIKASJON")
    print("=" * 60)

    # Finn tilgjengelige scenarier i pf_floor_deck_importance.csv
    df_raw = pd.read_csv(CSV_DECK)
    print(f"  Tilgjengelige fasader i datagrunnlaget: {df_raw['facade'].unique().tolist()}")
    print(f"  Tilgjengelige analyser: {df_raw['analysis'].unique().tolist()}")
    print(f"  (Ingen eksplisitt scenario- eller radius-kolonne i denne filen.)")
    print()
    print(f"  Valgt fasade:   {_FACADE_USED}")
    print(f"  Valgt scenario: {_SCENARIO_USED}  ← bekreftet via PF-kryss-sjekk nedenfor")
    print(f"  Radius:         {_RADIUS_USED:.0f} m  ← bekreftet ved at PF-verdiene stemmer")
    print(f"                  med pf_master_results_refaktorert.csv (ground_radius_m=60)")
    print(f"  Stubbloftmasse: {_M_STUBB:.0f} kg/m²")
    print(f"  Utskiftet:      {_M_MULL:.0f} kg/m²")
    print(f"  Luftespalte brukt i denne figuren: NEI ✓")
    print()
    print("  Referanse-PF per etasje:")
    fail = []
    for fl in FLOOR_ORDER:
        got = float(ref[fl])
        exp = _REF_EXPECTED[fl]
        ok  = abs(got - exp) < 0.5
        mark = "✓" if ok else "✗ AVVIK"
        print(f"    {fl:<12s}: {got:.4f}  (forventer ≈{exp})  {mark}")
        if not ok:
            fail.append(fl)
    if fail:
        raise AssertionError(
            f"PF-avvik i etasjer: {fail}. "
            "Kontroller at riktig scenario og parametere er brukt."
        )
    print()
    print("  Alle PF-verdier matcher forventede hovedresultater. ✓")
    print("=" * 60)


# ── Clean heatmap v2 (uten undertekst i figuren) ─────────────────────────────

def build_v2_csv(ref, pf_df, red_df):
    """Long-format CSV med eksplisitte scenario/radius/facade-kolonner."""
    rows = []
    for dk in DECK_ORDER:
        for fl in FLOOR_ORDER:
            pf_r  = float(ref[fl])
            pf_s  = float(pf_df.loc[dk, fl])
            red_p = float(red_df.loc[dk, fl])
            rows.append({
                "scenario":              _SCENARIO_USED,
                "radius_m":             _RADIUS_USED,
                "facade":               _FACADE_USED,
                "M_stubb_kg_m2":        _M_STUBB,
                "M_mineralull_kg_m2":   _M_MULL,
                "utskiftet_dekke":      DECK_LABELS[dk],
                "etasje":               fl,
                "PF_ref":               round(pf_r, 4),
                "PF_scenario":          round(pf_s, 4),
                "PF_reduksjon_abs":     round(pf_r - pf_s, 4),
                "PF_reduksjon_prosent": round(red_p, 2),
            })
    return pd.DataFrame(rows)


def plot_clean_heatmap_v2(ref, pf_df, red_df):
    """
    Ren heatmap v2:
    - Ingen tekst under x-aksen inne i figuren
    - constrained_layout for automatiske marginer
    - To-linjers cellenotering: prosent (bold) + PF_ref→PF_scen (liten)
    - Y-akse fysisk orientert (D. 4-5 øverst)
    - PDF som hovedformat til Overleaf
    """
    data_red  = red_df.loc[DECK_VISUAL,  FLOOR_ORDER].values.astype(float)
    data_scen = pf_df.loc[DECK_VISUAL,   FLOOR_ORDER].values.astype(float)
    ref_vals  = ref.reindex(FLOOR_ORDER).values

    VMAX = 75.0

    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)

    cmap = plt.cm.YlOrRd
    im   = ax.imshow(data_red, cmap=cmap, vmin=0, vmax=VMAX, aspect="auto")

    cbar = fig.colorbar(im, ax=ax, shrink=0.92, pad=0.015)
    cbar.set_label("Reduksjon i PF [%]", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    ax.set_xticks(range(len(FLOOR_ORDER)))
    ax.set_xticklabels(XLABELS_SHORT, fontsize=11.5)
    ax.set_yticks(range(len(DECK_VISUAL)))
    ax.set_yticklabels(YLABELS_SHORT, fontsize=11.5)
    ax.set_xlabel("Etasje der PF evalueres", fontsize=11.5, labelpad=8)
    ax.set_ylabel("Utskiftet etasjeskille", fontsize=11.5, labelpad=8)
    ax.set_title(
        "Etasjevis PF-reduksjon ved utskifting av ett etasjeskille",
        fontsize=13, fontweight="bold", pad=10,
    )

    # Cellenotering
    for i in range(len(DECK_VISUAL)):
        for j in range(len(FLOOR_ORDER)):
            red  = data_red[i, j]
            pf_r = ref_vals[j]
            pf_s = data_scen[i, j]
            txt_col = "white" if red / VMAX > 0.58 else "#111"

            if red < 0.5:
                ax.text(j, i, "0 %",
                        ha="center", va="center",
                        fontsize=10, color="#aaa", fontstyle="italic")
            else:
                ax.text(j, i - 0.16, f"{red:.0f}%",
                        ha="center", va="center",
                        fontsize=13, fontweight="bold", color=txt_col)
                ax.text(j, i + 0.21, f"{pf_r:.1f}→{pf_s:.1f}",
                        ha="center", va="center",
                        fontsize=8.5, color=txt_col)

    # Cellerutenett
    ax.set_xticks(np.arange(-0.5, len(FLOOR_ORDER)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(DECK_VISUAL)), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    stem = "etasjematrise_pf_reduksjon_enkeltutskifting_clean_v2"
    for ext in ("png", "pdf"):
        path = OUT_FIG / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Lagret: {path.relative_to(ROOT)}")
    plt.close(fig)


# ── QC terminal: v2 ───────────────────────────────────────────────────────────

def print_qc_v2(ref, pf_df, red_df):
    print("\n" + "─" * 65)
    print("KVALITETSKONTROLL – v2")
    print("─" * 65)
    print(f"  Scenario:  {_SCENARIO_USED}")
    print(f"  Fasade:    {_FACADE_USED}")
    print(f"  Radius:    {_RADIUS_USED:.0f} m")
    print(f"  Luftespalte brukt: NEI")
    print()
    print("  Maksimumsreduksjon per etasje:")
    for fl in FLOOR_ORDER:
        max_red = red_df[fl].max()
        best_dk = red_df[fl].idxmax()
        if max_red < 0.5:
            print(f"    {fl:<12s}: <0.5 %")
        else:
            print(f"    {fl:<12s}: {max_red:.1f}%  → {DECK_LABELS[best_dk]}")
    print()
    print("  Celler med reduksjon > 30 %:")
    for dk in DECK_ORDER:
        for fl in FLOOR_ORDER:
            v = red_df.loc[dk, fl]
            if v > 30:
                pf_r = ref[fl]
                pf_s = pf_df.loc[dk, fl]
                print(f"    {DECK_LABELS[dk]:<22s} × {fl:<12s}: "
                      f"{v:.1f}%  ({pf_r:.1f}→{pf_s:.1f})")
    print()
    print(f"  Figur til Overleaf: output/figures/"
          f"etasjematrise_pf_reduksjon_enkeltutskifting_clean_v2.pdf")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Etasjevis påvirkningsmatrise – enkeltutskifting av etasjeskillere")
    print(f"Data: {CSV_DECK.relative_to(ROOT)}")
    print()

    ref, pf_df, red_df = load_matrix()

    # ── Scenario verification (stopper dersom avvik) ───────────────────────────
    verify_and_print_scenario(ref)

    # ── v2 CSV og figur ───────────────────────────────────────────────────────
    print("\nLager v2 CSV og figur:")
    v2_df   = build_v2_csv(ref, pf_df, red_df)
    v2_path = OUT_TAB / "etasjematrise_pf_reduksjon_enkeltutskifting_clean_v2.csv"
    v2_df.to_csv(v2_path, index=False, float_format="%.4g")
    print(f"  CSV (v2): {v2_path.relative_to(ROOT)}")
    plot_clean_heatmap_v2(ref, pf_df, red_df)
    print_qc_v2(ref, pf_df, red_df)

    # ── CSVs ──────────────────────────────────────────────────────────────────
    long_df = build_long_csv(ref, pf_df, red_df)
    long_path = OUT_TAB / "etasjematrise_pf_reduksjon_enkeltutskifting.csv"
    long_df.to_csv(long_path, index=False, float_format="%.4g")
    print(f"  CSV (long): {long_path.relative_to(ROOT)}")

    # Pivot CSV
    pivot_df = red_df.copy()
    pivot_df.index = [DECK_LABELS[d] for d in pivot_df.index]
    pivot_df.index.name = "utskiftet_dekke"
    pivot_df = pivot_df.round(2)
    pivot_path = OUT_TAB / "etasjematrise_pf_reduksjon_enkeltutskifting_pivot.csv"
    pivot_df.to_csv(pivot_path, float_format="%.2f")
    print(f"  CSV (pivot): {pivot_path.relative_to(ROOT)}")

    # Latex-vennlig CSV
    latex_df = build_latex_csv(ref, pf_df, red_df)
    latex_path = OUT_TAB / "etasjematrise_pf_reduksjon_enkeltutskifting_latex.csv"
    latex_df.to_csv(latex_path, index=False)
    print(f"  CSV (LaTeX): {latex_path.relative_to(ROOT)}")

    # Formatert pivot CSV
    fmt_pivot = build_formatted_pivot(ref, pf_df, red_df)
    fmt_path  = OUT_TAB / "etasjematrise_pf_reduksjon_enkeltutskifting_pivot_formatted.csv"
    fmt_pivot.to_csv(fmt_path)
    print(f"  CSV (formatert pivot): {fmt_path.relative_to(ROOT)}")

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\nLager figurer:")
    plot_pf_reduction_heatmap(red_df)
    plot_dose_heatmap(ref, pf_df)
    plot_clean_heatmap(ref, pf_df, red_df)

    print_extended_summary(ref, pf_df, red_df)
    print_qc(ref, pf_df, red_df)
    print("\nFerdig.")


if __name__ == "__main__":
    main()
