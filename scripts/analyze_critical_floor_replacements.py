"""
Marginalanalyse: kritiske etasjeskiller ved bevaring av stubbloft.

Analysen svarer på:
  1. Hvilken enkelt etasjeskille gir størst PF-reduksjon dersom den skiftes
     fra stubbloft (202 kg/m²) til mineralull (62 kg/m²)?
  2. Hvilke kombinasjoner av to utskiftede etasjeskillere gir størst / minst
     PF-reduksjon?
  3. Hvor er det mest kritisk å bevare stubbloftsleire dersom bare noen
     etasjeskillere kan bevares?

Bygningskonfigurasjon:
  Murbygg, Scenario: Kjellervindu 40×60 cm
  Alle parametre identiske med runner.py-konfigurasjonen.
  Kun etasjeskillermassen varieres per scenario.

Referanse: alle 5 dekk = stubbloft bevart (202 kg/m²)
Utskifting: ett eller to dekk settes til 62 kg/m², resten forblir 202 kg/m²

PF_bygg = aritmetisk middel av PF for 1.–4. etasje.
Begrunnelse: representerer beskyttet oppholdsareal i vanlige oppholdsetasjer.
Kjeller og 5. etasje rapporteres separat; disse styres i stor grad av
egne mekanismer (kjelleråpninger og roof shine dominans).

Illustrativt dosescenario (H_ute = 50 mSv / 48 t):
  H_inne = H_ute / PF_bygg  — lineær skalering, ingen dosekonvertering.
  Dette er ikke en ulykkesmodell. Se doseavvik_sip-analysen for ICRP103-kontekst.

Endrer ikke: PF-modell, fasade, tak, kjelleråpning, radius, buildup
eller geometri. Kun etteranalyse med variasjon av enkeltdekkmasser.
"""

import sys
import time
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from stubbloft_pf.calculations import calculate_rows_for_points
from stubbloft_pf.config import Building
from stubbloft_pf.geometry import default_five_points_for_building
from stubbloft_pf.scenarios import M_FULL_UTSKIFTING, M_STUBB_BEVART

# ── Constants ─────────────────────────────────────────────────────────────────
SCENARIO_NAME = "Kjellervindu 40x60 cm"
M_STUBB: float = M_STUBB_BEVART   # 202 kg/m²
M_MULL:  float = M_FULL_UTSKIFTING # 62 kg/m²
N_DECKS: int   = 5                 # b.floors

# Etasjeskillere (dekk) — indeks 0 = kjellerdekke, opp til indeks 4
DECK_NAMES = [
    "Kjellerdekke",      # mellom kjeller og 1. etasje
    "Dekke 1.–2. et.",   # mellom 1. og 2. etasje
    "Dekke 2.–3. et.",   # mellom 2. og 3. etasje
    "Dekke 3.–4. et.",   # mellom 3. og 4. etasje
    "Dekke 4.–5. et.",   # mellom 4. og 5. etasje
]

# Etasjeindekser som inngår i PF_bygg (1.–4. etasje)
PF_BYGG_FLOOR_IDX = [0, 1, 2, 3]

# Illustrativt utendørs dosescenario (lineær skalering av PF — ingen ulykkesmodell)
H_UTE_mSv: float = 50.0

OUTPUT_FIGURES = ROOT / "output" / "figures"
OUTPUT_TABLES  = ROOT / "output" / "tables"
OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)

# ── Modellkonfigurasjon (identisk med runner.py) ──────────────────────────────
BASE = Building(
    basement_depth_m=2.2,
    basement_exposed_wall_height_m=0.60,
    ground_radius_m=60.0,
    use_buildup=True,
    buildup_alpha=0.10,
    use_height_dependent_windows=True,
    roof_model="sloped",
    roof_pitch_deg=45.0,
)

MURBYGG_BASE = replace(
    BASE,
    basement_opening_width_m=0.40,
    basement_opening_height_m=0.60,
    M_window_kg_m2=20.0,
    M_wall_kg_m2=450.0,
    M_foundation_wall_kg_m2=650.0,
)

POINTS    = default_five_points_for_building(BASE)
POINT_NAME = "5 målepunkter i rommet"

plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})

# ── Calculation helpers ───────────────────────────────────────────────────────

def run_scenario(label: str, masses: list[float]) -> dict[int, float]:
    """Kjører PF-beregning for ett massescenario. Returnerer dict {floor_idx: PF}."""
    b = replace(MURBYGG_BASE, name=label, floor_masses_kg_m2=tuple(masses))
    rows = calculate_rows_for_points(SCENARIO_NAME, POINT_NAME, POINTS, b)
    return {r.floor: r.PF for r in rows}


def pf_metrics(pf: dict[int, float]) -> dict:
    """Beregner sammendragsmetrikker fra en floor→PF dict."""
    vals_1to4 = [pf[k] for k in PF_BYGG_FLOOR_IDX]
    bygg = float(np.mean(vals_1to4))
    return {
        "PF_bygg":           round(bygg, 4),
        "PF_3etasje":        round(pf[2], 4),
        "PF_kjeller":        round(pf[-1], 4),
        "PF_5etasje":        round(pf[4], 4),
        "minimum_PF_1til4":  round(float(min(vals_1to4)), 4),
    }


def dose_metrics(m: dict, ref_m: dict) -> dict:
    """Illustrativ dosekonsekvens ved H_ute = 50 mSv (lineær skalering av PF_bygg)."""
    h_in   = H_UTE_mSv / m["PF_bygg"]
    h_in_r = H_UTE_mSv / ref_m["PF_bygg"]
    return {
        "H_inne_bygg_50mSv": round(h_in, 4),
        "ekstra_dose_50mSv": round(h_in - h_in_r, 4),
    }


def reductions(m: dict, ref_m: dict) -> dict:
    r_bygg_abs = ref_m["PF_bygg"] - m["PF_bygg"]
    r_bygg_pct = 100.0 * r_bygg_abs / ref_m["PF_bygg"]
    r_3et_pct  = 100.0 * (ref_m["PF_3etasje"] - m["PF_3etasje"]) / ref_m["PF_3etasje"]
    return {
        "reduksjon_PF_bygg_abs":     round(r_bygg_abs, 4),
        "reduksjon_PF_bygg_prosent": round(r_bygg_pct, 2),
        "reduksjon_PF_3etasje_prosent": round(r_3et_pct, 2),
    }


# ── Run all scenarios ─────────────────────────────────────────────────────────

def run_all() -> tuple[dict, list[dict], list[dict]]:
    t0 = time.time()

    # ── Referanse: alle dekk = stubbloft ──────────────────────────────────────
    print("Kjører referansescenario (alle dekk = stubbloft)...")
    ref_pf = run_scenario("Referanse", [M_STUBB] * N_DECKS)
    ref_m  = pf_metrics(ref_pf)
    print(f"  PF_bygg_ref = {ref_m['PF_bygg']:.2f}")

    # ── Sanity check: alle dekk = mineralull ──────────────────────────────────
    print("Sanity check: alle dekk = mineralull (bør ≈ Murbygg|full utskifting)...")
    sc_pf = run_scenario("Sanity", [M_MULL] * N_DECKS)
    sc_m  = pf_metrics(sc_pf)
    # Forventet fra CSV: PF_bygg ≈ mean(6.08, 8.41, 7.58, 5.06) ≈ 6.78
    print(f"  PF_bygg_full_utsk = {sc_m['PF_bygg']:.2f}  "
          f"(forventer ≈ 6.78 fra 'Murbygg | full utskifting + mineralull')")

    # ── Enkeltutskifting: ett dekk om gangen ─────────────────────────────────
    print("\nKjører enkeltutskiftinger...")
    single_rows = []
    for k in range(N_DECKS):
        masses = [M_STUBB] * N_DECKS
        masses[k] = M_MULL
        pf  = run_scenario(f"Enkelt_k{k}", masses)
        m   = pf_metrics(pf)
        red = reductions(m, ref_m)
        dos = dose_metrics(m, ref_m)
        single_rows.append({
            "scenario_type":  "enkelt",
            "utskiftet_dekke_1": DECK_NAMES[k],
            "dekk_idx_1":     k,
            **m, **red, **dos,
        })
        print(f"  Dekk {k} ({DECK_NAMES[k]}): PF_bygg={m['PF_bygg']:.2f}  "
              f"reduksjon={red['reduksjon_PF_bygg_prosent']:.1f}%  "
              f"ekstra_dose={dos['ekstra_dose_50mSv']:.2f} mSv")

    # ── Dobbelutskifting: alle kombinasjoner av to dekk ───────────────────────
    print("\nKjører dobbelutskiftinger...")
    double_rows = []
    for k1, k2 in combinations(range(N_DECKS), 2):
        masses = [M_STUBB] * N_DECKS
        masses[k1] = M_MULL
        masses[k2] = M_MULL
        pf  = run_scenario(f"Dobbel_k{k1}k{k2}", masses)
        m   = pf_metrics(pf)
        red = reductions(m, ref_m)
        dos = dose_metrics(m, ref_m)
        double_rows.append({
            "scenario_type":  "dobbel",
            "utskiftet_dekke_1": DECK_NAMES[k1],
            "utskiftet_dekke_2": DECK_NAMES[k2],
            "dekk_idx_1":     k1,
            "dekk_idx_2":     k2,
            **m, **red, **dos,
        })
        print(f"  Dekk {k1}+{k2} ({DECK_NAMES[k1]} + {DECK_NAMES[k2]}): "
              f"PF_bygg={m['PF_bygg']:.2f}  reduksjon={red['reduksjon_PF_bygg_prosent']:.1f}%")

    elapsed = time.time() - t0
    print(f"\nAlle beregninger fullført på {elapsed:.1f} s.")
    return ref_m, single_rows, double_rows


# ── Save CSVs ─────────────────────────────────────────────────────────────────

def save_csvs(
    ref_m: dict,
    single_rows: list[dict],
    double_rows: list[dict],
) -> None:
    CSV_COLS_SINGLE = [
        "scenario_type", "utskiftet_dekke_1",
        "PF_bygg", "PF_3etasje", "PF_kjeller", "PF_5etasje", "minimum_PF_1til4",
        "reduksjon_PF_bygg_abs", "reduksjon_PF_bygg_prosent",
        "reduksjon_PF_3etasje_prosent", "H_inne_bygg_50mSv", "ekstra_dose_50mSv", "rang",
    ]
    CSV_COLS_DOUBLE = [
        "scenario_type", "utskiftet_dekke_1", "utskiftet_dekke_2",
        "PF_bygg", "PF_3etasje", "PF_kjeller", "PF_5etasje", "minimum_PF_1til4",
        "reduksjon_PF_bygg_abs", "reduksjon_PF_bygg_prosent",
        "reduksjon_PF_3etasje_prosent", "H_inne_bygg_50mSv", "ekstra_dose_50mSv", "rang",
    ]

    df_s = pd.DataFrame(single_rows).sort_values("reduksjon_PF_bygg_prosent", ascending=False)
    df_s["rang"] = range(1, len(df_s) + 1)
    df_s[CSV_COLS_SINGLE].to_csv(OUTPUT_TABLES / "kritiske_etasjeskiller_enkelt.csv",
                                 index=False, float_format="%.4g")

    df_d = pd.DataFrame(double_rows).sort_values("reduksjon_PF_bygg_prosent", ascending=False)
    df_d["rang"] = range(1, len(df_d) + 1)
    df_d[CSV_COLS_DOUBLE].to_csv(OUTPUT_TABLES / "kritiske_etasjeskiller_dobbel.csv",
                                 index=False, float_format="%.4g")

    # Summary CSV
    top1_s  = df_s.iloc[0]
    bot1_s  = df_s.iloc[-1]
    top1_d  = df_d.iloc[0]
    bot1_d  = df_d.iloc[-1]
    summary_rows = [
        {
            "type":          "referanse",
            "beskrivelse":   "Alle dekk stubbloft (referanse)",
            "PF_bygg":       ref_m["PF_bygg"],
            "reduksjon_%":   0.0,
            "ekstra_dose_mSv": 0.0,
        },
        {
            "type":          "verste_enkelt",
            "beskrivelse":   top1_s["utskiftet_dekke_1"],
            "PF_bygg":       top1_s["PF_bygg"],
            "reduksjon_%":   top1_s["reduksjon_PF_bygg_prosent"],
            "ekstra_dose_mSv": top1_s["ekstra_dose_50mSv"],
        },
        {
            "type":          "minst_alvorlig_enkelt",
            "beskrivelse":   bot1_s["utskiftet_dekke_1"],
            "PF_bygg":       bot1_s["PF_bygg"],
            "reduksjon_%":   bot1_s["reduksjon_PF_bygg_prosent"],
            "ekstra_dose_mSv": bot1_s["ekstra_dose_50mSv"],
        },
        {
            "type":          "verste_dobbel",
            "beskrivelse":   f"{top1_d['utskiftet_dekke_1']} + {top1_d['utskiftet_dekke_2']}",
            "PF_bygg":       top1_d["PF_bygg"],
            "reduksjon_%":   top1_d["reduksjon_PF_bygg_prosent"],
            "ekstra_dose_mSv": top1_d["ekstra_dose_50mSv"],
        },
        {
            "type":          "minst_alvorlig_dobbel",
            "beskrivelse":   f"{bot1_d['utskiftet_dekke_1']} + {bot1_d['utskiftet_dekke_2']}",
            "PF_bygg":       bot1_d["PF_bygg"],
            "reduksjon_%":   bot1_d["reduksjon_PF_bygg_prosent"],
            "ekstra_dose_mSv": bot1_d["ekstra_dose_50mSv"],
        },
    ]
    priority_note = "Bevaringsprioritet: " + ", ".join(
        df_s.sort_values("reduksjon_PF_bygg_prosent", ascending=False)["utskiftet_dekke_1"].tolist()
    )
    summary_rows.append({
        "type": "bevaringsprioritet",
        "beskrivelse": priority_note,
        "PF_bygg": None, "reduksjon_%": None, "ekstra_dose_mSv": None,
    })
    pd.DataFrame(summary_rows).to_csv(
        OUTPUT_TABLES / "kritiske_etasjeskiller_summary.csv", index=False
    )

    print(f"\n  CSV: {(OUTPUT_TABLES / 'kritiske_etasjeskiller_enkelt.csv').relative_to(ROOT)}")
    print(f"  CSV: {(OUTPUT_TABLES / 'kritiske_etasjeskiller_dobbel.csv').relative_to(ROOT)}")
    print(f"  CSV: {(OUTPUT_TABLES / 'kritiske_etasjeskiller_summary.csv').relative_to(ROOT)}")
    return df_s, df_d


# ── Figure 1: single replacement bar chart ───────────────────────────────────

def plot_enkelt(df_s: pd.DataFrame) -> None:
    df = df_s.sort_values("reduksjon_PF_bygg_prosent", ascending=True)
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(9, 4.5))

    bars = ax.barh(
        y, df["reduksjon_PF_bygg_prosent"].values,
        color="#2166AC", alpha=0.82, height=0.55,
        linewidth=0.5, edgecolor="white",
    )
    # Annotate bars with extra dose
    for i, (_, row) in enumerate(df.iterrows()):
        x = row["reduksjon_PF_bygg_prosent"]
        ed = row["ekstra_dose_50mSv"]
        ax.text(x + 0.4, i, f"+{ed:.2f} mSv", va="center", fontsize=8.5, color="#333")

    ax.set_yticks(y)
    ax.set_yticklabels(df["utskiftet_dekke_1"].values, fontsize=10)
    ax.set_xlabel("Reduksjon i PF₁₋₄ [%]", fontsize=10)
    ax.set_xlim(0, df["reduksjon_PF_bygg_prosent"].max() * 1.35)
    ax.set_title(
        "Kritiske etasjeskiller ved enkeltutskifting",
        fontsize=11.5, fontweight="bold", pad=8,
    )
    ax.text(
        0.01, -0.14,
        "Murbygg · Kjellervindu 40×60 cm · Referanse: alle dekk med stubbloft (202 kg/m²)\n"
        f"Annotert ekstra dose ved {H_UTE_mSv:.0f} mSv utendørs over 48 t "
        "(illustrativt, ikke en ulykkesmodell)",
        transform=ax.transAxes, fontsize=7.5, color="#555", style="italic",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.20, linewidth=0.6)

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    for ext in ("png", "pdf"):
        path = OUTPUT_FIGURES / f"kritiske_etasjeskiller_enkelt.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Lagret: {path.relative_to(ROOT)}")
    plt.close(fig)


# ── Figure 2: double replacement heatmap ─────────────────────────────────────

def plot_dobbel_heatmap(df_d: pd.DataFrame) -> None:
    matrix = np.full((N_DECKS, N_DECKS), np.nan)
    for _, row in df_d.iterrows():
        i, j = int(row["dekk_idx_1"]), int(row["dekk_idx_2"])
        v = row["reduksjon_PF_bygg_prosent"]
        matrix[i, j] = v
        matrix[j, i] = v  # symmetrisk

    short_names = [
        "Kjellerd.",
        "D. 1–2",
        "D. 2–3",
        "D. 3–4",
        "D. 4–5",
    ]

    fig, ax = plt.subplots(figsize=(7.5, 6))
    vmax = np.nanmax(matrix)
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="auto")

    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("Reduksjon PF₁₋₄ [%]", fontsize=9.5)

    ax.set_xticks(range(N_DECKS))
    ax.set_xticklabels(short_names, fontsize=9, rotation=25, ha="right")
    ax.set_yticks(range(N_DECKS))
    ax.set_yticklabels(short_names, fontsize=9)

    # Annotate cells
    for i in range(N_DECKS):
        for j in range(N_DECKS):
            v = matrix[i, j]
            if not np.isnan(v):
                txt_color = "white" if v > 0.6 * vmax else "#222"
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                        fontsize=9, color=txt_color, fontweight="bold")
            elif i == j:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=True, color="#e0e0e0", zorder=0
                ))

    ax.set_title(
        "Reduksjon i PF ved utskifting av to etasjeskiller",
        fontsize=11.5, fontweight="bold", pad=10,
    )
    ax.text(
        0.01, -0.11,
        "Murbygg · Kjellervindu 40×60 cm · Referanse: alle dekk med stubbloft\n"
        "Diagonale celler (én enkelt dekk) er ikke inkludert her (se figur 1).",
        transform=ax.transAxes, fontsize=7.5, color="#555", style="italic",
    )

    fig.tight_layout(rect=[0, 0.09, 1, 1])
    for ext in ("png", "pdf"):
        path = OUTPUT_FIGURES / f"kritiske_etasjeskiller_dobbel_heatmap.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Lagret: {path.relative_to(ROOT)}")
    plt.close(fig)


# ── Figure 3: extra dose bar chart ────────────────────────────────────────────

def plot_ekstra_dose(df_s: pd.DataFrame) -> None:
    df = df_s.sort_values("ekstra_dose_50mSv", ascending=True)
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(
        y, df["ekstra_dose_50mSv"].values,
        color="#C0622B", alpha=0.82, height=0.55,
        linewidth=0.5, edgecolor="white",
    )
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["ekstra_dose_50mSv"] + 0.02, i,
                f"{row['ekstra_dose_50mSv']:.2f} mSv",
                va="center", fontsize=8.5, color="#333")

    ax.set_yticks(y)
    ax.set_yticklabels(df["utskiftet_dekke_1"].values, fontsize=10)
    ax.set_xlabel(f"Ekstra innendørs dose [mSv]", fontsize=10)
    ax.set_xlim(0, df["ekstra_dose_50mSv"].max() * 1.35)
    ax.set_title(
        f"Ekstra innendørs dose ved enkeltutskifting",
        fontsize=11.5, fontweight="bold", pad=8,
    )
    ax.text(
        0.01, -0.14,
        f"Utendørs dose {H_UTE_mSv:.0f} mSv over 48 t — illustrativt, ikke en ulykkesmodell (ref. ICRP103)\n"
        "Ekstra dose = H_inne(mineralull) – H_inne(stubbloft)  ved H_ute = 50 mSv",
        transform=ax.transAxes, fontsize=7.5, color="#555", style="italic",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.20, linewidth=0.6)

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    for ext in ("png", "pdf"):
        path = OUTPUT_FIGURES / f"kritiske_etasjeskiller_ekstra_dose.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Lagret: {path.relative_to(ROOT)}")
    plt.close(fig)


# ── Terminal quality-control ──────────────────────────────────────────────────

def print_qc(ref_m: dict, df_s: pd.DataFrame, df_d: pd.DataFrame) -> None:
    print("\n" + "=" * 75)
    print("KVALITETSKONTROLL")
    print("=" * 75)
    print(f"\nReferanse PF_bygg       = {ref_m['PF_bygg']:.4f}")
    print(f"Referanse PF_kjeller    = {ref_m['PF_kjeller']:.4f}")
    print(f"Referanse PF_3etasje    = {ref_m['PF_3etasje']:.4f}")
    print(f"Referanse PF_5etasje    = {ref_m['PF_5etasje']:.4f}")
    print()
    print("Topp 3 mest kritiske enkeltutskiftinger:")
    for _, r in df_s.head(3).iterrows():
        print(f"  Rang {int(r.rang):1d}: {r.utskiftet_dekke_1:<22s}  "
              f"reduksjon={r.reduksjon_PF_bygg_prosent:.1f}%  "
              f"ekstra={r.ekstra_dose_50mSv:.2f} mSv")
    print()
    print("Topp 3 mest kritiske dobbelutskiftinger:")
    for _, r in df_d.head(3).iterrows():
        pair = f"{r.utskiftet_dekke_1} + {r.utskiftet_dekke_2}"
        print(f"  Rang {int(r.rang):2d}: {pair:<44s}  "
              f"reduksjon={r.reduksjon_PF_bygg_prosent:.1f}%")
    print()
    print("Verifisering:")
    print(f"  Alle dekkindekser som ble variert: {list(range(N_DECKS))} (0=kjellerdekke, 4=dekke 4-5)")
    print(f"  Fasadeparametre uendret: M_wall=450, M_fwall=650 kg/m²")
    print(f"  Takmodell uendret: sloped, 45°")
    print(f"  Kjelleråpning uendret: {MURBYGG_BASE.basement_opening_width_m:.2f}×"
          f"{MURBYGG_BASE.basement_opening_height_m:.2f} m")
    print(f"  Referanse stemmer med 'Murbygg | stubbloft bevart' (PF_bygg={ref_m['PF_bygg']:.2f})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Marginalanalyse: kritiske etasjeskiller")
    print(f"Bygning:  Murbygg  |  Scenario: {SCENARIO_NAME}")
    print(f"M_stubb = {M_STUBB:.0f} kg/m²  →  M_mull = {M_MULL:.0f} kg/m²")
    print(f"Etasjeskillere analysert: {N_DECKS}")
    print(f"Enkeltutskiftinger: {N_DECKS}  |  Dobbelutskiftinger: {N_DECKS*(N_DECKS-1)//2}\n")

    ref_m, single_rows, double_rows = run_all()

    print("\nLagrer CSV-tabeller...")
    df_s, df_d = save_csvs(ref_m, single_rows, double_rows)

    print("\nLager figurer:")
    plot_enkelt(df_s)
    plot_dobbel_heatmap(df_d)
    plot_ekstra_dose(df_s)

    print_qc(ref_m, df_s, df_d)

    print(f"\nFerdig.")


if __name__ == "__main__":
    main()
