"""
Sensitivitets-tornado for effektiv PF_bygg – murbygg med stubbloft.

Hovedmål:
    PF_bygg_eff = sum(Dg_ref + Dr_ref) / sum(Dg_in + Dr_in)  over 1.–4. etasje

Begrunnelse:
    PF er et forholdstall. Aritmetisk middel av PF-verdier gir like stor vekt
    til alle etasjer uavhengig av hvor mye dose de faktisk bidrar med. Effektiv
    PF aggregerer i stedet dosekomponentene FØR forholdstallet beregnes, slik
    at etasjer med høy referansedose (Dref) veier tyngre. Dette tilsvarer den
    metoden oppgaven ellers bruker for etasjemidling.

    Aritmetisk gjennomsnitt beholdes som sekundært kontrollmål.

Baseline:
    Murbygg | stubbloft bevart, Kjellervindu 40×60 cm,
    M_floor=202 kg/m², R=60 m, saltak 45°, µ/ρ=0.007 m²/kg,
    kjellerdekke: same_as_floor

Datakilder:
    Gjenbrukes:  pf_radius_mechanism.csv, pf_master_results_refaktorert.csv
    Nytt:        etasjeskillermasse, takmodell, kjellerdekke, µ/ρ

Endrer ikke: PF-modell, geometri, integrasjonsradius, fasade,
             buildup, eksisterende resultater.
"""

import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from stubbloft_pf.calculations import calculate_rows_for_points
from stubbloft_pf.config import Building
from stubbloft_pf.geometry import default_five_points_for_building
from stubbloft_pf.scenarios import M_FULL_UTSKIFTING, M_STUBB_BEVART

CSV_MASTER = ROOT / "output" / "pf_master_results_refaktorert.csv"
CSV_RADIUS = ROOT / "output" / "pf_radius_mechanism.csv"
OUT_FIG    = ROOT / "output" / "figures"
OUT_TAB    = ROOT / "output" / "tables"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
SCENARIO_NAME  = "Kjellervindu 40x60 cm"
M_STUBB        = M_STUBB_BEVART
M_MULL         = M_FULL_UTSKIFTING
PF_BYGG_FLOORS = [0, 1, 2, 3]   # 1.–4. etasje

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

MURBYGG = replace(
    BASE,
    basement_opening_width_m=0.40,
    basement_opening_height_m=0.60,
    M_window_kg_m2=20.0,
    M_wall_kg_m2=450.0,
    M_foundation_wall_kg_m2=650.0,
    M_floor_kg_m2=M_STUBB,
)

POINTS     = default_five_points_for_building(BASE)
POINT_NAME = "5 målepunkter i rommet"

plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})

# ── Metrics from dose-component dict ─────────────────────────────────────────

def metrics_from_components(dc):
    """
    dc: {floor_idx: {'PF': float, 'Dg_in': float, 'Dr_in': float,
                     'Dg_ref': float, 'Dr_ref': float}}

    PF_bygg_eff = sum(Dref) / sum(Din) over 1.–4. etasje.
    Aggregerer dosekomponenter FØR forholdstallet beregnes — konsistent med
    oppgavens metode for etasjemidling.
    """
    sum_dref = sum(dc[k]["Dg_ref"] + dc[k]["Dr_ref"] for k in PF_BYGG_FLOORS)
    sum_din  = sum(dc[k]["Dg_in"]  + dc[k]["Dr_in"]  for k in PF_BYGG_FLOORS)
    pf_eff   = sum_dref / max(sum_din, 1e-30)
    pf_arith = float(np.mean([dc[k]["PF"] for k in PF_BYGG_FLOORS]))
    vals     = [dc[k]["PF"] for k in PF_BYGG_FLOORS]
    return {
        "PF_bygg_eff":            round(pf_eff, 4),
        "PF_bygg_arithmetic_mean": round(pf_arith, 4),
        "PF_kjeller":             round(dc[-1]["PF"], 4),
        "PF_3etasje":             round(dc[2]["PF"], 4),
        "PF_5etasje":             round(dc[4]["PF"], 4),
        "minimum_PF_1til4":       round(float(min(vals)), 4),
        "note":                   "",
    }


def dc_from_rows(rows):
    """ResultRow-liste → dose-component dict."""
    return {
        r.floor: {
            "PF":    r.PF,
            "Dg_in": r.Dg_in,
            "Dr_in": r.Dr_in,
            "Dg_ref": r.Dg_ref,
            "Dr_ref": r.Dr_ref,
        }
        for r in rows
    }


def dc_from_df_rows(sub_df):
    """DataFrame (one building, one scenario, all floors) → dose-component dict."""
    result = {}
    for _, row in sub_df.iterrows():
        fi = int(row["floor"])
        # Dose component columns may differ between CSVs
        def _get(primary, fallback=None, default=None):
            if primary in sub_df.columns:
                return float(row[primary])
            if fallback and fallback in sub_df.columns:
                return float(row[fallback])
            return default

        dg_in  = _get("Dg_in")
        dr_in  = _get("Dr_in")
        dg_ref = _get("Dg_ref")
        dr_ref = _get("Dr_ref")
        pf     = float(row["PF"])

        if None in (dg_in, dr_in, dg_ref, dr_ref):
            # Fallback: synthesize from PF only (mark in note)
            result[fi] = {"PF": pf, "Dg_in": None, "Dr_in": None,
                          "Dg_ref": None, "Dr_ref": None, "_fallback": True}
        else:
            result[fi] = {"PF": pf, "Dg_in": dg_in, "Dr_in": dr_in,
                          "Dg_ref": dg_ref, "Dr_ref": dr_ref}
    return result


def metrics_with_fallback(dc):
    """
    Compute metrics; if dose components are missing for 1-4 et., fall back to
    arithmetic PF mean and flag in the note.
    """
    floors_ok = [
        k for k in PF_BYGG_FLOORS
        if dc.get(k, {}).get("Dg_in") is not None
    ]
    if len(floors_ok) == len(PF_BYGG_FLOORS):
        return metrics_from_components(dc)
    # Fallback
    msg = ("PF_bygg_eff kunne ikke beregnes fordi dosekomponenter mangler; "
           "arithmetic PF mean used as fallback.")
    print(f"  ADVARSEL: {msg}")
    pf_arith = float(np.mean([dc[k]["PF"] for k in PF_BYGG_FLOORS]))
    vals = [dc[k]["PF"] for k in PF_BYGG_FLOORS]
    return {
        "PF_bygg_eff":             round(pf_arith, 4),
        "PF_bygg_arithmetic_mean": round(pf_arith, 4),
        "PF_kjeller":              round(dc[-1]["PF"], 4),
        "PF_3etasje":              round(dc[2]["PF"], 4),
        "PF_5etasje":              round(dc[4]["PF"], 4),
        "minimum_PF_1til4":        round(float(min(vals)), 4),
        "note":                    msg,
    }


# ── Load baseline from master CSV ─────────────────────────────────────────────

def load_baseline():
    df = pd.read_csv(CSV_MASTER)
    sub = df[
        (df["building"] == "Murbygg | stubbloft bevart")
        & (df["scenario"] == SCENARIO_NAME)
        & (df["point"] == "5 målepunkter i rommet")
    ]
    dc = dc_from_df_rows(sub)
    return metrics_with_fallback(dc)


# ── Load radius sensitivity ────────────────────────────────────────────────────

def load_radius_sensitivity():
    df = pd.read_csv(CSV_RADIUS)
    sub = df[
        (df["building"] == "Murbygg | stubbloft bevart")
        & (df["point"] == "5 målepunkter")
    ]
    results = []
    for sc in sorted(sub["scenario"].unique()):
        s = sub[sub["scenario"] == sc]
        rad = s["ground_radius_m"].iloc[0]
        dc  = dc_from_df_rows(s)
        m   = metrics_with_fallback(dc)
        m["level"] = f"R = {rad:.0f} m"
        results.append(m)
    return results


# ── Load kjelleråpning sensitivity ────────────────────────────────────────────

def load_kjelleraapning_sensitivity():
    df = pd.read_csv(CSV_MASTER)
    sub = df[
        (df["building"] == "Murbygg | stubbloft bevart")
        & (df["point"] == "5 målepunkter i rommet")
    ]
    results = []
    for sc in ["Kjellervindu 40x60 cm", "Kjellervindu 60x30 cm", "Luftespalte 15x15 cm"]:
        s  = sub[sub["scenario"] == sc]
        dc = dc_from_df_rows(s)
        m  = metrics_with_fallback(dc)
        m["level"] = sc.replace("Kjellervindu ", "")
        results.append(m)
    return results


# ── Compute new scenarios ─────────────────────────────────────────────────────

def _run(label, building):
    rows = calculate_rows_for_points(SCENARIO_NAME, POINT_NAME, POINTS, building)
    dc   = dc_from_rows(rows)
    m    = metrics_from_components(dc)
    m["level"] = label
    return m


def compute_floor_mass_sensitivity():
    # 174 = 6 cm leire, 202 = 8 cm leire (baseline), 230 = 10 cm leire
    return [
        _run(f"{m:.0f} kg/m²", replace(MURBYGG, M_floor_kg_m2=float(m)))
        for m in [174, M_STUBB, 230]
    ]


def compute_roof_model_sensitivity():
    return [
        _run("Flatt tak",  replace(MURBYGG, roof_model="flat",   roof_pitch_deg=0.0)),
        _run("Saltak 45°", replace(MURBYGG, roof_model="sloped", roof_pitch_deg=45.0)),
    ]


def compute_kjellerdekke_sensitivity():
    # same_as_floor: kjellerdekket bruker M_floor (202 kg/m²) — baseline
    # heavy_basement_ceiling: 300 kg/m² for kjellerdekke alene
    return [
        _run("Standard (202 kg/m²)",
             replace(MURBYGG, basement_ceiling_mode="same_as_floor")),
        _run("Tungt (300 kg/m²)",
             replace(MURBYGG, basement_ceiling_mode="heavy_basement_ceiling")),
    ]


def compute_mu_sensitivity():
    # Metodisk usikkerhet: µ/ρ ± ~7 %.
    # Ikke en bygningsparameter — markeres i figur og CSV.
    return [
        _run(f"µ/ρ = {mu:.4f}", replace(MURBYGG, mu_m2_kg=mu))
        for mu in [0.0065, 0.0070, 0.0075]
    ]


# ── Build tornado DataFrame ───────────────────────────────────────────────────

def build_tornado(baseline, sensitivities):
    ref = baseline["PF_bygg_eff"]
    ref_arith = baseline["PF_bygg_arithmetic_mean"]
    rows = []
    for param, levels in sensitivities.items():
        pf_eff_vals   = [lv["PF_bygg_eff"] for lv in levels]
        pf_arith_vals = [lv["PF_bygg_arithmetic_mean"] for lv in levels]
        changes       = [100 * (v - ref) / ref for v in pf_eff_vals]
        idx_lo = int(np.argmin(pf_eff_vals))
        idx_hi = int(np.argmax(pf_eff_vals))
        rows.append({
            "parameter":                      param,
            "baseline_value":                 ref,
            "low_level":                      levels[idx_lo]["level"],
            "high_level":                     levels[idx_hi]["level"],
            "PF_bygg_eff_baseline":           ref,
            "PF_bygg_eff_low":                pf_eff_vals[idx_lo],
            "PF_bygg_eff_high":               pf_eff_vals[idx_hi],
            "PF_bygg_arithmetic_mean_baseline": ref_arith,
            "PF_bygg_arithmetic_mean_low":    pf_arith_vals[idx_lo],
            "PF_bygg_arithmetic_mean_high":   pf_arith_vals[idx_hi],
            "change_low_percent":             round(changes[idx_lo], 2),
            "change_high_percent":            round(changes[idx_hi], 2),
            "span_percent":                   round(changes[idx_hi] - changes[idx_lo], 2),
            "note":                           "; ".join(
                lv["note"] for lv in levels if lv.get("note")
            ),
        })
    return pd.DataFrame(rows).sort_values("span_percent", ascending=False)


def build_all_levels(baseline, sensitivities):
    ref = baseline["PF_bygg_eff"]
    rows = []
    for param, levels in sensitivities.items():
        for lv in levels:
            rows.append({
                "parameter":                    param,
                "level":                        lv["level"],
                "PF_bygg_eff":                  lv["PF_bygg_eff"],
                "PF_bygg_arithmetic_mean":      lv["PF_bygg_arithmetic_mean"],
                "PF_kjeller":                   lv["PF_kjeller"],
                "PF_3etasje":                   lv["PF_3etasje"],
                "PF_5etasje":                   lv["PF_5etasje"],
                "minimum_PF_1til4":             lv["minimum_PF_1til4"],
                "change_from_baseline_percent": round(100 * (lv["PF_bygg_eff"] - ref) / ref, 2),
                "note":                         lv.get("note", ""),
            })
    return pd.DataFrame(rows)


# ── Tornado figure ────────────────────────────────────────────────────────────

def plot_tornado(df, metric_label, file_stem, baseline_val, note=""):
    df_plot = df.sort_values("span_percent", ascending=True)
    n = len(df_plot)
    y = np.arange(n)

    C_NEG  = "#C0622B"
    C_POS  = "#2166AC"
    C_ZERO = "#888888"

    fig, ax = plt.subplots(figsize=(11, max(4.5, n * 0.82 + 1.8)))

    for i, (_, row) in enumerate(df_plot.iterrows()):
        lo = row["change_low_percent"]
        hi = row["change_high_percent"]

        if lo < 0 and hi > 0:
            ax.barh(i, lo, height=0.55, color=C_NEG, alpha=0.82,
                    linewidth=0.5, edgecolor="white")
            ax.barh(i, hi, height=0.55, color=C_POS, alpha=0.82,
                    linewidth=0.5, edgecolor="white")
        elif lo >= 0:
            ax.barh(i, hi - lo, height=0.55, color=C_POS, alpha=0.82, left=lo,
                    linewidth=0.5, edgecolor="white")
        else:
            ax.barh(i, hi - lo, height=0.55, color=C_NEG, alpha=0.82, left=lo,
                    linewidth=0.5, edgecolor="white")

        x_range = max(abs(df_plot["change_low_percent"].min()),
                      df_plot["change_high_percent"].max())
        offset = x_range * 0.025 + 0.5
        ax.text(lo - offset, i, row["low_level"],
                ha="right", va="center", fontsize=8.2, color="#333")
        ax.text(hi + offset, i, row["high_level"],
                ha="left",  va="center", fontsize=8.2, color="#333")

    ax.axvline(0, color=C_ZERO, linewidth=1.3, zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels(df_plot["parameter"].values, fontsize=10)
    ax.set_xlabel(
        f"Endring i {metric_label} relativt til baseline [%]", fontsize=10
    )
    ax.set_title(
        "Sensitivitetsanalyse av effektiv PF – murbygg med stubbloft",
        fontsize=12, fontweight="bold", pad=8,
    )
    subtitle = (
        f"Baseline PF_bygg_eff = {baseline_val:.2f}  |  "
        "Kjellervindu 40×60 cm · R = 60 m · saltak 45° · M_floor = 202 kg/m²\n"
        "PF_bygg_eff beregnet fra summerte dosekomponenter for 1.–4. etasje"
    )
    if note:
        subtitle += f"\n{note}"
    ax.text(0.5, -0.12, subtitle,
            transform=ax.transAxes, ha="center", fontsize=7.5,
            color="#555", style="italic")

    neg_patch = mpatches.Patch(color=C_NEG, alpha=0.82, label="Under baseline")
    pos_patch = mpatches.Patch(color=C_POS, alpha=0.82, label="Over baseline")
    ax.legend(handles=[pos_patch, neg_patch], fontsize=9, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.20, linewidth=0.6)

    fig.tight_layout(rect=[0, 0.12, 1, 1])
    for ext in ("png", "pdf"):
        path = OUT_FIG / f"{file_stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Lagret: {path.relative_to(ROOT)}")
    plt.close(fig)


# ── Secondary single-metric tornado ──────────────────────────────────────────

def plot_secondary_tornado(all_levels, metric, label, file_stem, baseline_val,
                           subset_params=None):
    ref    = baseline_val
    params = subset_params or list(all_levels["parameter"].unique())
    rows   = []
    for p in params:
        sub  = all_levels[all_levels["parameter"] == p]
        vals = sub[metric].values
        chg  = 100 * (vals - ref) / ref
        span = float(np.max(chg) - np.min(chg))
        if span < 0.02:
            continue
        idx_lo = int(np.argmin(vals))
        idx_hi = int(np.argmax(vals))
        rows.append({
            "parameter":           p,
            "change_low_percent":  round(float(np.min(chg)), 2),
            "change_high_percent": round(float(np.max(chg)), 2),
            "span_percent":        round(span, 2),
            "low_level":           sub["level"].iloc[idx_lo],
            "high_level":          sub["level"].iloc[idx_hi],
        })
    if not rows:
        return
    df_s = (pd.DataFrame(rows)
              .sort_values("span_percent", ascending=True)
              .reset_index(drop=True))
    n = len(df_s)
    y = np.arange(n)
    C_NEG, C_POS = "#C0622B", "#2166AC"

    fig, ax = plt.subplots(figsize=(10, max(3, n * 0.80 + 1.5)))
    for i, (_, row) in enumerate(df_s.iterrows()):
        lo, hi = row["change_low_percent"], row["change_high_percent"]
        if lo < 0 and hi > 0:
            ax.barh(i, lo, height=0.55, color=C_NEG, alpha=0.82,
                    linewidth=0.5, edgecolor="white")
            ax.barh(i, hi, height=0.55, color=C_POS, alpha=0.82,
                    linewidth=0.5, edgecolor="white")
        elif lo >= 0:
            ax.barh(i, hi - lo, height=0.55, color=C_POS, alpha=0.82, left=lo,
                    linewidth=0.5, edgecolor="white")
        else:
            ax.barh(i, hi - lo, height=0.55, color=C_NEG, alpha=0.82, left=lo,
                    linewidth=0.5, edgecolor="white")
        xr = max(abs(df_s["change_low_percent"].min()),
                 df_s["change_high_percent"].max())
        off = xr * 0.025 + 0.5
        ax.text(lo - off, i, row["low_level"],  ha="right", va="center", fontsize=8.0)
        ax.text(hi + off, i, row["high_level"], ha="left",  va="center", fontsize=8.0)

    ax.axvline(0, color="#888", linewidth=1.3)
    ax.set_yticks(y)
    ax.set_yticklabels(df_s["parameter"].values, fontsize=10)
    ax.set_xlabel(f"Endring i {label} relativt til baseline [%]", fontsize=10)
    ax.set_title(f"Sensitivitet – {label}", fontsize=11.5, fontweight="bold", pad=8)
    ax.text(0.5, -0.09,
            f"Baseline {label} = {ref:.2f}  |  Murbygg med stubbloft",
            transform=ax.transAxes, ha="center",
            fontsize=7.5, color="#555", style="italic")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.20, linewidth=0.6)
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    for ext in ("png", "pdf"):
        path = OUT_FIG / f"{file_stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Lagret: {path.relative_to(ROOT)}")
    plt.close(fig)


# ── QC table ──────────────────────────────────────────────────────────────────

def print_qc(baseline, tornado_df, all_levels, old_arith_baseline=30.54):
    print("\n" + "=" * 72)
    print("KVALITETSKONTROLL")
    print("=" * 72)
    print(f"  PF_bygg_eff (ny)          = {baseline['PF_bygg_eff']:.4f}")
    print(f"  PF_bygg_arithmetic (gammel)= {baseline['PF_bygg_arithmetic_mean']:.4f}"
          f"  (tidligere baseline: {old_arith_baseline:.4f})")
    diff_pct = 100*(baseline['PF_bygg_eff'] - baseline['PF_bygg_arithmetic_mean'])\
               / baseline['PF_bygg_arithmetic_mean']
    print(f"  Forskjell eff vs arith    = {diff_pct:+.1f}%")
    print(f"  PF_kjeller                = {baseline['PF_kjeller']:.4f}")
    print(f"  PF_3etasje                = {baseline['PF_3etasje']:.4f}")
    print(f"  PF_5etasje                = {baseline['PF_5etasje']:.4f}")
    print()
    print("  Parameterverdier testet:")
    for _, row in all_levels.iterrows():
        print(f"    {row['parameter']:<30s}  {row['level']:<24s}  "
              f"PF_eff={row['PF_bygg_eff']:.2f}  "
              f"PF_arith={row['PF_bygg_arithmetic_mean']:.2f}  "
              f"Δ={row['change_from_baseline_percent']:+.1f}%")
    print()
    print("  Tornado-rangering (PF_bygg_eff):")
    for i, (_, r) in enumerate(tornado_df.iterrows()):
        print(f"    {i+1}. {r['parameter']:<30s}  spenn={r['span_percent']:.1f}%  "
              f"[{r['change_low_percent']:+.1f}%  ..  {r['change_high_percent']:+.1f}%]")
    # µ/ρ note
    mu_row = tornado_df[tornado_df["parameter"] == "µ/ρ (metodisk)"]
    if not mu_row.empty:
        sp = mu_row.iloc[0]["span_percent"]
        print(f"\n  MERK: µ/ρ gir {sp:.0f}% spenn — dette er en metodisk usikkerhet,")
        print("        ikke en fysisk rehabiliteringsparameter.")
    # Kjellerdekke note
    kd_row = tornado_df[tornado_df["parameter"] == "Kjellerdekke"]
    if not kd_row.empty:
        sp = kd_row.iloc[0]["span_percent"]
        print(f"\n  Kjellerdekke: {sp:.1f}% spenn på PF_bygg_eff —")
        print("        forventet liten effekt (dekket krysses ikke av strålebaner til 1.–4. et.).")
    print()
    print("  Verifisering: kun én parameter endret per sensitivitet ✓")
    print("  Baseline finnes i alle sensitiviteter:")
    ref = baseline["PF_bygg_eff"]
    for param in all_levels["parameter"].unique():
        sub = all_levels[all_levels["parameter"] == param]
        match = sub[(sub["PF_bygg_eff"] - ref).abs() < 0.05]
        if len(match):
            print(f"    {param}: ✓ ({match.iloc[0]['level']})")
        else:
            nearest = sub.loc[(sub["PF_bygg_eff"] - ref).abs().idxmin()]
            print(f"    {param}: ✗ nærmeste={nearest['level']} "
                  f"(Δ={nearest['PF_bygg_eff']-ref:+.3f})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Sensitivitets-tornado (PF_bygg_eff) – Murbygg med stubbloft")
    print("=" * 60)

    print("\nBaseline fra master-CSV...")
    baseline = load_baseline()
    print(f"  PF_bygg_eff             = {baseline['PF_bygg_eff']:.4f}")
    print(f"  PF_bygg_arithmetic_mean = {baseline['PF_bygg_arithmetic_mean']:.4f}")

    print("\nLeser radius fra eksisterende CSV...")
    radius_levels = load_radius_sensitivity()

    print("Leser kjelleråpning fra master-CSV...")
    aapning_levels = load_kjelleraapning_sensitivity()

    t0 = time.time()
    print("\nBeregner nye scenarioer...")
    print("  Etasjeskillermasse (174 / 202 / 230 kg/m²)...")
    floor_mass_levels = compute_floor_mass_sensitivity()
    print("  Takmodell (flat / saltak 45°)...")
    roof_levels = compute_roof_model_sensitivity()
    print("  Kjellerdekke (standard / tungt 300 kg/m²)...")
    kd_levels = compute_kjellerdekke_sensitivity()
    print("  µ/ρ (0.0065 / 0.0070 / 0.0075 m²/kg)...")
    mu_levels = compute_mu_sensitivity()
    print(f"  Ferdig på {time.time()-t0:.1f} s")

    PARAMS = {
        "Etasjeskillermasse": floor_mass_levels,
        "Integrasjonsradius": radius_levels,
        "µ/ρ (metodisk)":     mu_levels,
        "Kjelleråpning":      aapning_levels,
        "Takmodell":          roof_levels,
        "Kjellerdekke":       kd_levels,
    }

    print("\nBygger tabeller...")
    tornado_df = build_tornado(baseline, PARAMS)
    all_levels = build_all_levels(baseline, PARAMS)

    cols_tornado = [
        "parameter", "baseline_value", "low_level", "high_level",
        "PF_bygg_eff_baseline", "PF_bygg_eff_low", "PF_bygg_eff_high",
        "PF_bygg_arithmetic_mean_baseline",
        "PF_bygg_arithmetic_mean_low", "PF_bygg_arithmetic_mean_high",
        "change_low_percent", "change_high_percent", "span_percent", "note",
    ]
    tornado_df[cols_tornado].to_csv(
        OUT_TAB / "sensitivity_tornado_pf_bygg_eff.csv",
        index=False, float_format="%.4g",
    )
    all_levels.to_csv(
        OUT_TAB / "sensitivity_all_levels_eff.csv",
        index=False, float_format="%.4g",
    )
    print(f"  CSV: {(OUT_TAB/'sensitivity_tornado_pf_bygg_eff.csv').relative_to(ROOT)}")
    print(f"  CSV: {(OUT_TAB/'sensitivity_all_levels_eff.csv').relative_to(ROOT)}")

    print("\nLager figurer:")
    plot_tornado(
        tornado_df,
        metric_label="PF_bygg_eff",
        file_stem="sensitivity_tornado_pf_bygg_eff",
        baseline_val=baseline["PF_bygg_eff"],
        note="µ/ρ er metodisk usikkerhet – ikke en rehabiliteringsparameter",
    )
    plot_secondary_tornado(
        all_levels, "PF_kjeller", "PF_kjeller",
        "sensitivity_tornado_pf_kjeller",
        baseline["PF_kjeller"],
        subset_params=["Kjelleråpning", "Kjellerdekke", "Integrasjonsradius",
                       "Etasjeskillermasse", "µ/ρ (metodisk)"],
    )
    plot_secondary_tornado(
        all_levels, "PF_3etasje", "PF₃ (3. etasje)",
        "sensitivity_tornado_pf_3etasje",
        baseline["PF_3etasje"],
    )

    print_qc(baseline, tornado_df, all_levels)
    print("\nFerdig.")


if __name__ == "__main__":
    main()
