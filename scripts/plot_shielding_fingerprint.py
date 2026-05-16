"""
Shielding fingerprint – skjermingsprofil per etasje.

For hver bygningsvariant vises:
  - Venstre panel: normalisert PF per etasje (PF / maks(PF) innenfor varianten)
  - Høyre panel:   andel innendørs dose fra ground shine vs. roof shine per etasje

Alle tall er hentet fra etasjemidlede dosekomponenter (5 målepunkter per etasje).
ground_andel + roof_andel = 1 er verifisert; se terminal-tabell for sjekk.

Output:
  figures/shielding_profiles/shielding_profile_<variant>.png/.pdf
  figures/shielding_profiles/shielding_profile_comparison.png/.pdf
  figures/shielding_profiles/shielding_profile_data.csv
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "output" / "pf_master_results_refaktorert.csv"
OUTPUT_DIR = ROOT / "figures" / "shielding_profiles"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
SCENARIO = "Kjellervindu 40x60 cm"

FLOOR_ORDER = [
    "Kjeller",
    "1. etasje",
    "2. etasje",
    "3. etasje",
    "4. etasje",
    "5. etasje",
]

# Variants: stem → (building column value, display title)
VARIANTS: dict[str, tuple[str, str]] = {
    "murbygg_stubbloft": (
        "Murbygg | stubbloft bevart",
        "Murbygg – Stubbloft bevart  (M$_{etasje}$ = 202 kg/m²)",
    ),
    "murbygg_rehabilitert": (
        "Murbygg | full utskifting + mineralull",
        "Murbygg – Full utskifting + mineralull  (M$_{etasje}$ = 62 kg/m²)",
    ),
    "trebygg_stubbloft": (
        "Trebygg | stubbloft bevart",
        "Trebygg – Stubbloft bevart  (M$_{etasje}$ = 202 kg/m²)",
    ),
    "trebygg_lett": (
        "Trebygg | referanse lett",
        "Trebygg – Referanse lett  (M$_{etasje}$ = 40 kg/m²)",
    ),
}

# Colors
C_GROUND = "#C0622B"
C_ROOF = "#3A7AB8"
C_PF = "#4A7A4A"

# Per-variant colors/styles for comparison figure
VARIANT_STYLE: dict[str, dict] = {
    "murbygg_stubbloft":   {"color": "#154A7A", "ls": "-",  "marker": "o"},
    "murbygg_rehabilitert":{"color": "#5BA3D5", "ls": "--", "marker": "s"},
    "trebygg_stubbloft":   {"color": "#8B1A1A", "ls": "-",  "marker": "^"},
    "trebygg_lett":        {"color": "#D97070", "ls": "--", "marker": "D"},
}

# ── Data loading ──────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    df = df[df["scenario"] == SCENARIO].copy()
    df["label"] = pd.Categorical(df["label"], categories=FLOOR_ORDER, ordered=True)
    return df.sort_values(["building", "label"]).reset_index(drop=True)


def compute_profile(df: pd.DataFrame, building_name: str) -> pd.DataFrame:
    """
    Returnerer DataFrame indeksert på FLOOR_ORDER med kolonner:
      PF, PF_norm, ground_andel, roof_andel, Dg_in, Dr_in, M_floor_kg_m2, M_wall_kg_m2
    """
    sub = (
        df[df["building"] == building_name]
        .set_index("label")
        .reindex(FLOOR_ORDER)
    )
    # ground_andel/roof_andel beregnes fra råkomponenter (ekvivalent med ground_frac/roof_frac)
    din = sub["Dg_in"] + sub["Dr_in"]
    sub = sub.copy()
    sub["ground_andel"] = sub["Dg_in"] / din
    sub["roof_andel"] = sub["Dr_in"] / din
    sub["PF_norm"] = sub["PF"] / sub["PF"].max()
    return sub


# ── Individual variant figure ─────────────────────────────────────────────────

def plot_variant(
    df: pd.DataFrame,
    stem: str,
    building_name: str,
    title: str,
) -> pd.DataFrame:
    sub = compute_profile(df, building_name)
    y = np.arange(len(FLOOR_ORDER))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    fig.subplots_adjust(wspace=0.06)

    # ── Venstre: normalisert PF ───────────────────────────────────────────────
    pf_norm = sub["PF_norm"].values
    ax1.barh(
        y, pf_norm,
        color=C_PF, alpha=0.80, height=0.55,
        linewidth=0.5, edgecolor="white",
    )
    ax1.set_xlim(0, 1.35)
    ax1.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax1.set_xticklabels(["0", "0.25", "0.50", "0.75", "1.00"], fontsize=9)
    ax1.set_xlabel("PF normalisert  (maks = 1)", fontsize=10)
    ax1.set_yticks(y)
    ax1.set_yticklabels(FLOOR_ORDER, fontsize=10)
    ax1.set_title("Relativ beskyttelsesfaktor\nper etasje", fontsize=10.5, pad=7)
    ax1.axvline(1.0, color="#999", linestyle="--", linewidth=0.8, alpha=0.7)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(axis="x", alpha=0.18, linewidth=0.6)

    for i, floor in enumerate(FLOOR_ORDER):
        pf = sub.loc[floor, "PF"]
        pn = sub.loc[floor, "PF_norm"]
        ax1.text(
            pn + 0.03, i,
            f"PF = {pf:.1f}",
            va="center", fontsize=8.5, color="#222",
        )

    # ── Høyre: ground / roof andel ────────────────────────────────────────────
    g = sub["ground_andel"].values
    r = sub["roof_andel"].values

    ax2.barh(y, g, color=C_GROUND, alpha=0.83, height=0.55,
             linewidth=0.5, edgecolor="white", label="Ground shine")
    ax2.barh(y, r, left=g, color=C_ROOF, alpha=0.83, height=0.55,
             linewidth=0.5, edgecolor="white", label="Roof shine")

    ax2.set_xlim(0, 1.0)
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_xticklabels(["0 %", "25 %", "50 %", "75 %", "100 %"], fontsize=9)
    ax2.set_xlabel("Andel innendørs dose", fontsize=10)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_title("Dosekilde per etasje\n(Ground shine / Roof shine)", fontsize=10.5, pad=7)
    ax2.axvline(0.5, color="#999", linestyle="--", linewidth=0.8, alpha=0.7)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="x", alpha=0.18, linewidth=0.6)
    ax2.legend(loc="lower right", fontsize=9, framealpha=0.92)

    for i, floor in enumerate(FLOOR_ORDER):
        gv = sub.loc[floor, "ground_andel"]
        rv = sub.loc[floor, "roof_andel"]
        if gv > 0.11:
            ax2.text(gv / 2, i, f"{gv*100:.0f}%",
                     va="center", ha="center", fontsize=8, color="white", fontweight="bold")
        if rv > 0.11:
            ax2.text(gv + rv / 2, i, f"{rv*100:.0f}%",
                     va="center", ha="center", fontsize=8, color="white", fontweight="bold")

    fig.suptitle(title, fontsize=11.5, fontweight="bold", y=1.02)

    for ext in ("png", "pdf"):
        path = OUTPUT_DIR / f"shielding_profile_{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Lagret: {path.relative_to(ROOT)}")
    plt.close(fig)
    return sub


# ── Comparison figure (all 4 variants, same axes) ─────────────────────────────

def plot_comparison(all_profiles: dict[str, pd.DataFrame]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    fig.subplots_adjust(wspace=0.06)

    y = np.arange(len(FLOOR_ORDER))

    for stem, sub in all_profiles.items():
        label = VARIANTS[stem][1].split("  (")[0]  # drop mass annotation
        st = VARIANT_STYLE[stem]

        ax1.plot(
            sub["PF_norm"].values, y,
            color=st["color"], linestyle=st["ls"], marker=st["marker"],
            markersize=7, linewidth=1.8, label=label,
        )
        ax2.plot(
            sub["ground_andel"].values, y,
            color=st["color"], linestyle=st["ls"], marker=st["marker"],
            markersize=7, linewidth=1.8, label=label,
        )

    # Panel A styling
    ax1.set_xlim(-0.05, 1.15)
    ax1.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax1.set_xlabel("PF normalisert  (maks = 1 per variant)", fontsize=10)
    ax1.set_yticks(y)
    ax1.set_yticklabels(FLOOR_ORDER, fontsize=10)
    ax1.set_title("Relativ beskyttelsesfaktor", fontsize=11, pad=7)
    ax1.axvline(1.0, color="#aaa", linestyle=":", linewidth=1.0)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(loc="lower right", fontsize=8.5, framealpha=0.93)
    ax1.grid(axis="x", alpha=0.20, linewidth=0.6)

    # Panel B styling
    ax2.set_xlim(-0.02, 1.05)
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_xticklabels(["0 %", "25 %", "50 %", "75 %", "100 %"])
    ax2.set_xlabel("Ground shine-andel av innendørs dose", fontsize=10)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_title("Dosekildeprofil  (ground shine-andel)", fontsize=11, pad=7)
    ax2.axvline(0.5, color="#aaa", linestyle=":", linewidth=1.0)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.legend(loc="lower right", fontsize=8.5, framealpha=0.93)
    ax2.grid(axis="x", alpha=0.20, linewidth=0.6)

    fig.suptitle(
        "Skjermingsprofil per etasje – sammenligning av bygningsvarianter\n"
        f"Scenario: {SCENARIO}",
        fontsize=12, fontweight="bold", y=1.02,
    )

    for ext in ("png", "pdf"):
        path = OUTPUT_DIR / f"shielding_profile_comparison.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Lagret: {path.relative_to(ROOT)}")
    plt.close(fig)


# ── Terminal quality-control table ────────────────────────────────────────────

def print_table(all_profiles: dict[str, pd.DataFrame]) -> None:
    w = 100
    print("\n" + "=" * w)
    hdr = (
        f"{'VARIANT':<38} {'ETASJE':<12} {'PF':>7} "
        f"{'PF_norm':>8} {'ground_andel':>13} {'roof_andel':>11} {'SUM':>7}"
    )
    print(hdr)
    print("=" * w)
    for stem, sub in all_profiles.items():
        short = VARIANTS[stem][1].split("  (")[0]
        for floor in FLOOR_ORDER:
            row = sub.loc[floor]
            sumf = row["ground_andel"] + row["roof_andel"]
            ok = "OK" if abs(sumf - 1.0) < 1e-10 else "FEIL"
            print(
                f"{short:<38} {floor:<12} {row['PF']:>7.2f} "
                f"{row['PF_norm']:>8.4f} {row['ground_andel']:>13.6f} "
                f"{row['roof_andel']:>11.6f} {sumf:>7.4f} {ok}"
            )
        print("-" * w)


# ── Save CSV ──────────────────────────────────────────────────────────────────

def save_csv(all_profiles: dict[str, pd.DataFrame]) -> None:
    rows = []
    for stem, sub in all_profiles.items():
        building_name, _ = VARIANTS[stem]
        for floor in FLOOR_ORDER:
            row = sub.loc[floor]
            rows.append(
                {
                    "variant_stem": stem,
                    "building": building_name,
                    "scenario": SCENARIO,
                    "etasje": floor,
                    "PF": round(float(row["PF"]), 4),
                    "PF_norm": round(float(row["PF_norm"]), 6),
                    "ground_andel": round(float(row["ground_andel"]), 6),
                    "roof_andel": round(float(row["roof_andel"]), 6),
                    "sum_check": round(float(row["ground_andel"] + row["roof_andel"]), 10),
                    "Dg_in": float(row["Dg_in"]),
                    "Dr_in": float(row["Dr_in"]),
                    "M_floor_kg_m2": float(row["M_floor_kg_m2"]),
                    "M_wall_kg_m2": float(row["M_wall_kg_m2"]),
                }
            )
    out = pd.DataFrame(rows)
    path = OUTPUT_DIR / "shielding_profile_data.csv"
    out.to_csv(path, index=False, float_format="%.8g")
    print(f"\n  CSV lagret: {path.relative_to(ROOT)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Leser data:  {DATA_FILE.relative_to(ROOT)}")
    print(f"Output:      {OUTPUT_DIR.relative_to(ROOT)}/")
    print(f"Scenario:    {SCENARIO}\n")

    df = load_data()

    all_profiles: dict[str, pd.DataFrame] = {}

    print("Individuelle figurer:")
    for stem, (building_name, title) in VARIANTS.items():
        print(f"\n  [{stem}]")
        sub = plot_variant(df, stem, building_name, title)
        all_profiles[stem] = sub

    print("\nSammenligningsfigur:")
    plot_comparison(all_profiles)

    print_table(all_profiles)
    save_csv(all_profiles)

    print(f"\nFerdig. Figurer i: {OUTPUT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
