"""
Etteranalyse: Praktisk dosebetydning av PF-forskjell ved leirfjerning fra stubbloft.

Scriptet beregner og visualiserer innendørs akkumulert dose og doseavvik mellom
bevart stubbloft og full utskifting for et shelter-in-place (SIP) scenario over
48 timer, gitt illustrative utendørs dosescenarioer.

Faglig grunnlag:
    PF er dimensjonsløs og uavhengig av absolutt utendørs dose og tid.
    H_inne = H_ute / PF  →  ingen dosekonvertering eller Bq/m² benyttes.

ICRP103-kontekst:
    Illustrative utendørs dosescenarioer (5, 20, 50, 100 mSv / 48 timer) er valgt
    med referanse til ICRP103, som angir at sheltering tidligere ble vurdert ved
    avverget dose 5–50 mSv over 2 døgn, og at nyere referansenivåer for
    nødssituasjoner ligger i området 20–100 mSv.

    Disse verdiene er IKKE automatiske utløsningsgrenser for SIP.
    De representerer IKKE et konkret ulykkesscenario.
    De benyttes utelukkende som illustrative skaleringseksempler for å vise
    den praktiske konsekvensen av PF-differansen mellom bygningsvariantene.

Endrer ikke: PF-modell, bygningsparametre, geometri, transmisjon, buildup,
integrasjonsradius eller noen annen beregningsforutsetning.
Dette er ren etteranalyse av eksisterende resultater.
"""

import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "output" / "pf_master_results_refaktorert.csv"
OUTPUT_TABLES = ROOT / "output" / "tables"
OUTPUT_FIGURES = ROOT / "output" / "figures"
OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)

# ── Analysis constants ────────────────────────────────────────────────────────
SCENARIO = "Kjellervindu 40x60 cm"

FLOOR_ORDER = [
    "Kjeller",
    "1. etasje",
    "2. etasje",
    "3. etasje",
    "4. etasje",
    "5. etasje",
]

# Illustrative outdoor accumulated dose scenarios over 48 hours for a SIP event.
# Selected with reference to ICRP103 sheltering guidance; NOT action levels.
H_UTE_mSv = [5, 20, 50, 100]

# Scenario for Figur 2 og 3 (enkeltscenario)
H_UTE_ENKELT_mSv = 50

# ICRP103 historical sheltering band for averted dose over ~2 days.
# Used as a reference range for contextualisation only.
ICRP_BAND_LOW = 5.0
ICRP_BAND_HIGH = 50.0

# Building pairs: (stubbloft, rehabilitert, facade_label, file_stem)
BUILDING_PAIRS = [
    (
        "Murbygg | stubbloft bevart",
        "Murbygg | full utskifting + mineralull",
        "Murbygg",
        "murbygg",
    ),
    (
        "Trebygg | stubbloft bevart",
        "Trebygg | full utskifting + mineralull",
        "Trebygg",
        "trebygg",
    ),
]

# Colors
C_STUBBLOFT = "#2C6E49"
C_REHABILITERT = "#B54A1A"
C_ICRP_BAND = "#FAEFC2"
C_ICRP_EDGE = "#C4A92B"

H_UTE_COLORS = {
    5:   "#A8C4DC",
    20:  "#5A9BC4",
    50:  "#1F5F8A",
    100: "#0D2D45",
}
H_UTE_LINESTYLES = {
    5: ":",
    20: "--",
    50: "-",
    100: "-.",
}

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.20,
        "grid.linewidth": 0.6,
    }
)

# ── Data loading ──────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)

    # Print available buildings if expected variants are missing
    available = df["building"].unique()
    expected = {b for pair in BUILDING_PAIRS for b in pair[:2]}
    missing = expected - set(available)
    if missing:
        print("OBS: Følgende forventede varianter finnes ikke i resultatfilen:")
        for m in missing:
            print(f"  – {m}")
        print("Tilgjengelige varianter:")
        for b in available:
            print(f"  + {b}")

    df = df[df["scenario"] == SCENARIO].copy()
    df["label"] = pd.Categorical(df["label"], categories=FLOOR_ORDER, ordered=True)
    return df.sort_values(["building", "label"]).reset_index(drop=True)


def get_pf_series(df: pd.DataFrame, building_name: str) -> pd.Series:
    """Returns PF indexed by floor label for one building variant."""
    return (
        df[df["building"] == building_name]
        .set_index("label")["PF"]
        .reindex(FLOOR_ORDER)
    )


# ── Core calculation ──────────────────────────────────────────────────────────

def compute_doseavvik(
    df: pd.DataFrame,
    stubbloft_name: str,
    rehabilitert_name: str,
    facade_label: str,
) -> pd.DataFrame:
    """
    For each floor × H_ute combination, beregn:
      H_inne = H_ute / PF
      doseavvik = H_inne_rehabilitert - H_inne_stubbloft  (positivt der PF_s > PF_r)
      relativ_økning = H_inne_rehabilitert / H_inne_stubbloft
      avverget_dose = H_ute - H_inne
    """
    pf_s = get_pf_series(df, stubbloft_name)
    pf_r = get_pf_series(df, rehabilitert_name)

    rows = []
    for floor in FLOOR_ORDER:
        ps = pf_s.loc[floor]
        pr = pf_r.loc[floor]
        for h in H_UTE_mSv:
            h_in_s = h / ps
            h_in_r = h / pr
            rows.append(
                {
                    "scenario": SCENARIO,
                    "building_type": facade_label,
                    "floor": floor,
                    "H_ute_mSv": h,
                    "PF_stubbloft": round(ps, 4),
                    "PF_rehabilitert": round(pr, 4),
                    "H_inne_stubbloft_mSv": h_in_s,
                    "H_inne_rehabilitert_mSv": h_in_r,
                    "doseavvik_mSv": h_in_r - h_in_s,
                    "relativ_økning": h_in_r / h_in_s,
                    "avverget_dose_stubbloft_mSv": h - h_in_s,
                    "avverget_dose_rehabilitert_mSv": h - h_in_r,
                }
            )
    return pd.DataFrame(rows)


# ── Figur 1: doseavvik per etasje, alle H_ute-scenarioer ─────────────────────

def plot_doseavvik_leirfjerning(data: pd.DataFrame, facade_label: str, stem: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5))

    x = np.arange(len(FLOOR_ORDER))

    for h in H_UTE_mSv:
        sub = data[data["H_ute_mSv"] == h].set_index("floor").reindex(FLOOR_ORDER)
        ax.plot(
            x, sub["doseavvik_mSv"].values,
            color=H_UTE_COLORS[h],
            linestyle=H_UTE_LINESTYLES[h],
            marker="o", markersize=6, linewidth=1.8,
            label=f"{h} mSv utendørs / 48 t",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(FLOOR_ORDER, fontsize=10)
    ax.set_ylabel("Ekstra innendørs dose [mSv]", fontsize=10)
    ax.set_xlabel("Etasje", fontsize=10)
    ax.set_ylim(bottom=0)
    ax.set_title(
        f"Ekstra innendørs dose ved leirfjerning – {facade_label}",
        fontsize=11, fontweight="bold", pad=10,
    )
    ax.legend(title="Utendørs dose (SIP 48 t)", fontsize=9, title_fontsize=9)

    # Note that these are illustrative scenarios, not action levels
    ax.text(
        0.01, 0.98,
        "Illustrative dosescenarioer – ikke automatiske tiltaksgrenser (ref. ICRP103)",
        transform=ax.transAxes, fontsize=7.5, color="#555",
        va="top", ha="left", style="italic",
    )

    for ext in ("png", "pdf"):
        path = OUTPUT_FIGURES / f"doseavvik_leirfjerning_{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Lagret: {path.relative_to(ROOT)}")
    plt.close(fig)


# ── Figur 2: innendørs dose ved 50 mSv, begge varianter ─────────────────────

def plot_innendors_dose(data: pd.DataFrame, facade_label: str, stem: str) -> None:
    sub = data[data["H_ute_mSv"] == H_UTE_ENKELT_mSv].set_index("floor").reindex(FLOOR_ORDER)

    x = np.arange(len(FLOOR_ORDER))

    fig, ax = plt.subplots(figsize=(8.5, 5))

    ax.plot(
        x, sub["H_inne_stubbloft_mSv"].values,
        color=C_STUBBLOFT, marker="o", markersize=7, linewidth=2.0,
        label="Stubbloft bevart",
    )
    ax.plot(
        x, sub["H_inne_rehabilitert_mSv"].values,
        color=C_REHABILITERT, marker="s", markersize=7, linewidth=2.0,
        label="Full utskifting + mineralull",
    )

    # Annotate doseavvik at each floor
    for i, floor in enumerate(FLOOR_ORDER):
        s_val = sub.loc[floor, "H_inne_stubbloft_mSv"]
        r_val = sub.loc[floor, "H_inne_rehabilitert_mSv"]
        avvik = r_val - s_val
        if avvik > 0.3:
            ax.annotate(
                f"+{avvik:.1f}",
                xy=(i, r_val),
                xytext=(i + 0.12, r_val + 0.25),
                fontsize=8, color=C_REHABILITERT,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(FLOOR_ORDER, fontsize=10)
    ax.set_ylabel(f"Innendørs akkumulert dose [mSv]", fontsize=10)
    ax.set_xlabel("Etasje", fontsize=10)
    ax.set_ylim(bottom=0)
    ax.set_title(
        f"Innendørs dose ved {H_UTE_ENKELT_mSv} mSv utendørs dose over 48 timer",
        fontsize=11, fontweight="bold", pad=10,
    )
    ax.legend(fontsize=9)
    ax.text(
        0.01, 0.98,
        f"Utendørs dose {H_UTE_ENKELT_mSv} mSv er illustrativt – ikke en spesifikk ulykkesmodell (ref. ICRP103)",
        transform=ax.transAxes, fontsize=7.5, color="#555",
        va="top", ha="left", style="italic",
    )

    for ext in ("png", "pdf"):
        path = OUTPUT_FIGURES / f"innendors_dose_{H_UTE_ENKELT_mSv}mSv_48h_{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Lagret: {path.relative_to(ROOT)}")
    plt.close(fig)


# ── Figur 3: avverget dose med ICRP103-band ───────────────────────────────────

def plot_avverget_dose_icrp(data: pd.DataFrame, facade_label: str, stem: str) -> None:
    sub = data[data["H_ute_mSv"] == H_UTE_ENKELT_mSv].set_index("floor").reindex(FLOOR_ORDER)

    x = np.arange(len(FLOOR_ORDER))

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    # ICRP103 historical sheltering band (5–50 mSv averted dose over ~2 days).
    # This is a HISTORICAL REFERENCE RANGE from ICRP103, not an automatic action level.
    # It is shown to contextualise the calculated averted doses.
    ax.axhspan(
        ICRP_BAND_LOW, ICRP_BAND_HIGH,
        color=C_ICRP_BAND, alpha=0.55, zorder=0,
        label=f"ICRP103: historisk sheltering-intervall\nfor avverget dose over 2 døgn\n({ICRP_BAND_LOW:.0f}–{ICRP_BAND_HIGH:.0f} mSv, ikke en tiltaksgrense)",
    )
    ax.axhline(ICRP_BAND_LOW,  color=C_ICRP_EDGE, linewidth=0.8, linestyle="--", alpha=0.7)
    ax.axhline(ICRP_BAND_HIGH, color=C_ICRP_EDGE, linewidth=0.8, linestyle="--", alpha=0.7)

    ax.plot(
        x, sub["avverget_dose_stubbloft_mSv"].values,
        color=C_STUBBLOFT, marker="o", markersize=7, linewidth=2.0,
        label="Stubbloft bevart", zorder=3,
    )
    ax.plot(
        x, sub["avverget_dose_rehabilitert_mSv"].values,
        color=C_REHABILITERT, marker="s", markersize=7, linewidth=2.0,
        label="Full utskifting + mineralull", zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(FLOOR_ORDER, fontsize=10)
    ax.set_ylabel("Avverget dose [mSv]", fontsize=10)
    ax.set_xlabel("Etasje", fontsize=10)
    ax.set_ylim(0, H_UTE_ENKELT_mSv * 1.08)
    ax.set_title(
        f"Avverget dose ved {H_UTE_ENKELT_mSv} mSv utendørs dose over 48 timer – {facade_label}",
        fontsize=11, fontweight="bold", pad=10,
    )
    ax.legend(fontsize=9, loc="lower right")

    for ext in ("png", "pdf"):
        path = OUTPUT_FIGURES / f"avverget_dose_icrp_band_{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Lagret: {path.relative_to(ROOT)}")
    plt.close(fig)


# ── Terminal quality-control table ────────────────────────────────────────────

def print_qc_table(data_murbygg: pd.DataFrame) -> None:
    print("\n" + "=" * 105)
    print("KVALITETSKONTROLL – Murbygg (etasjemidlede PF fra modellen)")
    print("=" * 105)
    hdr = (
        f"{'ETASJE':<12} {'PF_s':>6} {'PF_r':>6} {'H_ute':>6} "
        f"{'H_in_s':>9} {'H_in_r':>9} {'avvik':>8} {'lin.sjekk':>10} {'OK':>4}"
    )
    print(hdr)
    print("-" * 105)

    # Linearity check: doseavvik / H_ute should be constant per floor
    prev_ratio: dict[str, float] = {}
    for _, row in data_murbygg.sort_values(["floor", "H_ute_mSv"]).iterrows():
        fl = row["floor"]
        h = row["H_ute_mSv"]
        avvik = row["doseavvik_mSv"]
        ratio = avvik / h  # should equal 1/PF_r - 1/PF_s

        if fl not in prev_ratio:
            prev_ratio[fl] = ratio
            lin_ok = "ref"
        else:
            # ratio should be constant across H_ute for same floor
            lin_ok = f"{ratio:.6f}"

        sign_ok = "OK" if avvik >= 0 else "FEIL"
        print(
            f"{fl:<12} {row['PF_stubbloft']:>6.2f} {row['PF_rehabilitert']:>6.2f} "
            f"{h:>6.0f} {row['H_inne_stubbloft_mSv']:>9.4f} "
            f"{row['H_inne_rehabilitert_mSv']:>9.4f} {avvik:>8.4f} "
            f"{ratio:>10.6f} {sign_ok:>4}"
        )

    print("-" * 105)
    print(
        "lin.sjekk = avvik / H_ute: konstant per etasje bekrefter lineær skalering.\n"
        "Positivt avvik = rehabilitert konstruksjon gir høyere innendørs dose enn stubbloft."
    )


# ── Save CSV ──────────────────────────────────────────────────────────────────

def save_csv(all_data: pd.DataFrame) -> None:
    cols = [
        "scenario", "building_type", "floor", "H_ute_mSv",
        "PF_stubbloft", "PF_rehabilitert",
        "H_inne_stubbloft_mSv", "H_inne_rehabilitert_mSv",
        "doseavvik_mSv", "relativ_økning",
        "avverget_dose_stubbloft_mSv", "avverget_dose_rehabilitert_mSv",
    ]
    path = OUTPUT_TABLES / "doseavvik_sip_48h.csv"
    all_data[cols].to_csv(path, index=False, float_format="%.6g")
    print(f"\n  CSV lagret: {path.relative_to(ROOT)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Leser data:  {DATA_FILE.relative_to(ROOT)}")
    print(f"Scenario:    {SCENARIO}")
    print(f"H_ute [mSv]: {H_UTE_mSv}  (illustrativt, ref. ICRP103 – ikke automatiske tiltaksgrenser)\n")

    df = load_data()

    all_frames = []

    for stubbloft_name, rehabilitert_name, facade_label, stem in BUILDING_PAIRS:
        print(f"[{facade_label}]  {stubbloft_name}")
        print(f"         vs.  {rehabilitert_name}")

        data = compute_doseavvik(df, stubbloft_name, rehabilitert_name, facade_label)
        all_frames.append(data)

        print("  Figur 1 – doseavvik leirfjerning:")
        plot_doseavvik_leirfjerning(data, facade_label, stem)

        print("  Figur 2 – innendørs dose 50 mSv:")
        plot_innendors_dose(data, facade_label, stem)

        print("  Figur 3 – avverget dose med ICRP-band:")
        plot_avverget_dose_icrp(data, facade_label, stem)
        print()

    all_data = pd.concat(all_frames, ignore_index=True)
    save_csv(all_data)

    murbygg_data = all_data[all_data["building_type"] == "Murbygg"]
    print_qc_table(murbygg_data)

    print("\nFerdig.")


if __name__ == "__main__":
    main()
