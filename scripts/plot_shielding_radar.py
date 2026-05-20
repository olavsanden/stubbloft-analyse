"""
Shielding fingerprint – radardiagram/polarplot per etasje.

For to bygningsvarianter plottes tre kurver på et radardiagram:
  - PF normalisert = PF / maks(PF) innenfor varianten
  - Ground shine-andel = Dg_in / (Dg_in + Dr_in)
  - Roof shine-andel   = Dr_in / (Dg_in + Dr_in)

Etasjene er arrangert som akser i polarplottet (Kjeller øverst, med klokken).
Polygonene lukkes ved å gjenta første verdi til slutt.

Ingen gamle tall benyttes. Ingen PF-modellparametre endres.
Data hentes utelukkende fra output/pf_master_results_refaktorert.csv.
"""

import shutil
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "output" / "pf_master_results_refaktorert.csv"
OUTPUT_FIGURES = ROOT / "output" / "figures"
OUTPUT_TABLES = ROOT / "output" / "tables"
OLD_HORIZONTAL_DIR = ROOT / "figures" / "shielding_profiles"
ARCHIVE_DIR = ROOT / "figures" / "archive" / "shielding_profiles_horizontal"
OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
SCENARIO = "Luftespalte 15x15 cm"

FLOOR_ORDER = [
    "Kjeller",
    "1. etasje",
    "2. etasje",
    "3. etasje",
    "4. etasje",
    "5. etasje",
]
N = len(FLOOR_ORDER)

MURBYGG_NAME = "Murbygg | stubbloft bevart"
TREBYGG_PREFERRED = "Trebygg | referanse lett"
TREBYGG_FALLBACK = "Trebygg | full utskifting + mineralull"

# Metric definitions: (column, legend label, color, linestyle, marker, fill_alpha)
METRICS = [
    ("PF_norm",      "PF normalisert",     "#2166AC", "-",  "o", 0.11),
    ("ground_andel", "Ground shine-andel", "#D95F02", "--", "s", 0.09),
    ("roof_andel",   "Roof shine-andel",   "#1B9E77", "-.", "^", 0.09),
]

# Radar angles: Kjeller at top, clockwise
ANGLES = [n / float(N) * 2 * np.pi for n in range(N)]
ANGLES_CLOSED = ANGLES + [ANGLES[0]]

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})

# ── Data ──────────────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(DATA_FILE)
    available = set(df["building"].unique())

    if TREBYGG_PREFERRED in available:
        trebygg_name = TREBYGG_PREFERRED
    else:
        trebygg_name = TREBYGG_FALLBACK
        print(f"OBS: '{TREBYGG_PREFERRED}' ikke funnet, bruker '{TREBYGG_FALLBACK}'")

    print(f"Trebygg-variant brukt: {trebygg_name}")

    df = df[df["scenario"] == SCENARIO].copy()
    df["label"] = pd.Categorical(df["label"], categories=FLOOR_ORDER, ordered=True)
    return df.sort_values(["building", "label"]).reset_index(drop=True), trebygg_name


def compute_profile(df: pd.DataFrame, building_name: str) -> pd.DataFrame:
    sub = (
        df[df["building"] == building_name]
        .set_index("label")
        .reindex(FLOOR_ORDER)
        .copy()
    )
    # Beregn fra råkomponenter; ekvivalent med ground_frac/roof_frac i resultfilen
    din = sub["Dg_in"] + sub["Dr_in"]
    sub["ground_andel"] = sub["Dg_in"] / din
    sub["roof_andel"] = sub["Dr_in"] / din
    sub["PF_norm"] = sub["PF"] / sub["PF"].max()
    return sub


# ── Radar axis setup ──────────────────────────────────────────────────────────

def configure_radar_ax(ax: plt.Axes) -> None:
    ax.set_theta_offset(np.pi / 2)   # Kjeller øverst
    ax.set_theta_direction(-1)        # Med klokken

    ax.set_xticks(ANGLES)
    ax.set_xticklabels(FLOOR_ORDER, fontsize=9.5, color="#222")

    ax.set_rlim(0, 1.0)
    ax.set_rticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(
        ["0.25", "0.50", "0.75", "1.00"],
        fontsize=7.5, color="#777",
    )
    ax.set_rlabel_position(20)  # plasser radiale etiketter ved 20° fra Kjeller-aksen
    ax.grid(color="#bbb", linestyle="--", linewidth=0.55, alpha=0.55)
    ax.spines["polar"].set_color("#ccc")
    ax.spines["polar"].set_linewidth(0.8)
    ax.set_facecolor("#FAFAFA")


def draw_radar_curves(ax: plt.Axes, sub: pd.DataFrame) -> list:
    """Plotter kurvene på en allerede konfigurert polarakse."""
    handles = []
    for col, label, color, ls, marker, alpha in METRICS:
        vals = sub[col].values.tolist()
        vals_closed = vals + [vals[0]]

        (line,) = ax.plot(
            ANGLES_CLOSED, vals_closed,
            color=color, linestyle=ls, linewidth=2.0,
            marker=marker, markersize=6.5,
            label=label, zorder=3,
        )
        ax.fill(
            ANGLES_CLOSED, vals_closed,
            color=color, alpha=alpha, zorder=2,
        )
        handles.append(line)
    return handles


# ── Figur 1 og 2: individuelle radarplott ─────────────────────────────────────

def plot_single(
    df: pd.DataFrame,
    building_name: str,
    panel_title: str,
    file_stem: str,
) -> pd.DataFrame:
    sub = compute_profile(df, building_name)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="polar")
    configure_radar_ax(ax)
    handles = draw_radar_curves(ax, sub)
    ax.set_title(panel_title, fontsize=11.5, fontweight="bold", pad=20)

    fig.legend(
        handles=handles,
        labels=[m[1] for m in METRICS],
        loc="lower center",
        ncol=3,
        fontsize=9,
        framealpha=0.93,
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.suptitle(
        "Skjermingsfingeravtrykk ved hovedscenario",
        fontsize=12, fontweight="bold", y=1.03,
    )
    fig.subplots_adjust(bottom=0.12)

    for ext in ("png", "pdf"):
        path = OUTPUT_FIGURES / f"shielding_radar_{file_stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Lagret: {path.relative_to(ROOT)}")
    plt.close(fig)
    return sub


# ── Figur 3: sammenlignende radardiagram ──────────────────────────────────────

def plot_comparison(
    df: pd.DataFrame,
    murbygg_name: str,
    trebygg_name: str,
    trebygg_panel_title: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub_m = compute_profile(df, murbygg_name)
    sub_t = compute_profile(df, trebygg_name)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(13.5, 7),
        subplot_kw={"projection": "polar"},
    )
    fig.subplots_adjust(wspace=0.55, bottom=0.12, top=0.90)

    configure_radar_ax(axes[0])
    handles = draw_radar_curves(axes[0], sub_m)
    axes[0].set_title("Murbygg med stubbloft", fontsize=11, fontweight="bold", pad=20)

    configure_radar_ax(axes[1])
    draw_radar_curves(axes[1], sub_t)
    axes[1].set_title(trebygg_panel_title, fontsize=11, fontweight="bold", pad=20)

    fig.legend(
        handles=handles,
        labels=[m[1] for m in METRICS],
        loc="lower center",
        ncol=3,
        fontsize=10,
        framealpha=0.93,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Skjermingsfingeravtrykk ved hovedscenario",
        fontsize=13, fontweight="bold",
    )

    for ext in ("png", "pdf"):
        path = OUTPUT_FIGURES / f"shielding_radar_comparison.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Lagret: {path.relative_to(ROOT)}")
    plt.close(fig)
    return sub_m, sub_t


# ── Terminal quality-control table ────────────────────────────────────────────

def print_table(profiles: dict[str, tuple[str, pd.DataFrame]]) -> None:
    w = 90
    print("\n" + "=" * w)
    print("KVALITETSKONTROLL – verdier brukt i radardiagrammene")
    print("=" * w)
    hdr = (
        f"{'VARIANT':<35} {'ETASJE':<12} {'PF':>7} "
        f"{'PF_norm':>8} {'ground':>8} {'roof':>8} {'sum':>6} {'OK':>4}"
    )
    print(hdr)
    print("-" * w)
    for stem, (bname, sub) in profiles.items():
        short = bname.split(" | ")[1] if " | " in bname else bname[:30]
        building_label = bname.split(" | ")[0]
        label = f"{building_label} | {short}"[:35]
        for floor in FLOOR_ORDER:
            row = sub.loc[floor]
            sumf = row["ground_andel"] + row["roof_andel"]
            in_range = (
                0 <= row["PF_norm"] <= 1
                and 0 <= row["ground_andel"] <= 1
                and 0 <= row["roof_andel"] <= 1
            )
            ok = "OK" if abs(sumf - 1.0) < 1e-9 and in_range else "FEIL"
            print(
                f"{label:<35} {floor:<12} {row['PF']:>7.2f} "
                f"{row['PF_norm']:>8.4f} {row['ground_andel']:>8.4f} "
                f"{row['roof_andel']:>8.4f} {sumf:>6.4f} {ok:>4}"
            )
        print("-" * w)
    print("PF_norm = PF / maks(PF) per variant. ground + roof = 1.0 er verifisert.")


# ── Save CSV ──────────────────────────────────────────────────────────────────

def save_csv(profiles: dict[str, tuple[str, pd.DataFrame]]) -> None:
    rows = []
    for stem, (bname, sub) in profiles.items():
        for floor in FLOOR_ORDER:
            row = sub.loc[floor]
            rows.append(
                {
                    "variant_stem": stem,
                    "building": bname,
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
    path = OUTPUT_TABLES / "shielding_radar_data.csv"
    out.to_csv(path, index=False, float_format="%.8g")
    print(f"\n  CSV lagret: {path.relative_to(ROOT)}")


# ── Archive old horizontal figures ────────────────────────────────────────────

def archive_old_horizontal_figures() -> None:
    """
    Flytter PNG/PDF fra figures/shielding_profiles/ til
    figures/archive/shielding_profiles_horizontal/.
    CSV-filer bevares på plass (rådata).
    """
    if not OLD_HORIZONTAL_DIR.exists():
        print("\n  (Ingen gamle horisontale figurer funnet – ingenting å flytte.)")
        return

    to_move = [
        p for p in OLD_HORIZONTAL_DIR.iterdir()
        if p.suffix.lower() in (".png", ".pdf")
    ]
    if not to_move:
        print("\n  (Ingen PNG/PDF i figures/shielding_profiles/ – ingenting å flytte.)")
        return

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n  Arkiverer {len(to_move)} gamle horisontale figurer → {ARCHIVE_DIR.relative_to(ROOT)}/")
    for f in to_move:
        dest = ARCHIVE_DIR / f.name
        shutil.move(str(f), str(dest))
        print(f"    Flyttet: {f.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Leser data:  {DATA_FILE.relative_to(ROOT)}")
    print(f"Scenario:    {SCENARIO}\n")

    df, trebygg_name = load_data()

    # Panel title for trebygg (kortere versjon for figur)
    trebygg_panel = "Trebygg med lett etasjeskiller"

    print("\nIndividuelle radarfigurer:")
    sub_m = plot_single(df, MURBYGG_NAME, "Murbygg med stubbloft", "murbygg_stubbloft")
    sub_t = plot_single(df, trebygg_name, trebygg_panel, "trebygg_lett")

    print("\nSammenligningsfigur:")
    plot_comparison(df, MURBYGG_NAME, trebygg_name, trebygg_panel)

    profiles = {
        "murbygg_stubbloft": (MURBYGG_NAME, sub_m),
        "trebygg_lett": (trebygg_name, sub_t),
    }

    print_table(profiles)
    save_csv(profiles)

    archive_old_horizontal_figures()

    print(f"\nFerdig. Figurer i: {OUTPUT_FIGURES.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
