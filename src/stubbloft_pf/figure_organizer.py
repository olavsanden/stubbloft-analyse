# -*- coding: utf-8 -*-
"""
figure_organizer.py
-------------------
Kopierer genererte figurer fra figures/ til en organisert undermappestruktur.
Originale filer i figures/ beholdes uendret.
"""
from __future__ import annotations

import shutil
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, List, Tuple

# ── Hovedfigurer (kopi til main/ med nytt løpenummer-navn) ────────────────
_MAIN_FIGURES: List[Tuple[str, str, str]] = [
    (
        "pf_band_murbygg_kjellervindu_40x60_cm.png",
        "01_pf_band_murbygg.png",
        "PF-bånd med sensitivitetsbånd — Murbygg | Kjellervindu 40×60 cm",
    ),
    (
        "pf_ratio_radius_murbygg.png",
        "02_pf_ratio_radius_murbygg.png",
        "PF-ratio stubb/utsk per etasje, ulike ground radius — Murbygg",
    ),
    (
        "ground_pct_radius_murbygg.png",
        "03_ground_pct_radius_murbygg.png",
        "Ground%-andel per etasje ved ulike ground radius — Murbygg",
    ),
    (
        "pf_deck_multifocus_shapley_murbygg.png",
        "04_deck_multifocus_shapley_murbygg.png",
        "Shapley-verdier per dekke for tre målsett — Murbygg",
    ),
    (
        "pf_deck_multifocus_best_by_n_murbygg.png",
        "05_deck_multifocus_best_by_n_murbygg.png",
        "Beste kombinasjon per N stubbloft-dekker (Målsett A vs B) — Murbygg",
    ),
    (
        "pf_kjeller_sensitivity_murbygg.png",
        "06_kjeller_sensitivity_murbygg.png",
        "Kjeller-PF per åpningsscenario — Murbygg",
    ),
]

# ── Kategoriseringsregler (first-match-wins, mer spesifikke regler først) ─
# Hvert element: (glob-mønster, undermappenavn)
_RULES: List[Tuple[str, str]] = [
    # Uten kjeller — før pf_band_*
    ("pf_band_nkjeller_*.png",         "appendix/no_basement"),
    # PF-båndfigurer per scenario
    ("pf_band_*.png",                  "sensitivity/pf_bands"),
    # Radius-sensitivitet — radius-varianter før generelle pf_ratio_*
    ("pf_ratio_radius_*.png",          "sensitivity/radius"),
    ("ground_pct_radius_*.png",        "sensitivity/radius"),
    # Dekke-kombinasjonsanalyse — spesifikke mønstre før generelle pf_deck_*
    ("pf_deck_best_by_n_*.png",        "sensitivity/deck_combinations"),
    ("pf_deck_multifocus_*.png",       "sensitivity/deck_combinations"),
    ("pf_deck_combo_*.png",            "sensitivity/deck_combinations"),
    ("pf_deck_pair_*.png",             "sensitivity/deck_combinations"),
    # Enkeltdekke-prioriteringsanalyse
    ("pf_deck_heatmap_*.png",          "sensitivity/deck_importance"),
    ("pf_deck_lines_*.png",            "sensitivity/deck_importance"),
    ("pf_deck_shapley_*.png",          "sensitivity/deck_importance"),
    ("pf_basement_ceiling_*.png",      "sensitivity/deck_importance"),
    # Takmodell-sammenligning
    ("pf_roof_comparison_*.png",       "sensitivity/roof_model"),
    # Kjelleråpnings-sensitivitet
    ("pf_kjeller_sensitivity_*.png",   "sensitivity/kjeller_openings"),
    # Ground/Roof-andeler
    ("ground_roof_main_*.png",         "sensitivity/ground_roof"),
    # Vedlegg: stolpediagram, reduksjon, diagnostikk
    ("pf_bar_*.png",                   "appendix/bar_charts"),
    ("pf_reduction_*.png",             "appendix/reduction"),
    ("pf_ratio_*.png",                 "appendix/bar_charts"),   # per-scenario ratio (ikke radius)
    ("diag_*.png",                     "appendix/diagnostics"),
    # Arkiv: opprinnelige profilfigurer og andre eldre arbeidsfigurer
    ("pf_profile_section_*.png",       "archive/legacy_flat_files"),
    ("pf_profile_*.png",               "archive/legacy_flat_files"),
    ("pf_heatmap_*.png",               "archive/legacy_flat_files"),
    ("location_factor_*.png",          "archive/legacy_flat_files"),
    ("dose_components_*.png",          "archive/legacy_flat_files"),
    ("floor_crossings_*.png",          "archive/legacy_flat_files"),
    ("ground_roof_share_*.png",        "archive/legacy_flat_files"),
]

# Alle filer med "trebygg" i navnet kopieres i tillegg hit
_TREBYGG_DIR = "appendix/trebygg"


def organize_figures(figures_dir: Path) -> Tuple[int, int]:
    """
    Kopier figurer til organisert undermappestruktur under figures_dir.
    Originale filer i figures_dir/ forblir uendret.

    Returnerer (n_kopiert_unike_filer, n_manglende_hovdfigurer).
    """
    if not figures_dir.exists():
        print(f"  [ADVARSEL] figures_dir finnes ikke: {figures_dir}")
        return 0, 0

    # ── Opprett alle undermapper ──────────────────────────────────────────
    needed_dirs = (
        ["main"]
        + list(dict.fromkeys(r[1] for r in _RULES))   # unik rekkefølge bevart
        + [_TREBYGG_DIR, "archive/legacy_flat_files", "archive/original_top_level"]
    )
    for d in needed_dirs:
        (figures_dir / d).mkdir(parents=True, exist_ok=True)

    n_copied   = 0
    n_missing  = 0

    # ── 1. Kopier og omdøp til main/ ─────────────────────────────────────
    manifest_main: List[Tuple[str, str, bool]] = []
    for src_name, dst_name, desc in _MAIN_FIGURES:
        src = figures_dir / src_name
        dst = figures_dir / "main" / dst_name
        if src.exists():
            shutil.copy2(src, dst)
            n_copied += 1
            manifest_main.append((dst_name, desc, True))
        else:
            print(f"  [ADVARSEL] Manglende forventet figur: {src_name}")
            n_missing += 1
            manifest_main.append((dst_name, desc, False))

    # ── 2. Kategoriser og kopier alle PNG-er (first-match-wins) ──────────
    all_pngs = sorted(figures_dir.glob("*.png"))
    categorized: Dict[str, List[str]] = {}
    uncategorized: List[str] = []

    for png in all_pngs:
        name = png.name
        matched = False
        for pattern, subdir in _RULES:
            if fnmatch(name, pattern):
                dst = figures_dir / subdir / name
                shutil.copy2(png, dst)
                n_copied += 1
                categorized.setdefault(subdir, []).append(name)
                matched = True
                break
        if not matched:
            uncategorized.append(name)

    # ── 3. Ekstra kopi av alle trebygg-filer til appendix/trebygg/ ───────
    for png in all_pngs:
        if "trebygg" in png.name.lower():
            dst = figures_dir / _TREBYGG_DIR / png.name
            shutil.copy2(png, dst)
            # Telles ikke i n_copied (tilleggskopi, ikke primærkategorisering)

    # ── 4. Arkiver gjenværende toppnivå-PNG-er ───────────────────────────
    archive_top = figures_dir / "archive" / "original_top_level"
    top_pngs = sorted(figures_dir.glob("*.png"))   # re-les: bare filer direkte i figures/
    for png in top_pngs:
        shutil.move(str(png), archive_top / png.name)
    print(f"  Toppnivåfigurer arkivert: {len(top_pngs)} filer")

    # ── 5. Skriv manifestfil ──────────────────────────────────────────────
    _write_manifest(figures_dir, manifest_main, categorized, uncategorized, n_copied, n_missing)

    return n_copied, n_missing


def _write_manifest(
    figures_dir: Path,
    manifest_main: List[Tuple[str, str, bool]],
    categorized: Dict[str, List[str]],
    uncategorized: List[str],
    n_copied: int,
    n_missing: int,
) -> None:
    lines: List[str] = [
        "# Figur-manifest",
        f"Generert: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Kopiert: {n_copied}  ·  Manglende forventede: {n_missing}",
        "",
    ]

    # Hovedfigurer
    lines += ["---", "", "## Hovedfigurer — figures/main/", ""]
    for dst_name, desc, found in manifest_main:
        status = "✓" if found else "✗ MANGLER"
        lines.append(f"- `{dst_name}`  {status}")
        lines.append(f"  _{desc}_")
    lines.append("")

    # Sensitivitetsfigurer
    lines += ["---", "", "## Sensitivitetsfigurer — figures/sensitivity/", ""]
    for subdir in sorted(k for k in categorized if k.startswith("sensitivity/")):
        cat = subdir.split("/", 1)[-1]
        lines.append(f"### {cat}/")
        for f in sorted(categorized[subdir]):
            lines.append(f"- `{f}`")
        lines.append("")

    # Vedleggsfigurer
    lines += ["---", "", "## Vedleggsfigurer — figures/appendix/", ""]
    for subdir in sorted(k for k in categorized if k.startswith("appendix/")):
        cat = subdir.split("/", 1)[-1]
        lines.append(f"### {cat}/")
        for f in sorted(categorized[subdir]):
            lines.append(f"- `{f}`")
        lines.append("")

    # Arkiv
    arch_key = "archive/legacy_flat_files"
    if categorized.get(arch_key):
        lines += ["---", "", "## Arkiv — figures/archive/legacy_flat_files/", ""]
        for f in sorted(categorized[arch_key]):
            lines.append(f"- `{f}`")
        lines.append("")

    # Ukategoriserte
    if uncategorized:
        lines += ["---", "", "## Ukategoriserte filer (beholdt i figures/)", ""]
        for f in sorted(uncategorized):
            lines.append(f"- `{f}`")
        lines.append("")

    (figures_dir / "FIGURE_MANIFEST.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
