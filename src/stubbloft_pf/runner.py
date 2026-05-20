# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import List, Sequence

from .calculations import calculate_rows_for_points, rows_to_dataframe
from .figure_organizer import organize_figures
from .config import Building, ResultRow
from .geometry import basement_opening_fraction, default_five_points_for_building
from .physics import effective_light_joist_floor_mass_kg_m2
from .plots import (
    plot_basement_opening_sensitivity,
    plot_floor_crossings,
    plot_ground_roof_main_buildings,
    plot_ground_roof_share_by_floor,
    plot_indoor_dose_components,
    plot_location_factor,
    plot_pf_band_per_scenario,
    plot_pf_bars,
    plot_pf_heatmap,
    plot_pf_profile_with_building_section,
    plot_pf_profiles,
    plot_pf_ratio,
    plot_pf_reduction_pct,
    plot_pf_sensitivity_band_by_facade,
)
from .physics import (
    stubbloft_floor_mass_from_clay_kg_m2,
    STUBB_CLAY_THICKNESS_LOW_M,
    STUBB_CLAY_THICKNESS_HIGH_M,
)
from .scenarios import (
    M_FULL_UTSKIFTING,
    M_STUBB_BEVART,
    calculate_plot_data,
    make_buildings,
    make_scenarios,
)
from .sensitivity import (
    plot_basement_ceiling_sensitivity,
    plot_best_combo_by_n,
    plot_deck_combo_heatmap,
    plot_deck_importance_heatmap,
    plot_deck_importance_lines,
    plot_deck_pair_heatmap,
    plot_deck_pair_heatmap_setA,
    plot_deck_shapley,
    plot_multifocus_best_by_n,
    plot_multifocus_shapley,
    plot_pf_by_radius,
    plot_roof_model_comparison,
    run_basement_ceiling_sensitivity,
    run_clay_thickness_sensitivity,
    run_deck_combination_analysis,
    run_floor_deck_importance,
    run_floor_mass_sensitivity,
    run_ground_radius_sensitivity,
    run_roof_model_comparison,
)


def print_result_table(rows: Sequence[ResultRow], building_name: str) -> None:
    print("\n" + building_name)
    print("-" * 90)
    print(
        f"{'Etasje':10s} "
        f"{'PF':>8s} "
        f"{'PF_raw':>8s} "
        f"{'LF':>8s} "
        f"{'Ground%':>9s} "
        f"{'Roof%':>8s}"
    )

    for r in rows:
        print(
            f"{r.label:10s} "
            f"{r.PF:8.2f} "
            f"{r.PF_raw:8.2f} "
            f"{r.LF:8.3f} "
            f"{100 * r.ground_frac:8.1f}% "
            f"{100 * r.roof_frac:7.1f}%"
        )


def _scenario_slug(name: str) -> str:
    # Gjør om scenario-navn til filnavn-vennlig streng, f.eks. "Kjellervindu 40×60 cm" → "kjellervindu_40x60_cm"
    slug = name.lower()
    slug = slug.replace("×", "x").replace("*", "x")
    slug = "".join(c if c.isalnum() else "_" for c in slug)
    slug = "_".join(p for p in slug.split("_") if p)
    return slug


def main() -> None:
    # ------------------------------------------------------------
    # Kjøringsvalg
    # ------------------------------------------------------------
    SHOW_FIGURES = False
    SAVE_FIGURES = True
    FIGURE_DIR = Path("figures")

    RUN_FIGURES = True
    RUN_PROFILE_WITH_BUILDING_SECTION = True

    # Hold disse av mens du tester. Skru på når du vil lage flere figurer.
    RUN_EXTRA_FIGURES = False

    # Sensitiviteter gir egne CSV-filer i output/.
    RUN_RADIUS_SENSITIVITY = False
    RUN_CLAY_SENSITIVITY   = False
    # Takmodellsammenligning: flat tak, saltak 35°/45°/50°.
    # Kjøres ikke som standard siden beregningene er tidkrevende.
    # Aktiver ved å sette True, eller kjør run_roof_model_comparison() direkte.
    RUN_ROOF_COMPARISON    = False

    # Bør aktiveres når kjellerresultater diskuteres i rapporten.
    RUN_BASEMENT_CEILING_SENSITIVITY = False

    # Etasjeskillermasse-spenn: stubbloft 174–230 kg/m² (6–10 cm leire) og utskiftet 49–76 kg/m².
    # Bør aktiveres når 2.–4. etasje-resultater presenteres.
    RUN_FLOOR_MASS_SENSITIVITY = False

    # Per-dekke prioriteringsanalyse: hvilket enkeltdekke bidrar mest til PF?
    # Genererer heatmap + linjefigurer i figures/ og CSV i output/.
    RUN_DECK_IMPORTANCE = True

    # Alle 2^5 = 32 kombinasjoner. Tyngre analyse (~2 min extra).
    # Genererer heatmap, beste-per-N, par-heatmap og Shapley-figurer.
    RUN_DECK_COMBINATION = True

    # PRINT_FULL_TABLES = True → printer detaljerte PF/LF-tabeller per bygg og etasje
    #                            (opprinnelig debugg-utdata).
    # PRINT_FULL_TABLES = False → kompakt kjørerapport (standard).
    PRINT_FULL_TABLES = False

    import warnings as _warnings
    _warnings.filterwarnings("ignore", message=".*tight_layout.*")

    def _status(label: str) -> None:
        # Kompakt statuslinje for sensitiviteter.
        print(f"  {label:<48} ✓")

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    if SAVE_FIGURES:
        FIGURE_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------------------
    # Modelloppsett
    # ------------------------------------------------------------
    # Takmodell: historisk tegningsgrunnlag viser bratt saltak.
    # 45° brukes som representativ hovedverdi.
    # Flat tak beholdes som referanse i run_roof_model_comparison().
    base = Building(
        basement_depth_m=2.2,
        basement_exposed_wall_height_m=0.60,
        ground_radius_m=60.0,
        use_buildup=True,
        buildup_alpha=0.10,
        use_height_dependent_windows=True,
        roof_model="sloped",
        roof_pitch_deg=45.0,
    )

    scenarios = make_scenarios(base)
    points = default_five_points_for_building(base)
    point_name = "5 målepunkter i rommet"

    all_rows: List[ResultRow] = []

    from .physics import light_reference_floor_mass_kg_m2
    _sep = "-" * 56
    print(f"\nSTUBBLOFT PF-MODELL")
    print(_sep)
    print(f"  Takmodell:        {base.roof_model}, {base.roof_pitch_deg:.0f}°")
    print(f"  Ground radius:    {base.ground_radius_m:.0f} m")
    print(f"  Etasjeskillere:   stubbloft {M_STUBB_BEVART:.0f} kg/m²  ·  utskifting {M_FULL_UTSKIFTING:.0f} kg/m²")
    print(f"  Scenarioer:       {len(scenarios)}")
    print(f"  Målepunkter:      {point_name}")
    if PRINT_FULL_TABLES:
        print(f"\n  [PRINT_FULL_TABLES=True: detaljerte tabeller aktivert]")
    print()

    # ------------------------------------------------------------
    # Hovedberegning
    # ------------------------------------------------------------
    # Sensitivitetsspenn for stubbloft: 6–10 cm leire → 174–230 kg/m²
    M_STUBB_LOW  = stubbloft_floor_mass_from_clay_kg_m2(STUBB_CLAY_THICKNESS_LOW_M)   # 174 kg/m²
    M_STUBB_HIGH = stubbloft_floor_mass_from_clay_kg_m2(STUBB_CLAY_THICKNESS_HIGH_M)  # 230 kg/m²
    M_UTSK_LOW,   M_UTSK_HIGH  = 49.0,  76.0

    _facade_configs = [
        ("Murbygg", 450.0, 650.0, "murbygg"),
        ("Trebygg", 80.0,  350.0, "trebygg"),
    ]

    all_scenario_data: dict = {}

    _key_data: dict = {}   # scenario_name → data, for nøkkelfunn-utskrift

    for scenario in scenarios:
        scenario_name = scenario.name
        buildings = make_buildings(scenario)

        if PRINT_FULL_TABLES:
            print("\n" + "=" * 90)
            print(f"SCENARIO: {scenario_name}")
            print("=" * 90)
            print(f"Åpningsandel kjeller: {100 * basement_opening_fraction(scenario):.2f}%")
            print(f"Analysepunkt: {point_name}")
        else:
            print(f"  Beregner: {scenario_name} ...", end=" ", flush=True)

        data = calculate_plot_data(scenario_name, point_name, points, buildings)
        all_scenario_data[scenario_name] = data
        _key_data[scenario_name] = data

        for b in buildings:
            rows = data[b.name]
            all_rows.extend(rows)
            if PRINT_FULL_TABLES:
                print_result_table(rows, b.name)

        if not PRINT_FULL_TABLES:
            print("ferdig")

        if RUN_FIGURES:
            slug = _scenario_slug(scenario_name)

            plot_pf_profiles(
                data,
                scenario_name,
                point_name,
                save_path=FIGURE_DIR / f"pf_profile_{slug}.png" if SAVE_FIGURES else None,
                show=SHOW_FIGURES,
            )

            if RUN_PROFILE_WITH_BUILDING_SECTION:
                plot_pf_profile_with_building_section(
                    data,
                    scenario_name,
                    point_name,
                    building_image_path=None,
                    show_height_in_labels=False,
                    save_path=FIGURE_DIR / f"pf_profile_section_{slug}.png" if SAVE_FIGURES else None,
                    show=SHOW_FIGURES,
                )

            # ── Figur 2: stolpediagram PF per etasje ────────────────
            for facade_label, _, _, fslug in _facade_configs:
                plot_pf_bars(
                    data, scenario_name, facade_label,
                    save_path=FIGURE_DIR / f"pf_bar_{fslug}_{slug}.png" if SAVE_FIGURES else None,
                    show=SHOW_FIGURES,
                )

            # ── Figur 3: PF-ratio stubbloft/utskifting ────────────────
            for facade_label, _, _, fslug in _facade_configs:
                plot_pf_ratio(
                    data, scenario_name, facade_label,
                    save_path=FIGURE_DIR / f"pf_ratio_{fslug}_{slug}.png" if SAVE_FIGURES else None,
                    show=SHOW_FIGURES,
                )

            # ── Figur 4: prosentvis PF-reduksjon ─────────────────────
            for facade_label, _, _, fslug in _facade_configs:
                plot_pf_reduction_pct(
                    data, scenario_name, facade_label,
                    save_path=FIGURE_DIR / f"pf_reduction_{fslug}_{slug}.png" if SAVE_FIGURES else None,
                    show=SHOW_FIGURES,
                )

            # ── Figur 5: ground/roof-andeler for 4 hovedvarianter ────
            plot_ground_roof_main_buildings(
                data, scenario_name,
                save_path=FIGURE_DIR / f"ground_roof_main_{slug}.png" if SAVE_FIGURES else None,
                show=SHOW_FIGURES,
            )

            # ── Figur 1 + 7: sensitivitetsbånd med scenario i tittel ─
            for facade_label, M_wall, M_fwall, fslug in _facade_configs:

                def _make_b(M_fl, _lbl=facade_label, _w=M_wall, _fw=M_fwall):
                    return replace(
                        scenario,
                        name=f"{_lbl} | band",
                        M_floor_kg_m2=M_fl,
                        M_wall_kg_m2=_w,
                        M_foundation_wall_kg_m2=_fw,
                    )

                _rows_sm = calculate_rows_for_points(scenario_name, point_name, points, _make_b(M_STUBB_BEVART))
                _rows_sl = calculate_rows_for_points(scenario_name, point_name, points, _make_b(M_STUBB_LOW))
                _rows_sh = calculate_rows_for_points(scenario_name, point_name, points, _make_b(M_STUBB_HIGH))
                _rows_um = calculate_rows_for_points(scenario_name, point_name, points, _make_b(M_FULL_UTSKIFTING))
                _rows_ul = calculate_rows_for_points(scenario_name, point_name, points, _make_b(M_UTSK_LOW))
                _rows_uh = calculate_rows_for_points(scenario_name, point_name, points, _make_b(M_UTSK_HIGH))

                plot_pf_band_per_scenario(
                    facade_label, scenario_name, point_name,
                    _rows_sm, _rows_sl, _rows_sh, _rows_um, _rows_ul, _rows_uh,
                    include_basement=True,
                    save_path=FIGURE_DIR / f"pf_band_{fslug}_{slug}.png" if SAVE_FIGURES else None,
                    show=SHOW_FIGURES,
                )

                plot_pf_band_per_scenario(
                    facade_label, scenario_name, point_name,
                    _rows_sm, _rows_sl, _rows_sh, _rows_um, _rows_ul, _rows_uh,
                    include_basement=False,
                    save_path=FIGURE_DIR / f"pf_band_nkjeller_{fslug}_{slug}.png" if SAVE_FIGURES else None,
                    show=SHOW_FIGURES,
                )

            if RUN_EXTRA_FIGURES:
                plot_location_factor(
                    data, scenario_name, point_name,
                    save_path=FIGURE_DIR / f"location_factor_{slug}.png" if SAVE_FIGURES else None,
                    show=SHOW_FIGURES,
                )
                plot_pf_heatmap(
                    data, scenario_name, point_name,
                    save_path=FIGURE_DIR / f"pf_heatmap_{slug}.png" if SAVE_FIGURES else None,
                    show=SHOW_FIGURES,
                )
                plot_indoor_dose_components(
                    data, scenario_name, point_name,
                    save_path=FIGURE_DIR / f"dose_components_{slug}.png" if SAVE_FIGURES else None,
                    show=SHOW_FIGURES,
                )
                plot_floor_crossings(
                    data, scenario_name, point_name,
                    save_path=FIGURE_DIR / f"floor_crossings_{slug}.png" if SAVE_FIGURES else None,
                    show=SHOW_FIGURES,
                )
                plot_ground_roof_share_by_floor(
                    data, scenario_name, point_name,
                    save_path=FIGURE_DIR / f"ground_roof_share_{slug}.png" if SAVE_FIGURES else None,
                    show=SHOW_FIGURES,
                )

    df = rows_to_dataframe(all_rows)
    csv_path = output_dir / "pf_master_results_refaktorert.csv"
    df.to_csv(csv_path, index=False)

    # ── Kompakt nøkkelfunn for representativt scenario (første scenario) ──
    if not PRINT_FULL_TABLES:
        _ref_sname = scenarios[0].name
        _ref_data  = _key_data.get(_ref_sname, {})
        from .calculations import rows_to_plot_df

        def _pf_for(building_name: str, floor_label: str) -> str:
            rows = _ref_data.get(building_name, [])
            if not rows:
                return "—"
            df_tmp = rows_to_plot_df(rows)
            row = df_tmp[df_tmp["Etasje"] == floor_label]
            return f"{float(row['PF'].values[0]):.1f}" if len(row) else "—"

        def _ratio(sname: str, uname: str, fl: str) -> str:
            try:
                ps = float(_pf_for(sname, fl).replace("—", "nan"))
                pu = float(_pf_for(uname, fl).replace("—", "nan"))
                return f"{ps/pu:.1f}×" if pu > 0 else "—"
            except Exception:
                return "—"

        print()
        print(f"Resultater lagret:")
        print(f"  {csv_path}  ({len(df)} rader)")
        print()
        print(f"Nøkkelfunn — {_ref_sname}:")
        for facade_label, _, _, _ in _facade_configs:
            sn = f"{facade_label} | stubbloft bevart"
            un = f"{facade_label} | full utskifting + mineralull"
            pf2s = _pf_for(sn, "2. etasje");  pf2u = _pf_for(un, "2. etasje")
            pf3s = _pf_for(sn, "3. etasje");  pf3u = _pf_for(un, "3. etasje")
            pf4s = _pf_for(sn, "4. etasje");  pf4u = _pf_for(un, "4. etasje")
            r2 = _ratio(sn, un, "2. etasje")
            r3 = _ratio(sn, un, "3. etasje")
            r4 = _ratio(sn, un, "4. etasje")
            print(f"  {facade_label}  stubb/utsk:")
            print(f"    2. etasje  PF {pf2s} / {pf2u}  (ratio {r2})")
            print(f"    3. etasje  PF {pf3s} / {pf3u}  (ratio {r3})")
            print(f"    4. etasje  PF {pf4s} / {pf4u}  (ratio {r4})")
        print()

    # ------------------------------------------------------------
    # PF-band med sensitivitet per fasadetype
    # ------------------------------------------------------------
    if RUN_FIGURES:
        # Samme spenn som i Hovedberegning: 6–10 cm leire → 174–230 kg/m²
        M_STUBB_LOW  = stubbloft_floor_mass_from_clay_kg_m2(STUBB_CLAY_THICKNESS_LOW_M)
        M_STUBB_HIGH = stubbloft_floor_mass_from_clay_kg_m2(STUBB_CLAY_THICKNESS_HIGH_M)
        M_UTSK_LOW,   M_UTSK_HIGH  = 49.0,  76.0

        band_scenario = scenarios[0]

        facade_configs = [
            ("Murbygg", 450.0, 650.0, "murbygg"),
            ("Trebygg", 80.0,  350.0, "trebygg"),
        ]

        for facade_label, M_wall, M_fwall, fname_slug in facade_configs:

            def _make(M_fl, _lbl=facade_label, _w=M_wall, _fw=M_fwall):
                return replace(
                    band_scenario,
                    name=f"{_lbl} | band",
                    M_floor_kg_m2=M_fl,
                    M_wall_kg_m2=_w,
                    M_foundation_wall_kg_m2=_fw,
                )

            rows_sm = calculate_rows_for_points(band_scenario.name, point_name, points, _make(M_STUBB_BEVART))
            rows_sl = calculate_rows_for_points(band_scenario.name, point_name, points, _make(M_STUBB_LOW))
            rows_sh = calculate_rows_for_points(band_scenario.name, point_name, points, _make(M_STUBB_HIGH))
            rows_um = calculate_rows_for_points(band_scenario.name, point_name, points, _make(M_FULL_UTSKIFTING))
            rows_ul = calculate_rows_for_points(band_scenario.name, point_name, points, _make(M_UTSK_LOW))
            rows_uh = calculate_rows_for_points(band_scenario.name, point_name, points, _make(M_UTSK_HIGH))

            plot_pf_sensitivity_band_by_facade(
                facade_label,
                rows_sm, rows_sl, rows_sh,
                rows_um, rows_ul, rows_uh,
                save_path=FIGURE_DIR / f"pf_band_{fname_slug}.png" if SAVE_FIGURES else None,
                show=SHOW_FIGURES,
            )

        _status("Sensitivitetsbånd (M_floor-spenn)")

        # ── Figur 6: kjelleråpning-sensitivitet (alle scenarioer) ────
        for facade_label, _, _, fslug in _facade_configs:
            plot_basement_opening_sensitivity(
                all_scenario_data, facade_label,
                save_path=FIGURE_DIR / f"pf_kjeller_sensitivity_{fslug}.png" if SAVE_FIGURES else None,
                show=SHOW_FIGURES,
            )

        _status("Kjelleråpnings-sensitivitet")

    # ------------------------------------------------------------
    # Valgfrie sensitiviteter
    # ------------------------------------------------------------
    # ------------------------------------------------------------
    # Mekanismeanalyse: ground radius vs. PF-ratio og regime-skifte
    # Kjøres alltid når RUN_FIGURES er aktivt.
    # Tester R = 30, 60, 100, 150 m for å belyse:
    #   - om ratio ≈ 1 i 5. etasje er robust mot radius
    #   - hvilke etasjer som er mest sensitive for stubbloft-effekten
    #   - overgangen mellom ground-dominert og roof-dominert regime
    # ------------------------------------------------------------
    if RUN_FIGURES:
        _mech_df = run_ground_radius_sensitivity(
            base, scenarios[0], radii_m=(30.0, 60.0, 100.0, 150.0)
        )
        _mech_df.to_csv(output_dir / "pf_radius_mechanism.csv", index=False)

        for facade_label, _, _, fslug in _facade_configs:
            plot_pf_by_radius(
                _mech_df, facade_label,
                save_path=FIGURE_DIR / f"pf_ratio_radius_{fslug}.png" if SAVE_FIGURES else None,
                show=SHOW_FIGURES,
            )

        _status("Mekanismeanalyse (ground radius)")

    if RUN_RADIUS_SENSITIVITY:
        sensitivity_df = run_ground_radius_sensitivity(base, scenarios[0], radii_m=(30.0, 60.0, 100.0))
        sensitivity_df.to_csv(output_dir / "pf_ground_radius_sensitivity.csv", index=False)
        _status("Ground-radius-sensitivitet")

    if RUN_CLAY_SENSITIVITY:
        clay_df = run_clay_thickness_sensitivity(base, scenarios[0], clay_thicknesses_m=(0.06, 0.08, 0.10))
        clay_df.to_csv(output_dir / "pf_clay_thickness_sensitivity.csv", index=False)
        _status("Leiretykkelse-sensitivitet")

    if RUN_ROOF_COMPARISON:
        roof_df = run_roof_model_comparison(base, scenarios[0])
        roof_df.to_csv(output_dir / "pf_roof_model_comparison.csv", index=False)
        for facade_label, fname_slug in [("Murbygg", "murbygg"), ("Trebygg", "trebygg")]:
            plot_roof_model_comparison(
                roof_df,
                facade_label=facade_label,
                save_path=FIGURE_DIR / f"pf_roof_comparison_{fname_slug}.png" if SAVE_FIGURES else None,
                show=SHOW_FIGURES,
            )
        _status("Takmodell-sammenligning (flat vs. saltak)")

    if RUN_BASEMENT_CEILING_SENSITIVITY:
        bsmt_df = run_basement_ceiling_sensitivity(base, scenarios, heavy_mass_kg_m2=300.0)
        bsmt_df.to_csv(output_dir / "pf_basement_ceiling_sensitivity.csv", index=False)
        plot_basement_ceiling_sensitivity(
            bsmt_df,
            heavy_mass_kg_m2=300.0,
            save_path=FIGURE_DIR / "pf_basement_ceiling_sensitivity.png" if SAVE_FIGURES else None,
            show=SHOW_FIGURES,
        )
        _status("Kjellerdekke-sensitivitet")

    if RUN_FLOOR_MASS_SENSITIVITY:
        fm_df = run_floor_mass_sensitivity(base, scenarios)
        fm_df.to_csv(output_dir / "pf_floor_mass_sensitivity.csv", index=False)
        _status("Etasjeskillermasse-sensitivitet")

    if RUN_DECK_IMPORTANCE:
        deck_df = run_floor_deck_importance(base, scenarios[0])
        deck_df.to_csv(output_dir / "pf_floor_deck_importance.csv", index=False)
        for facade_label, _, _, fslug in _facade_configs:
            plot_deck_importance_heatmap(
                deck_df, facade_label,
                save_path=FIGURE_DIR / f"pf_deck_heatmap_{fslug}.png" if SAVE_FIGURES else None,
                show=SHOW_FIGURES,
            )
            plot_deck_importance_lines(
                deck_df, facade_label,
                save_path=FIGURE_DIR / f"pf_deck_lines_{fslug}.png" if SAVE_FIGURES else None,
                show=SHOW_FIGURES,
            )
        _status("Per-dekke prioriteringsanalyse")

    if RUN_DECK_COMBINATION:
        comb_df = run_deck_combination_analysis(base, scenarios[0])
        comb_df.to_csv(output_dir / "pf_deck_combination_analysis.csv", index=False)
        for facade_label, _, _, fslug in _facade_configs:
            plot_deck_combo_heatmap(
                comb_df, facade_label,
                save_path=FIGURE_DIR / f"pf_deck_combo_heatmap_{fslug}.png" if SAVE_FIGURES else None,
                show=SHOW_FIGURES,
            )
            plot_best_combo_by_n(
                comb_df, facade_label,
                save_path=FIGURE_DIR / f"pf_deck_best_by_n_{fslug}.png" if SAVE_FIGURES else None,
                show=SHOW_FIGURES,
            )
            plot_deck_pair_heatmap(
                comb_df, facade_label, mode="preserve",
                save_path=FIGURE_DIR / f"pf_deck_pair_preserve_{fslug}.png" if SAVE_FIGURES else None,
                show=SHOW_FIGURES,
            )
            plot_deck_pair_heatmap(
                comb_df, facade_label, mode="remove",
                save_path=FIGURE_DIR / f"pf_deck_pair_remove_{fslug}.png" if SAVE_FIGURES else None,
                show=SHOW_FIGURES,
            )
            plot_deck_shapley(
                comb_df, facade_label,
                save_path=FIGURE_DIR / f"pf_deck_shapley_{fslug}.png" if SAVE_FIGURES else None,
                show=SHOW_FIGURES,
            )
        # Robusthetssjekk: Målsett A (2.-4. etasje), B (Kjeller), C (samlet)
        for facade_label, _, _, fslug in _facade_configs:
            plot_multifocus_shapley(
                comb_df, facade_label,
                save_path=FIGURE_DIR / f"pf_deck_multifocus_shapley_{fslug}.png" if SAVE_FIGURES else None,
                show=SHOW_FIGURES,
            )
            plot_multifocus_best_by_n(
                comb_df, facade_label,
                save_path=FIGURE_DIR / f"pf_deck_multifocus_best_by_n_{fslug}.png" if SAVE_FIGURES else None,
                show=SHOW_FIGURES,
            )
            plot_deck_pair_heatmap_setA(
                comb_df, facade_label,
                save_path=FIGURE_DIR / f"pf_deck_pair_setA_{fslug}.png" if SAVE_FIGURES else None,
                show=SHOW_FIGURES,
            )
        _status("Dekke-kombinasjonsanalyse (32 kombinasjoner)")

    # ── Rapportklare hovedfigurer og tabeller ────────────────────────────────
    # EXPORT_MAIN_PDF = False: kun PNG (300 dpi) i figures/main/ — klar for Overleaf.
    # Sett True for å også generere figures/main_pdf/ (PDF-versjoner).
    EXPORT_MAIN_PDF = False
    if True:   # kjøres alltid — lager figures/main/ og output/tables/
        from .report_figures import create_report_figures
        _df = rows_to_dataframe(all_rows)
        create_report_figures(_df, FIGURE_DIR, output_dir, export_pdf=EXPORT_MAIN_PDF, base=base)

    # ── Organiser figurer i undermappestruktur ───────────────────────────────
    if SAVE_FIGURES:
        n_org, n_miss = organize_figures(FIGURE_DIR)
        n_figs = len(list(FIGURE_DIR.glob("*.png")))
        print(f"Figurer lagret i {FIGURE_DIR}/  ({n_figs} filer)")
        print(f"  Organisert: {n_org} kopiert til undermapper, {n_miss} manglende hovdfigurer")
        print(f"  Manifest:   {FIGURE_DIR}/FIGURE_MANIFEST.md")
    else:
        print("(Figurer ikke lagret — SAVE_FIGURES=False)")

    print(f"\nFerdig.")
    if PRINT_FULL_TABLES:
        print("\nTips: sett PRINT_FULL_TABLES = False for kompakt kjørerapport.")
