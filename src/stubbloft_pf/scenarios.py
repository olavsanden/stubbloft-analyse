# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Sequence

from .calculations import calculate_rows_for_points
from .config import Building, Point, ResultRow
from .physics import effective_light_joist_floor_mass_kg_m2, light_reference_floor_mass_kg_m2

# -----------------------------------------------------------------------
# Etasjeskillermasser — faglig begrunnet fra tegningsgrunnlag og analyse
# -----------------------------------------------------------------------
# Stubbloft bevart: tegningsbasert representativ verdi for bjelke 150×200 mm,
# 8 cm stubbloftsleire (1400 kg/m³), c/c ~600 mm + realistisk strukturmasse.
# Historiske/litteraturbaserte verdier kan ligge høyere (opp mot 202 kg/m²);
# spennet dekkes av run_floor_mass_sensitivity.
M_STUBB_BEVART: float = 145.0   # kg/m²

# Full utskifting: nye gulvåser c/c 300 mm (48×220 mm), 200 mm mineralull,
# moderne gulvoppbygning og himling. Alle gamle konstruksjonselementer fjernet.
# Spennet 49–76 kg/m² dekkes av run_floor_mass_sensitivity.
M_FULL_UTSKIFTING: float = 62.0  # kg/m²


def make_buildings(base: Building) -> List[Building]:
    # Seks bygningsvarianter: tre etasjeskillernivåer × to fasadetyper.
    #
    # Etasjeskillernivåer:
    #   stubbloft bevart    145 kg/m²  (tegningsbasert midtverdi)
    #   full utskifting      62 kg/m²  (nye åser + mineralull + modern oppbygning)
    #   referanse lett       ~40 kg/m²  (komplett minimal ny konstruksjon, nedre grense)
    #
    # Fasadetyper:
    #   Murbygg: M_wall=450, M_foundation_wall=650 kg/m²
    #   Trebygg: M_wall=80,  M_foundation_wall=350 kg/m²

    M_lett = light_reference_floor_mass_kg_m2()   # ≈ 40 kg/m²

    def mur(name: str, M_floor: float) -> Building:
        return replace(base, name=name, M_floor_kg_m2=M_floor,
                       M_wall_kg_m2=450.0, M_foundation_wall_kg_m2=650.0)

    def tre(name: str, M_floor: float) -> Building:
        return replace(base, name=name, M_floor_kg_m2=M_floor,
                       M_wall_kg_m2=80.0, M_foundation_wall_kg_m2=350.0)

    return [
        mur("Murbygg | stubbloft bevart",          M_STUBB_BEVART),
        mur("Murbygg | full utskifting + mineralull", M_FULL_UTSKIFTING),
        mur("Murbygg | referanse lett",             M_lett),
        tre("Trebygg | stubbloft bevart",          M_STUBB_BEVART),
        tre("Trebygg | full utskifting + mineralull", M_FULL_UTSKIFTING),
        tre("Trebygg | referanse lett",             M_lett),
    ]


def make_scenarios(base: Building) -> List[Building]:
    # Scenarioer for kjelleråpning.

    return [
        # Samsvarer med tekstlig beskrivelse i rapporten.
        replace(
            base,
            name="Kjellervindu 40x60 cm",
            basement_opening_width_m=0.40,
            basement_opening_height_m=0.60,
            M_window_kg_m2=20.0,
        ),
        # Beholder også opprinnelig 60x30-scenario, siden flere tidligere figurer brukte dette.
        replace(
            base,
            name="Kjellervindu 60x30 cm",
            basement_opening_width_m=0.60,
            basement_opening_height_m=0.30,
            M_window_kg_m2=20.0,
        ),
        replace(
            base,
            name="Luftespalte 15x15 cm",
            basement_opening_width_m=0.15,
            basement_opening_height_m=0.15,
            M_window_kg_m2=0.0,
        ),
    ]


def calculate_plot_data(
    scenario_name: str,
    point_name: str,
    points: Sequence[Point],
    buildings: Sequence[Building],
) -> Dict[str, List[ResultRow]]:
    # Beregner alle resultater én gang, slik at plotting ikke regner på nytt.

    return {
        b.name: calculate_rows_for_points(scenario_name, point_name, points, b)
        for b in buildings
    }
