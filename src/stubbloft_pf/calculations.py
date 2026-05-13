# -*- coding: utf-8 -*-
from __future__ import annotations

import warnings
from typing import List, Sequence

import numpy as np
import pandas as pd

from .config import Building, DoseComponents, Point, ResultRow, validate_building
from .geometry import floor_label, floor_z_m
from .ground import integrate_ground_shine
from .roof import integrate_roof_shine

def calculate_dose_components(
    floor_index: int,
    x_p: float,
    y_p: float,
    b: Building,
) -> DoseComponents:
    # Samler alle dosekomponenter for ett punkt og én etasje.

    Dg_in, avg_cross, avg_soil = integrate_ground_shine(
        floor_index, x_p, y_p, b, shielded=True
    )

    Dr_in, floors_above = integrate_roof_shine(
        floor_index, x_p, y_p, b, shielded=True
    )

    Dg_ref, _, _ = integrate_ground_shine(
        floor_index, x_p, y_p, b, shielded=False
    )

    Dr_ref, _ = integrate_roof_shine(
        floor_index, x_p, y_p, b, shielded=False
    )

    return DoseComponents(
        Dg_in=Dg_in,
        Dr_in=Dr_in,
        Dg_ref=Dg_ref,
        Dr_ref=Dr_ref,
        avg_floor_crossings=avg_cross,
        avg_soil_path_m=avg_soil,
        floor_decks_above=floors_above,
    )


def average_dose_components(components: Sequence[DoseComponents]) -> DoseComponents:
    # Middelverdi av dosekomponenter. PF beregnes etterpå fra middelverdiene.
    #
    # Dosekomponenter midles FØR PF beregnes, ikke omvendt. Årsak: et sterkt
    # eksponert punkt (lavt PF) bidrar med høy innendørs dose og dominerer
    # korrekt det romlige gjennomsnittet. Dersom man midlet PF direkte, ville
    # det eksponerte punktet vektes likt med bedre skjermede punkter og
    # gjennomsnittet ville overvurdere skjermingen (jf. Jensens ulikhet).

    if not components:
        raise ValueError("Kan ikke beregne middelverdi av tom komponentliste.")

    return DoseComponents(
        Dg_in=float(np.mean([c.Dg_in for c in components])),
        Dr_in=float(np.mean([c.Dr_in for c in components])),
        Dg_ref=float(np.mean([c.Dg_ref for c in components])),
        Dr_ref=float(np.mean([c.Dr_ref for c in components])),
        avg_floor_crossings=float(np.mean([c.avg_floor_crossings for c in components])),
        avg_soil_path_m=float(np.mean([c.avg_soil_path_m for c in components])),
        # floor_decks_above er bestemt av etasjeindeks alene (ikke av (x,y)-posisjon)
        # og er identisk for alle punkter i samme etasje — hentes fra første element.
        floor_decks_above=components[0].floor_decks_above,
    )


def result_row_from_components(
    scenario: str,
    point_name: str,
    floor_index: int,
    b: Building,
    c: DoseComponents,
    warn_if_pf_below_one: bool = True,
) -> ResultRow:
    # Regner PF/LF og komponentandeler fra dosekomponenter.

    Din  = c.Dg_in + c.Dr_in
    Dref = c.Dg_ref + c.Dr_ref

    PF_raw = Dref / max(Din, 1e-30)

    if warn_if_pf_below_one and PF_raw < 1.0:
        warnings.warn(
            f"PF_raw < 1 for {b.name}, {floor_label(floor_index)}: PF_raw={PF_raw:.3f}",
            RuntimeWarning,
            stacklevel=2,
        )

    # max(1.0, ...) er et numerisk sikkerhetsnett. PF_raw < 1 er fysisk umulig
    # med gyldige parametere (Din ≤ Dref alltid siden shielded ≤ unshielded).
    # Klemmingen hindrer at LF > 1 ved eventuelle numeriske avrundingsfeil.
    PF = max(1.0, PF_raw)
    LF = 1.0 / PF

    ground_frac = c.Dg_in / max(Din, 1e-30)
    roof_frac   = c.Dr_in / max(Din, 1e-30)

    return ResultRow(
        scenario=scenario,
        building=b.name,
        point=point_name,
        label=floor_label(floor_index),
        floor=floor_index,
        z_m=floor_z_m(floor_index, b),
        PF=PF,
        PF_raw=PF_raw,
        LF=LF,
        ground_frac=ground_frac,
        roof_frac=roof_frac,
        avg_floor_crossings=c.avg_floor_crossings,
        avg_soil_path_m=c.avg_soil_path_m,
        floor_decks_above=c.floor_decks_above,
        Dg_in=c.Dg_in,
        Dr_in=c.Dr_in,
        Dg_ref=c.Dg_ref,
        Dr_ref=c.Dr_ref,
        Din=Din,
        Dref=Dref,
        basement_opening_mode=b.basement_opening_mode,
        basement_ceiling_mode=b.basement_ceiling_mode,
        roof_model=b.roof_model,
        ground_radius_m=b.ground_radius_m,
        M_floor_kg_m2=b.M_floor_kg_m2,
        M_roof_kg_m2=b.M_roof_kg_m2,
        M_wall_kg_m2=b.M_wall_kg_m2,
        M_foundation_wall_kg_m2=b.M_foundation_wall_kg_m2,
        M_basement_ceiling_kg_m2=b.M_basement_ceiling_kg_m2,
    )


def calculate_rows_for_point(
    scenario: str,
    point_name: str,
    x_p: float,
    y_p: float,
    b: Building,
) -> List[ResultRow]:
    # Beregner resultatrader for ett målepunkt.

    validate_building(b)
    rows: List[ResultRow] = []

    for floor_index in [-1] + list(range(b.floors)):
        components = calculate_dose_components(floor_index, x_p, y_p, b)
        rows.append(result_row_from_components(scenario, point_name, floor_index, b, components))

    return rows


def calculate_rows_for_points(
    scenario: str,
    point_set_name: str,
    points: Sequence[Point],
    b: Building,
) -> List[ResultRow]:
    # Beregner resultatrader for et punktsett, f.eks. 5 målepunkter i rommet.

    # PF beregnes fra gjennomsnittlige dosekomponenter, ikke som aritmetisk middel
    # av PF-verdier per punkt. Se average_dose_components for begrunnelse.
    #
    # Alle punkter i settet vektes likt (uvektet aritmetisk middel av dosekomponenter).
    # Dette er en rimelig tilnærming for et symmetrisk kryss-mønster der ingen
    # enkeltposisjon er mer representativ enn de andre.

    validate_building(b)
    rows: List[ResultRow] = []

    for floor_index in [-1] + list(range(b.floors)):
        components = [
            calculate_dose_components(floor_index, x_p, y_p, b)
            for _, x_p, y_p in points
        ]
        avg_components = average_dose_components(components)
        rows.append(result_row_from_components(scenario, point_set_name, floor_index, b, avg_components))

    return rows


def rows_to_dataframe(rows: Sequence[ResultRow]) -> pd.DataFrame:
    # Konverterer resultatrader til DataFrame.

    return pd.DataFrame([r.__dict__ for r in rows])


def rows_to_plot_df(rows: Sequence[ResultRow]) -> pd.DataFrame:
    # Forenklet DataFrame for plotting.

    return pd.DataFrame(
        [
            {
                "Etasje": r.label,
                "z": r.z_m,
                "PF": r.PF,
                "PF_raw": r.PF_raw,
                "LF": r.LF,
                "Ground%": 100 * r.ground_frac,
                "Roof%": 100 * r.roof_frac,
                "Cross": r.avg_floor_crossings,
                "Jordbane_m": r.avg_soil_path_m,
                "Etasjeskiller_over": r.floor_decks_above,
            }
            for r in rows
        ]
    )
