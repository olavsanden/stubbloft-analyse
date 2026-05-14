# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Tuple

import numpy as np

from .config import Building
from .geometry import (
    basement_opening_fraction,
    count_floor_crossings,
    find_first_facade_hit,
    floor_z_m,
    ground_window_fraction_by_height,
    rectangle_radius,
    sum_crossed_deck_masses,
)
from .physics import gamma_transmission
from .utils import trap_integral

def calculate_ground_shine_transmission(
    r: np.ndarray,
    phi: np.ndarray,
    x_p: float,
    y_p: float,
    floor_index: int,
    b: Building,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Beregner transmisjon fra terrengpunkt til innendørs målepunkt.

    # Returnerer:
    # - samlet transmisjon
    # - antall kryssede etasjeskiller
    # - effektiv jordbane [m]

    z_p = floor_z_m(floor_index, b)

    X = r * np.cos(phi)
    Y = r * np.sin(phi)

    dx = x_p - X
    dy = y_p - Y

    s_total = np.sqrt(dx**2 + dy**2 + z_p**2)

    lam, hit_name = find_first_facade_hit(X, Y, dx, dy, b)
    valid = np.isfinite(lam)

    x_entry = np.zeros_like(r)
    y_entry = np.zeros_like(r)
    z_entry = np.zeros_like(r)

    x_entry[valid] = X[valid] + lam[valid] * dx[valid]
    y_entry[valid] = Y[valid] + lam[valid] * dy[valid]
    z_entry[valid] = lam[valid] * z_p

    normal_component = np.where(
        (hit_name == "left_long") | (hit_name == "right_long"),
        np.abs(dx),
        np.abs(dy),
    )

    wall_path_factor = np.ones_like(r)
    wall_path_factor[valid] = s_total[valid] / np.maximum(normal_component[valid], 1e-6)
    wall_path_factor = np.clip(wall_path_factor, 1.0, b.max_path_factor_wall)

    below_grade = z_entry < 0.0
    exposed_basement_band = (
        (z_entry >= -b.basement_exposed_wall_height_m)
        & (z_entry < 0.0)
        & valid
    )

    raw_soil_length = np.zeros_like(r)
    soil_mask = below_grade & valid
    raw_soil_length[soil_mask] = np.sqrt(
        (x_entry[soil_mask] - X[soil_mask]) ** 2
        + (y_entry[soil_mask] - Y[soil_mask]) ** 2
        + z_entry[soil_mask] ** 2
    )

    # Over terreng: fasade med høydeavhengig vindusandel.
    f_window_ground = ground_window_fraction_by_height(z_entry, b)

    M_wall_eff = np.where(
        below_grade,
        b.M_foundation_wall_kg_m2,
        b.M_wall_kg_m2,
    )

    T_wall = gamma_transmission(b, M_wall_eff * wall_path_factor)
    T_window = gamma_transmission(b, b.M_window_kg_m2 * wall_path_factor)
    T_above = f_window_ground * T_window + (1.0 - f_window_ground) * T_wall

    # Under terreng: jord + grunnmur. Åpning bare i eksponert kjellerstripe.
    #
    # T_closed_below: all stråling (inkl. lukket vegg) passerer jord + grunnmur.
    # T_open_below: avhenger av basement_opening_mode.
    #
    # "exposed_above_grade": åpningen antas å ligge over terrengnivå eller i en
    #   lyssjakt med lufteksponering. Stråling til åpningen trenger ikke passere
    #   ekstra jord utover det som er beregnet til fasadetreffpunktet.
    #   → T_open_below = f_b × T_bopen + (1 − f_b) × T_closed_below
    #
    # "below_grade": åpningen antas å ligge under terrengnivå. All stråling,
    #   også den gjennom vinduet, passerer jordsøylen T_soil.
    #   → T_open_below = T_soil × (f_b × T_bopen + (1 − f_b) × T_foundation)
    f_b = basement_opening_fraction(b)
    T_foundation = gamma_transmission(b, b.M_foundation_wall_kg_m2 * wall_path_factor)
    T_bopen = gamma_transmission(b, b.M_window_kg_m2 * wall_path_factor)
    T_soil = gamma_transmission(b, b.rho_soil_kg_m3 * raw_soil_length)

    T_closed_below = T_soil * T_foundation

    if b.basement_opening_mode == "below_grade":
        T_open_below = T_soil * (f_b * T_bopen + (1.0 - f_b) * T_foundation)
    else:  # "exposed_above_grade"
        T_open_below = f_b * T_bopen + (1.0 - f_b) * T_closed_below

    T_below = np.where(exposed_basement_band, T_open_below, T_closed_below)

    T_facade_soil = np.where(below_grade, T_below, T_above)

    n_cross = np.zeros_like(r)
    n_cross[valid] = count_floor_crossings(z_entry[valid], z_p, b)

    inside_length = np.zeros_like(r)
    inside_length[valid] = np.sqrt(
        (x_p - x_entry[valid]) ** 2
        + (y_p - y_entry[valid]) ** 2
        + (z_p - z_entry[valid]) ** 2
    )

    # 0.15 m-minimumet er et numerisk vern mot ekstremt small vertical distances
    # (f.eks. nær-horisontale stråler). For standardparametrene (etasjeskillere 3 m
    # fra hverandre) binder dette aldri i praksis — en stråle som krysser et
    # etasjedekke har alltid |Δz| ≫ 0.15 m.
    dz_inside = np.maximum(np.abs(z_p - z_entry), 0.15)

    floor_path_factor = np.ones_like(r)
    floor_path_factor[valid] = inside_length[valid] / dz_inside[valid]
    floor_path_factor = np.clip(floor_path_factor, 1.0, b.max_path_factor_floor)

    if b.floor_masses_kg_m2 is not None:
        crossed_mass = np.zeros_like(r)
        crossed_mass[valid] = sum_crossed_deck_masses(z_entry[valid], z_p, b)
        T_floors = gamma_transmission(b, crossed_mass * floor_path_factor)
    else:
        T_floors = gamma_transmission(b, n_cross * b.M_floor_kg_m2 * floor_path_factor)

    trans = T_facade_soil * T_floors
    trans[~valid] = 0.0

    # Diagnostic only: this effective soil length is averaged and reported in the result
    # table (diag_avg_soil_path_m). It is NOT used as input to the PF/groundshine
    # calculation — all transmission terms above use raw_soil_length directly.
    #
    # For "exposed_above_grade": the opening fraction (f_b) reaches air without soil,
    # so the diagnostikk reflects that the opening contributes zero soil path.
    # For "below_grade": the opening is also below grade and has the same soil path
    # as the closed wall, so raw_soil_length is used without deduction.
    if b.basement_opening_mode == "exposed_above_grade":
        effective_soil_length = np.where(
            exposed_basement_band,
            raw_soil_length * (1.0 - f_b),
            raw_soil_length,
        )
    else:  # "below_grade"
        effective_soil_length = raw_soil_length

    return trans, n_cross, effective_soil_length


def integrate_ground_shine(
    floor_index: int,
    x_p: float,
    y_p: float,
    b: Building,
    shielded: bool = True,
) -> Tuple[float, float, float]:
    # Integrerer ground shine rundt hele bygget.

    z_p = floor_z_m(floor_index, b)

    # Geometrisk referansehøyde for h_ref/s³-leddet.
    # For etasjer over bakken: den faktiske høyden z_p over terrenget.
    # For kjeller (z_p < 0): abs(z_p) gir den vertikale avstanden fra terreng ned
    # til målepunktet — mer konsistent med den faktiske geometrien enn fast 1.0 m.
    h_ref = z_p if z_p > 0 else abs(z_p)

    phis = np.linspace(0, 2 * np.pi, b.n_phi, endpoint=False)
    dphi = 2 * np.pi / b.n_phi

    total = 0.0
    total_geom = 0.0
    weighted_cross = 0.0
    weighted_soil = 0.0

    for phi0 in phis:
        r_min = float(rectangle_radius(phi0, b))
        r_max = r_min + b.ground_radius_m

        r = np.linspace(r_min, r_max, b.n_r)
        phi = np.full_like(r, phi0)

        X = r * np.cos(phi)
        Y = r * np.sin(phi)

        s = np.sqrt((X - x_p) ** 2 + (Y - y_p) ** 2 + h_ref**2)
        geom = h_ref / s**3

        if shielded:
            trans, n_cross, soil_length = calculate_ground_shine_transmission(
                r, phi, x_p, y_p, floor_index, b
            )
        else:
            trans = np.ones_like(r)
            n_cross = np.zeros_like(r)
            soil_length = np.zeros_like(r)

        integrand = trans * geom * r

        total += float(trap_integral(integrand, r)) * dphi
        total_geom += float(trap_integral(geom * r, r)) * dphi

        weighted_cross += float(trap_integral(n_cross * geom * r, r)) * dphi
        weighted_soil += float(trap_integral(soil_length * geom * r, r)) * dphi

    avg_cross = weighted_cross / max(total_geom, 1e-30)
    # Diagnostic average soil path — see comment in calculate_ground_shine_transmission.
    diag_avg_soil_path_m = weighted_soil / max(total_geom, 1e-30)

    return total, avg_cross, diag_avg_soil_path_m
