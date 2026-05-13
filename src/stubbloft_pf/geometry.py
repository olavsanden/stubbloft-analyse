# -*- coding: utf-8 -*-
from __future__ import annotations

import warnings
from typing import List, Tuple, Union

import numpy as np

from .config import Building, Point

def floor_z_m(floor_index: int, b: Building) -> float:
    # Målepunktets høyde over terreng [m]. Kjeller har negativ z.

    if floor_index == -1:
        return -b.basement_depth_m + 1.1
    return floor_index * b.floor_height_m + 1.5


def roof_z_m(b: Building) -> float:
    # Takhøyde over terreng [m].

    return b.floors * b.floor_height_m


def floor_label(floor_index: int) -> str:
    # Lesbart etasjenavn.

    return "Kjeller" if floor_index == -1 else f"{floor_index + 1}. etasje"


def rectangle_radius(phi: Union[np.ndarray, float], b: Building) -> Union[np.ndarray, float]:
    # Avstand fra sentrum til rektangulær bygningskant i retning phi.

    # Fungerer både for skalar og numpy-array.

    c = np.abs(np.cos(phi))
    s = np.abs(np.sin(phi))
    eps = 1e-12

    return np.minimum(
        (b.width_m / 2) / np.maximum(c, eps),
        (b.length_m / 2) / np.maximum(s, eps),
    )


def basement_opening_fraction(b: Building) -> float:
    # Ekvivalent åpningsandel for eksponert kjellerstripe.

    # Dette modellerer ikke én retningsbestemt lyssjakt eksplisitt. Det er en forenklet
    # åpningsandel over en representativ rombredde.

    if b.basement_opening_height_m > b.basement_exposed_wall_height_m:
        warnings.warn(
            f"basement_opening_height_m ({b.basement_opening_height_m:.3f} m) er større enn "
            f"basement_exposed_wall_height_m ({b.basement_exposed_wall_height_m:.3f} m). "
            "Åpningsandelen kan bli overestimert fordi deler av åpningen ligger under eksponert kjellerstripe.",
            UserWarning,
            stacklevel=2,
        )

    A_opening = b.basement_opening_width_m * b.basement_opening_height_m
    A_exposed = b.basement_room_width_for_opening_fraction_m * b.basement_exposed_wall_height_m
    return float(np.clip(A_opening / max(A_exposed, 1e-9), 0.0, 0.8))


def ground_window_fraction_by_height(z_entry: np.ndarray, b: Building) -> np.ndarray:
    # Vindus-/åpningsandel som funksjon av treffhøyde i fasade, KUN for z >= 0.
    #
    # Kjelleråpningen (z < 0) håndteres separat i ground.py via T_below og
    # basement_opening_fraction. Returverdien her brukes bare for T_above,
    # som kun gjelder for stråler som treffer fasaden over terrengnivå.
    # Verdien for z < 0 er 0.0 og er uten effekt siden ground.py velger
    # T_below for alle below_grade-stråler.

    if not b.use_height_dependent_windows:
        return np.full_like(z_entry, b.window_fraction, dtype=float)

    f = np.zeros_like(z_entry, dtype=float)

    # 1. etasje: fra terrengnivå til én etasjehøyde.
    first_floor_band = (z_entry >= 0.0) & (z_entry < b.floor_height_m)
    f = np.where(first_floor_band, b.window_fraction, f)

    # 2. etasje.
    second_floor_band = (z_entry >= b.floor_height_m) & (z_entry < 2 * b.floor_height_m)
    f = np.where(second_floor_band, b.second_floor_window_factor * b.window_fraction, f)

    # 3. etasje og oppover.
    upper_floor_band = z_entry >= 2 * b.floor_height_m
    f = np.where(upper_floor_band, b.upper_floor_window_factor * b.window_fraction, f)

    return np.clip(f, 0.0, b.window_fraction)


def count_floor_crossings(z_entry: np.ndarray, z_p: float, b: Building) -> np.ndarray:
    # Antall etasjeskiller strålebanen krysser fra fasade til punkt.

    z_low = np.minimum(z_entry, z_p)
    z_high = np.maximum(z_entry, z_p)

    crossings = np.zeros_like(z_entry, dtype=float)

    # Kjellertak / dekke ved terrengnivå.
    crossings += ((z_low < 0.0) & (z_high > 0.0))

    for k in range(1, b.floors):
        z_slab = k * b.floor_height_m
        crossings += ((z_low < z_slab) & (z_high > z_slab))

    return crossings


def sum_crossed_deck_masses(z_entry: np.ndarray, z_p: float, b: Building) -> np.ndarray:
    # Total arealmasse for dekkene strålebanen krysser fra z_entry til z_p.
    # Brukes når b.floor_masses_kg_m2 er satt (per-dekke masser).
    # Indeks 0 = kjellerdekke ved z=0, indeks k = z = k × floor_height_m.

    z_low  = np.minimum(z_entry, z_p)
    z_high = np.maximum(z_entry, z_p)
    total  = np.zeros_like(z_entry, dtype=float)

    total += np.where((z_low < 0.0) & (z_high > 0.0), float(b.floor_masses_kg_m2[0]), 0.0)
    for k in range(1, b.floors):
        z_slab = k * b.floor_height_m
        total += np.where((z_low < z_slab) & (z_high > z_slab), float(b.floor_masses_kg_m2[k]), 0.0)

    return total


def find_first_facade_hit(
    X: np.ndarray,
    Y: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    b: Building,
) -> Tuple[np.ndarray, np.ndarray]:
    # Finner første treffpunkt med fasaden langs linjen fra kildepunkt til målepunkt.

    eps = 1e-9
    best_lam = np.full_like(X, np.inf, dtype=float)
    hit_name = np.full(X.shape, "none", dtype=object)

    faces = [
        ("left_long", "x", -b.width_m / 2),
        ("right_long", "x", b.width_m / 2),
        ("south_end", "y", -b.length_m / 2),
        ("north_end", "y", b.length_m / 2),
    ]

    for name, axis, coord in faces:
        lam = np.full_like(X, np.inf, dtype=float)

        if axis == "x":
            mask = np.abs(dx) > eps
            lam[mask] = (coord - X[mask]) / dx[mask]
            other = Y + lam * dy
            valid = (lam > 0) & (lam < 1) & (np.abs(other) <= b.length_m / 2)
        else:
            mask = np.abs(dy) > eps
            lam[mask] = (coord - Y[mask]) / dy[mask]
            other = X + lam * dx
            valid = (lam > 0) & (lam < 1) & (np.abs(other) <= b.width_m / 2)

        update = valid & (lam < best_lam)
        best_lam[update] = lam[update]
        hit_name[update] = name

    return best_lam, hit_name


def default_five_points_for_building(b: Building) -> List[Point]:
    # Fem representative målepunkter inne i bygget.

    x_offset = min(3.0, 0.25 * b.width_m)
    y_offset = min(8.0, 0.25 * b.length_m)

    return [
        ("Senter", 0.0, 0.0),
        ("Vest", -x_offset, 0.0),
        ("Øst", x_offset, 0.0),
        ("Sør", 0.0, -y_offset),
        ("Nord", 0.0, y_offset),
    ]
