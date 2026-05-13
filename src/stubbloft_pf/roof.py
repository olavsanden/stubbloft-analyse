# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from .config import Building
from .geometry import floor_z_m, roof_z_m
from .physics import gamma_transmission
from .utils import trap_integral

def floors_above_point(floor_index: int, b: Building) -> int:
    # Antall etasjeskiller mellom målepunkt og tak.

    if floor_index == -1:
        return b.floors
    return b.floors - 1 - floor_index


def calculate_roof_path_mass(
    floor_index: int,
    vertical_dz: Union[np.ndarray, float],
    s: np.ndarray,
    b: Building,
    roof_path_factor: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int]:
    # Effektiv transmisjonsmasse fra kildeplan til målepunkt langs strålebane.
    #
    # Takmassen (M_roof_kg_m2) er en homogenisert arealmasse som representerer
    # takkonstruksjonen uten å skille mellom enkeltkomponenter.
    # Baneøkningen gjennom taket beregnes langs normalretningen til kildeflaten:
    #   flatt tak:  s/dz         (identisk med etasjeskillere)
    #   skråtak:    s/normal_dot (langs takflatens normalretning)
    # Etasjeskillerne behandles alltid som horisontale sjikt med baneøkning s/vertical_dz,
    # uavhengig av takmodell.
    #
    # basement_ceiling_mode = "heavy_basement_ceiling":
    #   Kjellerdekket (z=0) skilles ut med M_basement_ceiling_kg_m2.
    #   Gjelder kun floor_index == -1 (kjellerens roof shine-bane).
    #   Øvrige (floors_above - 1) etasjeskillere bruker M_floor_kg_m2.
    # basement_ceiling_mode = "same_as_floor":
    #   Alle etasjeskillere inkl. kjellerdekket bruker M_floor_kg_m2.

    floors_above = floors_above_point(floor_index, b)

    floor_path_factor = s / np.maximum(vertical_dz, 1e-6)
    if roof_path_factor is None:
        roof_path_factor = floor_path_factor

    if b.floor_masses_kg_m2 is not None:
        # Per-dekke masser: summer de dekkene som befinner seg over målepunktet.
        # Indeks 0 = kjellerdekke (z=0). For floor_index = k ≥ 0 er dekkene
        # over z_p de med indeks k+1 til floors-1. For kjeller (floor_index = -1)
        # er alle dekker (0..floors-1) over målepunktet.
        if floor_index == -1:
            total_floor_mass = float(sum(b.floor_masses_kg_m2))
        else:
            total_floor_mass = float(sum(b.floor_masses_kg_m2[floor_index + 1:]))
        M = b.M_roof_kg_m2 * roof_path_factor + total_floor_mass * floor_path_factor
    elif (
        b.basement_ceiling_mode == "heavy_basement_ceiling"
        and floor_index == -1
        and floors_above > 0
    ):
        M = (
            b.M_roof_kg_m2 * roof_path_factor
            + b.M_basement_ceiling_kg_m2 * floor_path_factor
            + (floors_above - 1) * b.M_floor_kg_m2 * floor_path_factor
        )
    else:
        M = (
            b.M_roof_kg_m2 * roof_path_factor
            + floors_above * b.M_floor_kg_m2 * floor_path_factor
        )

    return M, floors_above


def integrate_flat_roof_shine(
    floor_index: int,
    x_p: float,
    y_p: float,
    b: Building,
    shielded: bool = True,
) -> Tuple[float, int]:
    # Integrerer roof shine fra flatt tak.

    z_p = floor_z_m(floor_index, b)
    dz = roof_z_m(b) - z_p

    if dz <= 0:
        return 0.0, 0

    x = np.linspace(-b.width_m / 2, b.width_m / 2, b.n_x)
    y = np.linspace(-b.length_m / 2, b.length_m / 2, b.n_y)

    X, Y = np.meshgrid(x, y, indexing="ij")

    s = np.sqrt((X - x_p) ** 2 + (Y - y_p) ** 2 + dz**2)
    geom = dz / s**3

    if shielded:
        M, floors_above = calculate_roof_path_mass(floor_index, dz, s, b)
        trans = gamma_transmission(b, M)
    else:
        trans = np.ones_like(geom)
        floors_above = 0

    total = float(trap_integral(trap_integral(trans * geom, y, axis=1), x, axis=0))
    return total, floors_above


def _sloped_roof_plane_contribution(
    floor_index: int,
    x_values: np.ndarray,
    y: np.ndarray,
    side: int,
    x_p: float,
    y_p: float,
    b: Building,
    shielded: bool,
) -> Tuple[float, int]:
    # Bidrag fra én takflate i et symmetrisk saltak.
    #
    # Modellen beskriver:
    #   - Kildegeometri: nedfallet antas å ligge på skrå takflater. Kildehøyde Z
    #     varierer fra z_eave (gesims) til z_eave + half_width × tan(pitch) (møne).
    #   - Geometrisk vekting: bidrag fra hvert kildelement vektes etter vinkelen
    #     mellom stråleretningen og takflatens normalvektor — normal_dot/s³ i stedet
    #     for dz/s³ som brukes for flatt tak.
    #   - Transmisjonsbane: roof_path_factor = s/normal_dot er effektiv banelengde
    #     gjennom den homogeniserte takmassen (M_roof_kg_m2) langs normalretningen.
    #
    # Modellen beskriver IKKE detaljert takkonstruksjon. M_roof_kg_m2 er én enkelt
    # homogenisert arealmasse — enkeltkomponenter, hulrom, luftesjikt og lagdeling
    # inngår ikke. Etasjeskillerne er alltid modellert som horisontale sjikt.
    #
    # side = -1 for venstre takflate, +1 for høyre. Mønet går langs y-aksen
    # (byggets lengderetning). roof_z_m(b) tolkes som gesimshøyde.

    z_p = floor_z_m(floor_index, b)
    pitch = np.deg2rad(b.roof_pitch_deg)
    m = np.tan(pitch)
    half_width = b.width_m / 2
    z_eave = roof_z_m(b)

    X, Y = np.meshgrid(x_values, y, indexing="ij")
    Z = z_eave + (half_width - np.abs(X)) * m

    vx = x_p - X
    vy = y_p - Y
    vz = z_p - Z
    s = np.sqrt(vx**2 + vy**2 + vz**2)

    # Normal inn i bygget fra takflaten. For venstre flate peker den mot +x og ned,
    # for høyre flate peker den mot -x og ned.
    nx = -side * m / np.sqrt(1.0 + m**2)
    nz = -1.0 / np.sqrt(1.0 + m**2)
    normal_dot = nx * vx + nz * vz
    normal_dot = np.maximum(normal_dot, 0.0)

    # Geometriledd: normal_dot/s³ = cos(θ_n)/s² der θ_n er vinkelen mellom
    # stråleretningen og takflatens innoverpekenende normalvektor.
    geom = normal_dot / np.maximum(s, 1e-12) ** 3

    # Dersom aktivitet angis per faktisk skrå takflate, må dxdy skaleres med 1/cos(pitch).
    # Standard er horizontal_projection fordi nedfall ofte beskrives per horisontal flate.
    if b.roof_deposition_mode == "surface":
        geom = geom / max(np.cos(pitch), 1e-6)

    vertical_dz = np.maximum(Z - z_p, 1e-6)

    if shielded:
        # Effektiv banelengde gjennom homogenisert takmasse langs takflatens normalretning.
        # s/normal_dot = 1/cos(θ_n). Begrenset til max_path_factor_floor for svært
        # granskrå innfallsvinkler.
        roof_path_factor = s / np.maximum(normal_dot, 1e-6)
        roof_path_factor = np.clip(roof_path_factor, 1.0, b.max_path_factor_floor)
        M, floors_above = calculate_roof_path_mass(
            floor_index, vertical_dz, s, b, roof_path_factor=roof_path_factor
        )
        trans = gamma_transmission(b, M)
    else:
        trans = np.ones_like(geom)
        floors_above = 0

    total = float(trap_integral(trap_integral(trans * geom, y, axis=1), x_values, axis=0))
    return total, floors_above


def integrate_sloped_roof_shine(
    floor_index: int,
    x_p: float,
    y_p: float,
    b: Building,
    shielded: bool = True,
) -> Tuple[float, int]:
    # Integrerer roof shine fra et symmetrisk saltak med møne langs byggets lengderetning.
    #
    # Modellen representerer kildegeometrien (skrå takflater, varierende kildehøyde fra
    # gesims til møne) og geometrisk vekting via takflatenes normalvektorer.
    # Takmassen er homogenisert (M_roof_kg_m2); detaljert takkonstruksjon modelleres ikke.
    # Nedfallsfordelingen styres av roof_deposition_mode ("horizontal_projection" eller
    # "surface").

    if b.roof_pitch_deg <= 1e-9:
        return integrate_flat_roof_shine(floor_index, x_p, y_p, b, shielded=shielded)

    y = np.linspace(-b.length_m / 2, b.length_m / 2, b.n_y)

    # Unngå dobbeltelling akkurat i mønet ved å bruke endpoint=False for venstre side.
    x_left = np.linspace(-b.width_m / 2, 0.0, max(2, b.n_x // 2), endpoint=False)
    x_right = np.linspace(0.0, b.width_m / 2, max(2, b.n_x // 2))

    total_left, floors_above = _sloped_roof_plane_contribution(
        floor_index, x_left, y, -1, x_p, y_p, b, shielded
    )
    total_right, _ = _sloped_roof_plane_contribution(
        floor_index, x_right, y, 1, x_p, y_p, b, shielded
    )

    return total_left + total_right, floors_above


def integrate_roof_shine(
    floor_index: int,
    x_p: float,
    y_p: float,
    b: Building,
    shielded: bool = True,
) -> Tuple[float, int]:
    # Integrerer roof shine fra valgt takmodell.

    if b.roof_model == "sloped":
        return integrate_sloped_roof_shine(floor_index, x_p, y_p, b, shielded=shielded)
    return integrate_flat_roof_shine(floor_index, x_p, y_p, b, shielded=shielded)
