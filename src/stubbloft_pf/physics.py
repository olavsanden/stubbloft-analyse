# -*- coding: utf-8 -*-
from __future__ import annotations

import warnings
from typing import Union

import numpy as np

from .config import Building

def simplified_buildup_factor(b: Building, M_kg_m2: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    # Forenklet buildup-korreksjon.

    # Dette er ikke en full energi- og materialavhengig buildup-tabell. Den fungerer som
    # en enkel korreksjon som gjør at transmisjonen blir noe høyere enn ren eksponentiell
    # attenuasjon.

    M_kg_m2 = np.maximum(M_kg_m2, 0.0)

    if not b.use_buildup:
        return np.ones_like(M_kg_m2, dtype=float) if isinstance(M_kg_m2, np.ndarray) else 1.0

    tau = b.mu_m2_kg * M_kg_m2
    return 1.0 + b.buildup_alpha * tau


def gamma_transmission(b: Building, M_kg_m2: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    # Transmisjon T = B * exp(-mu_m * M), begrenset til maksimum 1.
    #
    # np.maximum her er en ekstra sikkerhet for direkte kall; simplified_buildup_factor
    # gjør den samme klemmingen internt. np.minimum(T, 1.0) er teknisk sett redundant
    # for alpha >= 0 (T = 1 ved M = 0 og avtar monotont), men beholdes som forsvarslag.

    M_kg_m2 = np.maximum(M_kg_m2, 0.0)
    T = simplified_buildup_factor(b, M_kg_m2) * np.exp(-b.mu_m2_kg * M_kg_m2)
    return np.minimum(T, 1.0)


def effective_light_joist_floor_mass_kg_m2() -> float:
    # Forenklet arealmasse for lett trebjelkelag med to lag gipsplater, uten stubbloftsleire.
    # Brukes som underkomponent i light_reference_floor_mass_kg_m2.
    #
    # Bjelker: 45 × 250 mm tverrsnitt, 600 mm senter-til-senter avstand, furu/gran 500 kg/m³.
    # Gips: to lag standard gipsplater à 8,5 kg/m².
    # Gulvbord og undergulv er ikke inkludert — dette er et konservativt (lavt) estimat.

    joist_width_m          = 0.045   # bjelkebredde [m]
    joist_height_m         = 0.250   # bjelkehøyde [m]
    joist_spacing_m        = 0.600   # senter-til-senter avstand [m]
    wood_density_kg_m3     = 500.0   # furu/gran

    gypsum_layers          = 2
    gypsum_per_layer_kg_m2 = 8.5     # standard gipsplate

    wood_mass   = (joist_width_m * joist_height_m / joist_spacing_m) * wood_density_kg_m3
    gypsum_mass = gypsum_layers * gypsum_per_layer_kg_m2

    return wood_mass + gypsum_mass


def light_reference_floor_mass_kg_m2() -> float:
    # Nedre referansegrense: komplett men minimal ny trebjelkekonstruksjon.
    #
    # Bygger på effective_light_joist_floor_mass_kg_m2 (bjelker + 2×gips = 26.4 kg/m²)
    # og legger til:
    #   Gulvbord 22 mm softwood:  11.0 kg/m²
    #   Div. (dampsperrebane, parkett, festemateriell):  ~3 kg/m²
    #
    # Representerer et rehabilitert eller nybygget etasjeskille uten leire og uten
    # mineralull (absolutt nedre grense). Brukes som referansescenario, ikke som
    # representativ rehabilitering.

    return effective_light_joist_floor_mass_kg_m2() + 11.0 + 3.0   # ≈ 40 kg/m²


def stubbloft_clay_mass_kg_m2(
    clay_thickness_m: float,
    clay_density_kg_m3: float = 1400.0,
) -> float:
    # Arealmasse for selve leirlaget i stubbloft.

    # Tegningsgrunnlaget viser et spenn for stubbloftsleire. Derfor er dette lagt inn
    # som parameter i stedet for ett absolutt tall. Standard 1400 kg/m3 gir ca.
    # 140 kg/m2 ved 10 cm leire, som er i samme størrelsesorden som eldre litteratur.

    if clay_thickness_m < 0:
        raise ValueError("clay_thickness_m må være >= 0")
    if clay_density_kg_m3 <= 0:
        raise ValueError("clay_density_kg_m3 må være > 0")
    if clay_thickness_m > 0.20:
        warnings.warn(
            f"clay_thickness_m={clay_thickness_m:.3f} m er uvanlig høy for stubbloft "
            "(typisk 0.05–0.15 m). Kontroller at tykkelsen er oppgitt i meter, ikke cm.",
            UserWarning,
            stacklevel=2,
        )
    return clay_thickness_m * clay_density_kg_m3


def stubbloft_floor_mass_from_clay_kg_m2(
    clay_thickness_m: float,
    clay_density_kg_m3: float = 1400.0,
    other_floor_mass_kg_m2: float = 90.0,
) -> float:
    # Effektiv arealmasse for komplett eldre etasjeskiller med stubbloft.

    # other_floor_mass_kg_m2 representerer bjelker, bord, gulv, himling m.m. Verdien
    # 90 kg/m2 er valgt slik at 10 cm leire gir ca. 230 kg/m2 totalt når leira er
    # ca. 140 kg/m2. Dette gjør 6-10 cm til et nyttig usikkerhetsspenn.

    return other_floor_mass_kg_m2 + stubbloft_clay_mass_kg_m2(
        clay_thickness_m, clay_density_kg_m3
    )
