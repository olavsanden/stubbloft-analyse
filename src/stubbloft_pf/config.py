# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass(frozen=True)
class Building:
    # Samler geometri, arealmasser og numeriske parametere for ett bygg.

    name: str = "Bygg"

    # Geometri [m]
    floors: int = 5
    floor_height_m: float = 3.0
    basement_depth_m: float = 2.2
    basement_exposed_wall_height_m: float = 0.60
    width_m: float = 15.0
    length_m: float = 60.0
    ground_radius_m: float = 60.0

    # Takgeometri. "flat" gjenskaper opprinnelig modell.
    # "sloped" modellerer et symmetrisk saltak med møne langs byggets lengderetning.
    roof_model: str = "flat"
    roof_pitch_deg: float = 35.0
    roof_deposition_mode: str = "horizontal_projection"  # eller "surface"

    # Arealmasser [kg/m2]
    M_floor_kg_m2: float = 180.0
    M_roof_kg_m2: float = 35.0
    M_wall_kg_m2: float = 450.0
    M_foundation_wall_kg_m2: float = 650.0
    M_window_kg_m2: float = 20.0
    window_fraction: float = 0.28

    # Kjelleråpning / lyssjakt
    # Merk: modellen bruker dette som en ekvivalent åpningsandel i eksponert stripe.
    basement_opening_width_m: float = 0.40
    basement_opening_height_m: float = 0.60
    basement_room_width_for_opening_fraction_m: float = 3.0

    # Tolkning av kjelleråpningens plassering relativt til terreng.
    # "exposed_above_grade": åpningen er eksponert mot luft (synlig sokkel / lyssjakt
    #   over terrengnivå). Stråling når åpningen uten jordattenusasjon.
    # "below_grade": åpningen ligger under terrengnivå. All stråling til åpningen
    #   passerer jordsøylen (T_soil) før den treffer vindu/åpning.
    # Valget gir et modellspenn som dekker usikkerheten i kjellervindus geometri.
    basement_opening_mode: str = "exposed_above_grade"

    # Tolkning av kjellertakets konstruksjon (påvirker roof shine for kjeller).
    # "same_as_floor": kjellerdekket bruker samme M_floor_kg_m2 som øvrige etasjeskillere.
    #   Representerer tilfeller der dokumentasjon mangler, eller der kjellerdekket
    #   ikke er tyngre enn standard etasjeskille.
    # "heavy_basement_ceiling": kjellerdekket bruker M_basement_ceiling_kg_m2.
    #   Representerer tyngre konstruksjoner (betong, mur over kjeller, rehabilitert
    #   dekke). Bør tolkes som sensitivitetsverdi, ikke én sann modell.
    basement_ceiling_mode: str = "same_as_floor"

    # Arealmasse for kjellerdekke ved bruk av "heavy_basement_ceiling".
    # 300 kg/m2 er en sensitivitetsverdi, ikke en generell sannhet — den
    # representerer ett mulig tungt dekke (f.eks. armert betong 175 mm eller
    # kraftig fyllt tredekke). Bør alltid tilpasses faktisk bygningsdokumentasjon.
    M_basement_ceiling_kg_m2: float = 300.0

    # Per-dekke arealmasser [kg/m²], sortert nedenfra og opp.
    # Indeks 0 = kjellerdekke (z=0 m), indeks k = z = k × floor_height_m.
    # Lengde skal være lik b.floors.
    # Hvis None, brukes M_floor_kg_m2 for alle dekker (bakoverkompatibel standardoppførsel).
    # Brukes i sensitivitetsanalyser der enkeltdekker varieres uavhengig.
    floor_masses_kg_m2: Optional[tuple] = None

    # Jord
    rho_soil_kg_m3: float = 1700.0

    # Effektiv masseattenuasjonskoeffisient [m2/kg]
    mu_m2_kg: float = 0.007

    # Forenklet buildup-korreksjon
    use_buildup: bool = True
    buildup_alpha: float = 0.10

    # Høydeavhengige vindusfaktorer for ground shine
    use_height_dependent_windows: bool = True
    second_floor_window_factor: float = 0.5
    upper_floor_window_factor: float = 0.2

    # Numerikk
    n_x: int = 200
    n_y: int = 300
    n_phi: int = 300
    n_r: int = 260

    # Begrensning av skrå materialbaner
    max_path_factor_wall: float = 6.0
    max_path_factor_floor: float = 8.0


@dataclass(frozen=True)
class DoseComponents:
    # Dosebidrag for ett punkt og én etasje før PF/LF regnes ut.

    Dg_in: float
    Dr_in: float
    Dg_ref: float
    Dr_ref: float
    avg_floor_crossings: float
    diag_avg_soil_path_m: float  # diagnostic only — not an input to the PF calculation
    floor_decks_above: int


@dataclass(frozen=True)
class ResultRow:
    # Resultatrad for én etasje, ett bygg og ett punkt/punktsett.

    # Identifikasjon
    scenario: str
    building: str
    point: str
    label: str
    floor: int
    z_m: float

    # Primærresultater
    PF: float
    PF_raw: float
    LF: float
    ground_frac: float
    roof_frac: float

    # Diagnostikk fra geometrimodellen — ikke input til PF-beregningen
    avg_floor_crossings: float
    diag_avg_soil_path_m: float  # diagnostic only — reported for interpretation, not used in PF
    floor_decks_above: int

    # Råe dosekomponenter — for etterprøving og debugging.
    # Din = Dg_in + Dr_in (skjermet total innendørs dose).
    # Dref = Dg_ref + Dr_ref (uskjermet referansedose, samme geometri).
    # PF_raw = Dref / Din.
    Dg_in: float
    Dr_in: float
    Dg_ref: float
    Dr_ref: float
    Din: float
    Dref: float

    # Sentrale modellparametere — for reproduserbarhet av CSV-resultater uten
    # å måtte åpne kildekoden for å vite hvilken modell som ble kjørt.
    basement_opening_mode: str
    basement_ceiling_mode: str
    roof_model: str
    ground_radius_m: float
    M_floor_kg_m2: float
    M_roof_kg_m2: float
    M_wall_kg_m2: float
    M_foundation_wall_kg_m2: float
    M_basement_ceiling_kg_m2: float

Point = Tuple[str, float, float]

def validate_building(b: Building) -> None:
    # Validerer inputparametere. Bruker ValueError i stedet for assert.

    checks = {
        "floors > 0": b.floors > 0,
        "width_m > 0": b.width_m > 0,
        "length_m > 0": b.length_m > 0,
        "floor_height_m > 0": b.floor_height_m > 0,
        "basement_depth_m > 0": b.basement_depth_m > 0,
        # Målepunktet i kjeller beregnes som -basement_depth_m + 1.1 m. For at
        # z_p skal ligge under terrengnivå (z < 0) kreves basement_depth_m > 1.1 m.
        "basement_depth_m > 1.1 (kjellermålepunkt må ligge under terreng)": b.basement_depth_m > 1.1,
        "basement_exposed_wall_height_m >= 0": b.basement_exposed_wall_height_m >= 0,
        "M_floor_kg_m2 >= 0": b.M_floor_kg_m2 >= 0,
        "M_roof_kg_m2 >= 0": b.M_roof_kg_m2 >= 0,
        "M_wall_kg_m2 >= 0": b.M_wall_kg_m2 >= 0,
        "M_foundation_wall_kg_m2 >= 0": b.M_foundation_wall_kg_m2 >= 0,
        "M_window_kg_m2 >= 0": b.M_window_kg_m2 >= 0,
        "0 <= window_fraction <= 1": 0 <= b.window_fraction <= 1,
        "mu_m2_kg > 0": b.mu_m2_kg > 0,
        "rho_soil_kg_m3 > 0": b.rho_soil_kg_m3 > 0,
        "n_x >= 2": b.n_x >= 2,
        "n_y >= 2": b.n_y >= 2,
        "n_phi >= 4": b.n_phi >= 4,
        "n_r >= 2": b.n_r >= 2,
        "ground_radius_m > 0": b.ground_radius_m > 0,
        "basement_opening_mode in {exposed_above_grade, below_grade}": b.basement_opening_mode in {"exposed_above_grade", "below_grade"},
        "basement_ceiling_mode in {same_as_floor, heavy_basement_ceiling}": b.basement_ceiling_mode in {"same_as_floor", "heavy_basement_ceiling"},
        "M_basement_ceiling_kg_m2 >= 0": b.M_basement_ceiling_kg_m2 >= 0,
        "roof_model in {flat, sloped}": b.roof_model in {"flat", "sloped"},
        "floor_masses_kg_m2 length == floors (if set)": (
            b.floor_masses_kg_m2 is None or len(b.floor_masses_kg_m2) == b.floors
        ),
        "0 <= roof_pitch_deg <= 75": 0.0 <= b.roof_pitch_deg <= 75.0,
        "roof_deposition_mode in {horizontal_projection, surface}": b.roof_deposition_mode in {"horizontal_projection", "surface"},
        "max_path_factor_wall >= 1": b.max_path_factor_wall >= 1.0,
        "max_path_factor_floor >= 1": b.max_path_factor_floor >= 1.0,
        # Negativ alpha ville gi B < 1 og redusere transmisjonen under ren
        # eksponentiell attenuasjon, som er fysisk meningsløst.
        "buildup_alpha >= 0": b.buildup_alpha >= 0,
    }

    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError("Ugyldige bygningsparametere: " + ", ".join(failed))
