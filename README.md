# Stubbloft PF-modell — lagdelt fullversjon for VS Code

Dette er en lagdelt prosjektversjon av den fullverdige modellen.

## Kjøring

Åpne hele mappen i VS Code:

```text
File → Open Folder → stubbloft_pf_lagdelt_vscode
```

Installer pakker:

```bash
pip3 install -r requirements.txt
```

Kjør modellen:

```bash
python3 run.py
```

Resultater lagres i:

```text
output/pf_master_results_refaktorert.csv
```

## Mappestruktur

```text
stubbloft_pf_lagdelt_vscode/
├── run.py
├── requirements.txt
├── README.md
├── output/
├── figures/
└── src/
    └── stubbloft_pf/
        ├── config.py        # dataklasser og validering
        ├── utils.py         # trapesintegrasjon
        ├── physics.py       # transmisjon, buildup og arealmasser
        ├── geometry.py      # etasjer, fasade, vinduer og målepunkter
        ├── ground.py        # ground shine
        ├── roof.py          # roof shine, flatt tak og saltak
        ├── calculations.py  # PF/LF og dosekomponenter
        ├── scenarios.py     # scenarioer og bygningstyper
        ├── plots.py         # figurer med dobbel x-akse
        ├── sensitivity.py   # radius, leire og takmodell
        └── runner.py        # hovedkjøring og brytere
```

## Hvor endrer jeg hva?

- Endre geometri og arealmasser i `src/stubbloft_pf/config.py`
- Endre scenarioer i `src/stubbloft_pf/scenarios.py`
- Skru på flere figurer eller sensitiviteter i `src/stubbloft_pf/runner.py`
- Legg byggsnittbilder i `figures/`

## Viktig faglig merknad

Modellen er en forenklet deterministisk line-of-sight-modell.
Den er best egnet til relative sammenligninger mellom konstruksjoner,
etasjer og scenarioer, ikke som presis absolutt dosimetri.
