# DC Auto Insurance Spatial Risk and Crash Concentration Analysis

## Project Overview

This project examines whether crash concentration in Washington, DC is associated with roadway conditions, major-route characteristics, built-environment context, and census tract demographic patterns.

The broader motivation is to provide spatial risk context for auto insurance equity discussions. This project does **not** directly test insurance pricing discrimination because it does not use policy-level premiums, insurance claims, or insurer pricing models.

Instead, public crash records are used as a proxy for auto insurance risk exposure.

## Project Objective

This project asks two main questions:

1. Are roadway conditions, major-route exposure, zoning, and existing land use associated with crash concentration across DC census tracts?
2. Do high-crash areas spatially overlap with census tracts that have higher Black or Hispanic population shares?

The unit of analysis is the census tract.

## Data Sources

The analysis combines multiple public datasets, including:

- Crashes in DC
- 2020 Census Tracts
- ACS 5-year demographic estimates
- Roadway Block / roadway network features
- Roadway Functional Classification
- DDOT Traffic Signals
- Zoning Boundaries
- Existing Land Use

Key derived variables include:

- Total crashes
- Annual crash density
- Road density
- Signal density
- Intersection density
- AADT
- Major-route density
- Major-route share
- Zoning area shares
- Existing land-use area shares
- Black, Hispanic, Asian, and White population shares

## Methodology

The project uses a tract-level spatial analysis workflow:

1. Clean and process public crash records.
2. Spatially join crash points to 2020 census tracts.
3. Aggregate crash counts to the census-tract level.
4. Calculate roadway, traffic signal, intersection, and major-route exposure variables.
5. Convert zoning and existing land-use polygons into tract-level area-share variables.
6. Merge roadway, built-environment, and demographic variables into a tract-level modeling table.
7. Estimate Negative Binomial regression models for crash counts.

Negative Binomial regression is used because crash counts are count data and are overdispersed.

The regression outcome is:

```text
total_crashes
```

The model includes a log exposure offset:

```text
log(area_km² × n_years)
```

This adjusts for differences in census tract area and observation time.

## Model Structure

The models are estimated in stages:

- **Model 1:** Roadway exposure baseline
- **Model 2:** Roadway exposure + major route exposure
- **Model 3:** Roadway exposure + major route exposure + zoning and existing land use
- **Model 4:** Full model with roadway, major route, zoning, land use, and demographic controls
- **Model 5:** Interaction check for roadway exposure × Black / Hispanic population share

Model 4 is used as the preferred main explanatory model because it best matches the project objective and includes roadway, major-route, built-environment, and demographic controls.

Model 5 is treated as a sensitivity check. It tests whether selected roadway effects differ by Black or Hispanic population share.

## Key Findings

The results suggest that crash concentration in DC is associated with roadway intensity and built-environment context.

Key findings:

- Road density and AADT are positively associated with expected crash counts.
- Zoning and existing land-use variables add explanatory value beyond roadway variables alone.
- Black population share remains positively associated with expected crash counts after controls.
- Hispanic and Asian population shares are not statistically significant in the full model.
- Interaction checks do not provide strong evidence that selected roadway effects are amplified by Black or Hispanic population share.

These findings should be interpreted as tract-level spatial associations, not causal effects.

## Limitations

This project has several important limitations:

- This is a tract-level association study, not a causal analysis.
- Crash records are used as a proxy for insurance claim-risk exposure, not actual claims or pricing data.
- The Crashes in DC dataset is updated regularly, so results reflect the processed data snapshot.
- Some missing tract-month records were treated as zero-crash months and should be validated.
- Roadway and demographic variables are limited by available public data.
- ACS 5-year demographic estimates describe tract-level population context, not individual drivers.

## Repository Structure

```text
.
├── Data/
│   ├── Crashes_in_DC.csv
│   ├── Census_Tracts_2020/
│   ├── Roadway_Block_Intersection.csv
│   ├── Roadway_Functional_Classification.csv
│   ├── DDOT_Traffic_Signals.csv
│   ├── Zoning_Boundaries/
│   └── Existing_Land_Use/
│
├── notebooks/
│   ├── Deliverable 1 - Data Preparation.ipynb
│   ├── Deliverable 2 - EDA.ipynb
│   ├── Deliverable 3 - Modeling Analysis.ipynb
│   ├── Deliverable 4 - Major Route Exposure Analysis.ipynb
│   └── Deliverable 5 - Built Environment and Land Use Extension.ipynb
│
├── outputs/
│   ├── figures/
│   ├── maps/
│   └── model_results/
│
├── README.md
└── requirements.txt
```

## Required Python Packages

The project uses Python for data cleaning, geospatial processing, visualization, and statistical modeling.

Core packages:

```text
pandas
numpy
geopandas
shapely
pyproj
fiona
rtree
matplotlib
seaborn
plotly
folium
contextily
statsmodels
scikit-learn
jupyter
notebook
ipykernel
openpyxl
```

Install packages with:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
pandas
numpy
geopandas
shapely
pyproj
fiona
rtree
matplotlib
seaborn
plotly
folium
contextily
statsmodels
scikit-learn
jupyter
notebook
ipykernel
openpyxl
```

## How to Run

1. Clone the repository.
2. Install the required Python packages.
3. Place raw data files in the `Data/` folder.
4. Run notebooks in order from Deliverable 1 to Deliverable 5.
5. Review final model results and visualizations in the `outputs/` folder.

## Final Takeaway

Roadway exposure and built-environment context are associated with crash concentration in Washington, DC. Black population share remains positively associated with expected crash counts after controls. However, the results are tract-level associations and do not prove causality, individual-level risk differences, or direct insurance pricing discrimination.
