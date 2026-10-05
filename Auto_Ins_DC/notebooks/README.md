# Notebooks

This folder contains the main Jupyter notebooks for the DC Auto Insurance Spatial Risk and Crash Concentration project.

## Recommended Order

1. `01_data_preparation.ipynb`  
   Clean raw crash, roadway, traffic signal, census tract, and demographic data.

2. `02_eda.ipynb`  
   Explore crash concentration, roadway variables, demographic patterns, and tract-level distributions.

3. `03_modeling_analysis.ipynb`  
   Build the initial tract-level crash count modeling framework.

4. `04_major_route_exposure.ipynb`  
   Add major-route corridor exposure variables such as major-route length, density, and share.

5. `05_built_environment_land_use.ipynb`  
   Add zoning and existing land-use area-share variables and estimate the final staged models.

## Notes

- The final modeling unit is the census tract.
- The regression outcome is `total_crashes`.
- The model uses a log exposure offset: `log(area_km2 × n_years)`.
- Crash density is mainly used for EDA and visualization.
