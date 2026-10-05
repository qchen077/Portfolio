# Outputs

This folder stores generated figures, maps, and model results.

## Folder Structure

```text
outputs/
├── figures/
├── maps/
└── model_results/
```

## Suggested Contents

### `figures/`

Use this folder for static charts used in the README, presentation, or report.

Examples:

```text
crash_density_distribution.png
correlation_heatmap.png
high_vs_low_crash_route_tracts.png
zoning_share_comparison.png
landuse_share_comparison.png
model_comparison_table.png
```

### `maps/`

Use this folder for exported maps.

Examples:

```text
crash_density_map.html
major_route_exposure_map.html
built_environment_map.html
```

Large interactive HTML maps may be excluded from GitHub through `.gitignore`.

### `model_results/`

Use this folder for model summaries and comparison tables.

Examples:

```text
negative_binomial_model_summary.txt
model_comparison.csv
vif_results.csv
interaction_model_results.txt
```

## GitHub Recommendation

For a portfolio project, upload a few important static figures to GitHub, but avoid uploading very large raw data files or large interactive map files.
