# Sea-Level Variability and Trend Analysis (1993–2023)

This repository contains a reproducible Python workflow for analyzing spatial variability and long-term trends in gridded sea level anomaly data across the African Atlantic coast.

The analysis targets problems central to sea-level projection and climate risk research, including regional variability, trend estimation under observational uncertainty, and physically interpretable regional diagnostics.

---

## Scientific Scope

The workflow computes:

- Spatial standard deviation of sea level anomalies to characterize regional variability.
- Linear sea-level trends using memory-efficient regression applied grid point–wise.
- Regional mean time series for dynamically distinct Atlantic subregions.
- Trend uncertainty and statistical significance for regional diagnostics.
- A combined multi-panel figure summarizing spatial and regional behavior.

The regional focus includes the Canary Current system, the Gulf of Guinea, and the Benguela upwelling system.

---

## Data Requirements

The code expects a gridded NetCDF sea level anomaly dataset with:

- Dimensions: `time × latitude × longitude`
- Variable name: `zos` (configurable in the script)
- Temporal coverage: 1993–2023 (or subset)

The workflow is compatible with satellite altimetry products and reanalysis-based sea level fields.

---

## Methodology

- Trends are estimated using ordinary least squares regression along the time dimension.
- Time is converted to decimal years to avoid bias from irregular sampling.
- Computations are vectorized and compatible with xarray and Dask.
- Trend magnitudes are reported in mm/yr.
- Regional diagnostics retain physical interpretability.

The structure is designed to be extensible for probabilistic sea-level frameworks and emulator-based projections.

---

## Output

The script generates a publication-quality figure summarizing:

- Spatial standard deviation of sea level anomalies
- Spatial distribution of linear sea-level trends
- Regional mean time series with trend estimates and confidence intervals

Example output:
`Combined_STD_Trends.png`

---

## Relevance

This repository is intended as a minimal, transparent example of regional sea-level analysis suitable for climate risk assessment, projection development, and postdoctoral research applications.
