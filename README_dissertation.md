# Dissertation Code

This repository contains the Bayesian inference, observational-error modelling,
hyperparameter marginalization, sensitivity-analysis, and plotting code developed
for the dissertation.

The analysis uses the `brett2024_ic_traveltimes.parquet` dataset. Spatial
discretisation is first selected under the baseline block-IID observational error
model. The selected 10×5 mesh contains 450 spatial cells and 1,350 Earth-model
parameters corresponding to fractional perturbations in A, C, and F.

The primary correlated-error analysis evaluates the covariance hyperparameters
τ and ℓ on a 72-point grid and marginalizes over their uncertainty to obtain the
final Earth-model posterior. An orientation-invariant, anti-parallel-aware
ray-path covariance construction is also considered as a sensitivity analysis.

## Data

The analysis uses the inner-core seismic travel-time dataset

`data/brett2024_ic_traveltimes.parquet`.

The dataset contains 7,668 PKPdf travel-time observations. The analysis uses
the observed travel-time residual `delta_t`, inner-core travel time
`inner_core_travel_time`, reference phase `reference_phase`, and the inner-core
entry and exit locations `in_location` and `out_location`.


## Scripts

### `main_wanying.py`

Modified version of the main analysis code used for the dissertation. It is
provided separately so that the original `main.py` in the repository remains
unchanged. It contains core utilities used by the dissertation-specific
analysis scripts, including construction of the forward model and
observational-error covariance.

### `meshSearchSep6.py`

Performs mesh-resolution selection under the baseline block-IID observational
error model. Bayesian log evidence is evaluated for candidate radial and
lateral mesh resolutions while keeping the Earth-model prior and observational
error model fixed. The resulting evidence comparison is used to select the
10×5 mesh adopted in the subsequent analyses.

### `block_iid_posterior_r10_l5_prior_0p01.py`

Computes the baseline Earth-model posterior for the selected 10×5 mesh under
the block-IID observational error model. It uses phase-dependent observational
uncertainties and an Earth-model prior variance of 0.01
(prior standard deviation 0.1).

### `tau_ell_posterior_r10_l5_prior_0p01.py`

Performs the primary covariance-hyperparameter analysis for the selected 10×5
mesh. The ordered-path correlated observational-error model is evaluated over
a 72-point (τ, ℓ) grid. The script computes the conditional log evidence and
joint posterior distribution of the covariance hyperparameters, evaluates the
block-IID baseline, and computes the grid-approximated hyperparameter-marginalized
model evidence over the evaluated hyperparameter grid.

### `earth_posterior_marginalization_r10_l5.py`

Computes the final Earth-model posterior for the primary correlated-error model
by marginalizing over uncertainty in τ and ℓ across the full 72-point discrete
hyperparameter grid. The marginalized posterior mean and covariance are computed
from the first two moments of the Gaussian mixture, thereby including both
within-hyperparameter and between-hyperparameter uncertainty. The script saves
the marginalized posterior mean, covariance, and standard deviation.

### `tau_ell_posterior_r10_l5_prior_0p01_antiparallel.py`

Performs the orientation-invariant sensitivity analysis for the selected 10×5
mesh. The covariance-hyperparameter and evidence calculations are repeated
using an anti-parallel-aware ray-path covariance kernel. Same-orientation and
reversed-orientation ray-path kernels are averaged and normalized to unit
diagonal, so that reversal of the path endpoints is treated equivalently.

### `earth_posterior_marginalization_r10_l5_antiparallel.py`

Computes the hyperparameter-marginalized Earth-model posterior under the
orientation-invariant covariance model using the corresponding 72-point
hyperparameter posterior. This analysis is used to assess the sensitivity of
the inferred Earth structure to the treatment of anti-parallel or reversed
ray paths.

### `Map_marg.py`

Generates spatial visualizations of the final full hyperparameter-marginalized
10×5 correlated-error Earth posterior. It produces fixed-radius maps at the
inner-core boundary (ICB), 800-km radius, and 400-km radius, together with
equatorial and meridional cross-sections of the posterior mean and posterior
standard deviation for δA/A, δC/C, and δF/F.

The script reads the final posterior results from:

`outputs/earth_posterior_marginalized_r10_l5_prior_0p01_full72/`

and saves the figures in the
`earth_spatial_figures_marginalized/` subdirectory.

### `analyze_earth_posteriors_r10_l5_full72_improved.py`

Compares the block-IID Earth-model posterior with the full
hyperparameter-marginalized correlated posterior for the selected 10×5 mesh.
It computes parameter-specific and overall summaries of changes in posterior
means and posterior standard deviations for δA/A, δC/C, and δF/F.

The script also generates separate posterior error-bar plots, parameter-specific
difference plots, and combined sensitivity plots. It operates on previously
computed posterior results and does not recompute the Bayesian inversion.

## Analysis workflow

The primary analysis workflow is:

1. `meshSearchSep6.py`
   → select the spatial discretisation under the block-IID model.

2. `block_iid_posterior_r10_l5_prior_0p01.py`
   → compute the block-IID posterior for the selected 10×5 mesh.

3. `tau_ell_posterior_r10_l5_prior_0p01.py`
   → evaluate the correlated-error covariance hyperparameters and model evidence.

4. `earth_posterior_marginalization_r10_l5.py`
   → marginalize over τ and ℓ to obtain the final correlated Earth-model posterior.

5. `Map_marg.py`
   → generate the final spatial maps and cross-sections.

6. `analyze_earth_posteriors_r10_l5_full72_improved.py`
   → compare the final marginalized correlated posterior with the block-IID
   baseline.

The corresponding anti-parallel scripts provide the orientation-invariant
ray-path sensitivity analysis. 

## Environment

The project environment is managed using [`uv`](https://docs.astral.sh/uv/).
The original `icanilg` project environment was retained and extended with the
additional scientific Python packages required for the dissertation analyses
and visualizations.

The dissertation-specific `pyproject.toml` adds the following dependencies:

- `matplotlib>=3.11.1`
- `scipy>=1.18.1`
- `cartopy>=0.25.0`
- `obspy>=1.5.1`
- `numpy>=2.5.1`

The corresponding `uv.lock` file records the resolved dependency environment
used by the project. The lock file was updated by `uv` after the additional
dependencies were introduced.

To reproduce the environment from the repository, run:

```bash
uv sync
