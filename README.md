# Bayesian Spatial Inference of Earth's Inner-Core Structure

This repository contains the computational code developed for my MSc dissertation on Bayesian inference of Earth's inner-core structure using PKP travel-time observations.

The analysis considers spatially varying elastic parameters within a tilted transversely isotropic (TTI) representation of the inner core. Bayesian inference is used to estimate the Earth-model parameters and quantify their uncertainty. In addition to a baseline block-IID observational error model, spatially correlated observational errors are considered using a Matérn covariance model.

The repository contains code for mesh selection, baseline posterior inference, covariance-hyperparameter inference, hyperparameter marginalization, and sensitivity analysis.

## Main Scripts

### `main_wanying.py`

Modified version of the main analysis script used for the dissertation.

It contains the main data-processing and modelling functionality used by the dissertation analyses. It is provided separately as `main_wanying.py` so that the original `main.py` from the source repository remains unchanged.

### `meshSearchSep6.py`

Performs the mesh-resolution selection under the baseline block-IID observational error model.

The script computes the Bayesian log evidence for candidate radial and lateral mesh resolutions while keeping the Earth-model prior and observational error model fixed. The evidence comparison is used to select the spatial discretisation employed in the subsequent analyses.

The selected mesh used in the main dissertation analysis is a **10 × 5 radial-lateral mesh**.

### `block_iid_posterior_r10_l5_prior_0p01.py`

Computes the baseline Earth-model posterior for the selected **10 × 5 mesh** under the block-IID observational error model.

The observational uncertainties are allowed to differ between PKP phase groups, while observations within each group are treated as conditionally independent.

This analysis provides the baseline posterior against which the correlated-error results are compared.

### `tau_ell_posterior_r10_l5_prior_0p01.py`

Performs covariance-hyperparameter inference for the primary spatially correlated observational-error model.

The script evaluates the model over a grid of covariance hyperparameters:

- `tau` — magnitude of the correlated observational-error component;
- `ell` — spatial correlation length scale.

For each hyperparameter combination, the script evaluates the conditional Bayesian log evidence and constructs the joint posterior distribution of the covariance hyperparameters.

The block-IID model is also evaluated for comparison.

### `earth_posterior_marginalization_r10_l5.py`

Computes the final Earth-model posterior while accounting for uncertainty in the covariance hyperparameters.

Rather than conditioning the Earth model on only a single MAP estimate of `tau` and `ell`, the script marginalizes the conditional Earth-model posterior over the posterior distribution of the covariance hyperparameters.

The resulting marginalized posterior mean, covariance matrix, and posterior standard deviations are used in the main correlated-error analysis.

### `tau_ell_posterior_r10_l5_prior_0p01_antiparallel.py`

Performs the covariance-hyperparameter analysis using an orientation-invariant ray-path covariance construction.

This sensitivity analysis treats anti-parallel or reversed ray-path orientations equivalently when constructing the spatial covariance between observations.

The resulting hyperparameter posterior and Bayesian evidence can therefore be compared with those obtained using the primary covariance construction.

### `earth_posterior_marginalization_r10_l5_antiparallel.py`

Computes the hyperparameter-marginalized Earth-model posterior under the orientation-invariant covariance model.

This provides a sensitivity analysis for assessing how the inferred Earth-model structure and posterior uncertainty depend on the treatment of anti-parallel/reversed ray paths.

## Analysis Workflow

The main computational workflow is approximately:

1. **Mesh selection**  
   `meshSearchSep6.py`

2. **Baseline block-IID posterior**  
   `block_iid_posterior_r10_l5_prior_0p01.py`

3. **Correlated-error hyperparameter inference**  
   `tau_ell_posterior_r10_l5_prior_0p01.py`

4. **Hyperparameter-marginalized Earth-model posterior**  
   `earth_posterior_marginalization_r10_l5.py`

5. **Orientation-invariant sensitivity analysis**  
   `tau_ell_posterior_r10_l5_prior_0p01_antiparallel.py`

6. **Orientation-invariant posterior marginalization**  
   `earth_posterior_marginalization_r10_l5_antiparallel.py`

## Statistical Model

The inversion is formulated within a Bayesian framework. The Earth model consists of spatial perturbations to elastic parameters describing the inner core.

The baseline analysis assumes block-IID Gaussian observational errors, with separate uncertainty levels for different PKP phase groups.

The correlated-error model augments this structure with a spatial covariance component. A Matérn covariance model is used to represent correlations between travel-time observations, with covariance hyperparameters controlling the magnitude and spatial scale of the correlated errors.

For the correlated model, inference is performed both conditionally on covariance hyperparameters and after marginalizing over their posterior uncertainty.

## Sensitivity Analysis

An additional orientation-invariant covariance model is included to examine sensitivity to the representation of ray-path geometry.

The corresponding `*_antiparallel.py` scripts repeat the covariance-hyperparameter and posterior-marginalization calculations while treating reversed ray-path orientations equivalently.

These results can be compared with the primary correlated-error analysis to assess the robustness of the inferred inner-core structure.

## Software Environment

The project uses Python and the `uv` package/environment manager.

The original project contains a `pyproject.toml` and `uv.lock` for managing the Python environment and dependencies.

Typical scientific Python dependencies used by the analysis include:

- NumPy
- pandas
- SciPy
- Matplotlib
- Cartopy
- PyArrow

The exact environment should be reproduced from the corresponding project dependency files where available.

## Repository Structure

The principal dissertation files are:

```text
main_wanying.py
meshSearchSep6.py
block_iid_posterior_r10_l5_prior_0p01.py
tau_ell_posterior_r10_l5_prior_0p01.py
earth_posterior_marginalization_r10_l5.py
tau_ell_posterior_r10_l5_prior_0p01_antiparallel.py
earth_posterior_marginalization_r10_l5_antiparallel.py
```

Additional scripts and output files may be used for diagnostics, visualization, intermediate calculations, and preparation of dissertation figures.

## Notes

`main_wanying.py` contains the modified version of the main analysis code used for the dissertation. The filename is intentionally different from `main.py` so that the original version of `main.py` from the source repository is preserved.

The scripts in this repository were developed for the analyses reported in the MSc dissertation and may contain paths or configuration settings that need to be adjusted when running the code on another machine.
