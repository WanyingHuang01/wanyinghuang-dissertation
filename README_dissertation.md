# Dissertation Code

This repository contains the Bayesian inference and sensitivity-analysis code developed for the dissertation.

`meshSearchSep6.py` selects the spatial discretisation under the baseline block-IID observational error model. `block_iid_posterior_r10_l5_prior_0p01.py` computes the baseline posterior for the selected 10×5 mesh. `tau_ell_posterior_r10_l5_prior_0p01.py` infers the correlated-error covariance hyperparameters and evaluates model evidence, while `earth_posterior_marginalization_r10_l5.py` marginalizes over hyperparameter uncertainty to obtain the final Earth-model posterior. The corresponding `*_antiparallel.py` scripts repeat these analyses using an orientation-invariant ray-path covariance kernel as a sensitivity analysis.

## Scripts

### `main_wanying.py`

Modified version of the main analysis script used for the dissertation. It is provided separately so that the original `main.py` in the repository remains unchanged.

### `meshSearchSep6.py`

Performs the mesh-resolution selection under the baseline block-IID observational error model. Computes the Bayesian log evidence for candidate radial and lateral mesh resolutions while keeping the Earth-model prior and observational error model fixed. This search is used to select the 10×5 mesh used in the subsequent analyses.

### `block_iid_posterior_r10_l5_prior_0p01.py`

Computes the baseline Earth-model posterior for the selected 10×5 mesh under the block-IID observational error model. Uses phase-dependent observational uncertainties and an Earth-model prior variance of 0.01.

### `tau_ell_posterior_r10_l5_prior_0p01.py`

Main covariance-hyperparameter analysis for the selected 10×5 inner-core mesh. Evaluates the correlated observational-error model over the 72-point (τ, ℓ) grid, computes the conditional log evidence and joint posterior distribution of the covariance hyperparameters, and evaluates the block-IID baseline for comparison.

### `earth_posterior_marginalization_r10_l5.py`

Computes the final Earth-model posterior for the primary correlated-error model by marginalizing over uncertainty in τ and ℓ across the 72-point hyperparameter grid. Saves the marginalized posterior mean, covariance, and standard deviation.

### `tau_ell_posterior_r10_l5_prior_0p01_antiparallel.py`

Orientation-invariant sensitivity analysis for the selected 10×5 mesh. Repeats the covariance-hyperparameter and evidence calculations using an anti-parallel-aware ray-path covariance kernel that treats reversed path orientations equivalently.

### `earth_posterior_marginalization_r10_l5_antiparallel.py`

Computes the hyperparameter-marginalized Earth-model posterior under the orientation-invariant covariance model. Used to assess the sensitivity of the inferred Earth model to anti-parallel/reversed ray paths.