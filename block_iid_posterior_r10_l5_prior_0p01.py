"""
Posterior inference for the selected 10 x 5 spherical mesh
under the baseline block-IID observational error model.

Outputs:
    outputs/baseline_iid_r10_l5/
        posterior_summary.csv
        posterior_mean.npy
        posterior_sd.npy
        posterior_covariance.npy
        summary.txt

Model:
    d = G m + epsilon

    m ~ N(0, 0.01 I)

    epsilon ~ N(0, Cd)

where Cd is the block-IID observational covariance.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.linalg import (
    cho_factor,
    cho_solve,
)

from raytracer import SphericalMesh

from tti.traveltimes.traveltimes import (
    calculate_path_direction_vector,
)

from main import (
    construct_forward_map,
    determine_weights,
)


# ============================================================
# SETTINGS
# ============================================================

DATA_FILE = Path(
    "data/brett2024_ic_traveltimes.parquet"
)

OUTPUT_DIR = Path(
    "outputs/baseline_iid_r10_l5"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Selected mesh
RADIAL_RESOLUTION = 10
LATERAL_RESOLUTION = 5

INNER_CORE_RADIUS = 1221.5

N_PARAMETERS_PER_CELL = 3


# Same Earth-model prior used in mesh search:
#
#     m ~ N(0, 0.01 I)
#
PRIOR_VARIANCE = 0.01
PRIOR_SD = np.sqrt(PRIOR_VARIANCE)


# ============================================================
# BLOCK-IID OBSERVATIONAL STANDARD DEVIATIONS
# ============================================================

PHASE_SIGMA_SECONDS = {
    "ab": 0.95,
    "bc": 0.63,
    "cd": 0.29,
    "df": 0.95,
}


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_parquet(
    DATA_FILE
)


# Fractional travel-time residual:
#
#     d_i = delta_t_i / T_i
#
data = (
    df["delta_t"]
    / df["inner_core_travel_time"]
).astype(float).to_numpy()


inner_core_travel_time = (
    df["inner_core_travel_time"]
    .astype(float)
    .to_numpy()
)


reference_phase = (
    df["reference_phase"]
    .astype(str)
    .to_numpy()
)


ic_in = np.stack(
    df["in_location"].to_numpy()
)

ic_out = np.stack(
    df["out_location"].to_numpy()
)


path_directions = (
    calculate_path_direction_vector(
        ic_in,
        ic_out,
    )
)


n_data = len(data)

print(
    "Number of observations =",
    n_data,
)


# ============================================================
# CONSTRUCT BLOCK-IID OBSERVATIONAL VARIANCE
# ============================================================

print()
print("=" * 70)
print("CONSTRUCTING BLOCK-IID OBSERVATIONAL VARIANCE")
print("=" * 70)


sigma_seconds = np.empty(
    n_data,
    dtype=float,
)


phase_counts = {
    "ab": 0,
    "bc": 0,
    "cd": 0,
    "df": 0,
}


for i, phase in enumerate(reference_phase):

    phase_lower = (
        phase
        .strip()
        .lower()
    )

    matched = False

    for phase_type, sigma in PHASE_SIGMA_SECONDS.items():

        if phase_lower.endswith(
            phase_type
        ):

            sigma_seconds[i] = sigma

            phase_counts[
                phase_type
            ] += 1

            matched = True

            break

    if not matched:

        raise ValueError(
            f"Unknown reference phase "
            f"'{phase}' at observation {i}."
        )


# Convert uncertainty in seconds to uncertainty in
# fractional travel-time residual:
#
#     sigma_fractional_i = sigma_seconds_i / T_i
#
sigma_fractional = (
    sigma_seconds
    / inner_core_travel_time
)


# Since Cd is diagonal, we do not need to construct the
# full 7668 x 7668 covariance matrix.
#
# Cd_ii = sigma_fractional_i^2
#
noise_variance = (
    sigma_fractional**2
)

noise_precision = (
    1.0
    / noise_variance
)


print(
    "Phase counts =",
    phase_counts,
)

print(
    "Fractional noise SD range =",
    sigma_fractional.min(),
    "to",
    sigma_fractional.max(),
)


# ============================================================
# CONSTRUCT SELECTED 10 x 5 MESH
# ============================================================

print()
print("=" * 70)
print("CONSTRUCTING 10 x 5 MESH")
print("=" * 70)


mesh = SphericalMesh(
    INNER_CORE_RADIUS,
    RADIAL_RESOLUTION,
    LATERAL_RESOLUTION,
)


weights = determine_weights(
    mesh,
    ic_in,
    path_directions,
)


G = construct_forward_map(
    path_directions,
    weights,
)


n_obs_G, n_parameters = G.shape

if n_obs_G != n_data:

    raise ValueError(
        f"G has {n_obs_G} observations, "
        f"but data contain {n_data}."
    )


if (
    n_parameters
    % N_PARAMETERS_PER_CELL
    != 0
):

    raise ValueError(
        "Number of parameters is not divisible by 3."
    )


n_cells = (
    n_parameters
    // N_PARAMETERS_PER_CELL
)


print(
    "Forward matrix shape =",
    G.shape,
)

print(
    "Number of cells =",
    n_cells,
)

print(
    "Number of parameters =",
    n_parameters,
)


if n_cells != 450:

    print(
        "WARNING: expected 450 cells "
        "for the selected 10 x 5 mesh."
    )


if n_parameters != 1350:

    print(
        "WARNING: expected 1350 parameters "
        "for the selected 10 x 5 mesh."
    )


# ============================================================
# GAUSSIAN POSTERIOR
# ============================================================
#
# Prior:
#
#     m ~ N(0, Cm)
#
#     Cm = 0.01 I
#
#
# Likelihood:
#
#     d | m ~ N(Gm, Cd)
#
#
# Posterior covariance:
#
#     C_post
#       = (Cm^{-1} + G^T Cd^{-1} G)^{-1}
#
#
# Posterior mean:
#
#     mu_post
#       = C_post G^T Cd^{-1} d
#
# because the prior mean is zero.
#
# ============================================================

print()
print("=" * 70)
print("CALCULATING POSTERIOR")
print("=" * 70)


# ------------------------------------------------------------
# Posterior precision
# ------------------------------------------------------------

prior_precision = (
    1.0
    / PRIOR_VARIANCE
)


# Weight each row of G by the corresponding observational
# precision. This avoids constructing or inverting the full Cd.
#
weighted_G = (
    G
    * noise_precision[:, None]
)


posterior_precision = (
    prior_precision
    * np.eye(
        n_parameters
    )
    + G.T @ weighted_G
)


# ------------------------------------------------------------
# Right-hand side for posterior mean
# ------------------------------------------------------------

rhs = (
    G.T
    @ (
        noise_precision
        * data
    )
)


# ------------------------------------------------------------
# Cholesky factorisation of posterior precision
# ------------------------------------------------------------

chol = cho_factor(
    posterior_precision,
    lower=True,
    check_finite=False,
)


# ------------------------------------------------------------
# Posterior mean
# ------------------------------------------------------------

posterior_mean = cho_solve(
    chol,
    rhs,
    check_finite=False,
)


# ------------------------------------------------------------
# Posterior covariance
# ------------------------------------------------------------

posterior_covariance = cho_solve(
    chol,
    np.eye(
        n_parameters
    ),
    check_finite=False,
)


# Numerical symmetry cleanup
posterior_covariance = (
    0.5
    * (
        posterior_covariance
        + posterior_covariance.T
    )
)


# ------------------------------------------------------------
# Marginal posterior standard deviations
# ------------------------------------------------------------

posterior_variance = np.diag(
    posterior_covariance
)


# Protect against tiny negative numerical round-off
posterior_variance = np.maximum(
    posterior_variance,
    0.0,
)


posterior_sd = np.sqrt(
    posterior_variance
)


# ============================================================
# SUMMARY VALUES FOR DISSERTATION
# ============================================================

posterior_mean_min = float(
    np.min(
        posterior_mean
    )
)

posterior_mean_max = float(
    np.max(
        posterior_mean
    )
)

posterior_sd_min = float(
    np.min(
        posterior_sd
    )
)

posterior_sd_max = float(
    np.max(
        posterior_sd
    )
)


print()
print("=" * 70)
print("BASELINE BLOCK-IID POSTERIOR SUMMARY")
print("=" * 70)

print(
    "Mesh =",
    f"{RADIAL_RESOLUTION} x "
    f"{LATERAL_RESOLUTION}",
)

print(
    "Number of cells =",
    n_cells,
)

print(
    "Number of parameters =",
    n_parameters,
)

print(
    "Prior variance =",
    PRIOR_VARIANCE,
)

print(
    "Prior SD =",
    PRIOR_SD,
)

print()

print(
    "Posterior mean minimum =",
    posterior_mean_min,
)

print(
    "Posterior mean maximum =",
    posterior_mean_max,
)

print(
    "Posterior SD minimum =",
    posterior_sd_min,
)

print(
    "Posterior SD maximum =",
    posterior_sd_max,
)


# ============================================================
# SAVE FULL PARAMETER SUMMARY
# ============================================================

parameter_summary = pd.DataFrame(
    {
        "parameter_index":
            np.arange(
                n_parameters
            ),

        "posterior_mean":
            posterior_mean,

        "posterior_sd":
            posterior_sd,
    }
)


parameter_summary.to_csv(
    OUTPUT_DIR
    / "posterior_summary.csv",
    index=False,
)


# ============================================================
# SAVE NUMPY ARRAYS
# ============================================================

np.save(
    OUTPUT_DIR
    / "posterior_mean.npy",
    posterior_mean,
)

np.save(
    OUTPUT_DIR
    / "posterior_sd.npy",
    posterior_sd,
)

np.save(
    OUTPUT_DIR
    / "posterior_covariance.npy",
    posterior_covariance,
)


# ============================================================
# SAVE TEXT SUMMARY
# ============================================================

summary_text = f"""
Baseline block-IID posterior
============================

Mesh:
    radial resolution = {RADIAL_RESOLUTION}
    lateral resolution = {LATERAL_RESOLUTION}

Number of cells:
    {n_cells}

Number of inferred parameters:
    {n_parameters}

Prior:
    variance = {PRIOR_VARIANCE}
    SD = {PRIOR_SD}

Posterior mean range:
    {posterior_mean_min:.10f}
    to
    {posterior_mean_max:.10f}

Posterior SD range:
    {posterior_sd_min:.10e}
    to
    {posterior_sd_max:.10e}
"""


with open(
    OUTPUT_DIR / "summary.txt",
    "w",
) as f:

    f.write(
        summary_text
    )


print()
print("=" * 70)
print("OUTPUT FILES")
print("=" * 70)

print(
    OUTPUT_DIR.resolve()
)

print(
    OUTPUT_DIR
    / "posterior_summary.csv"
)

print(
    OUTPUT_DIR
    / "posterior_mean.npy"
)

print(
    OUTPUT_DIR
    / "posterior_sd.npy"
)

print(
    OUTPUT_DIR
    / "posterior_covariance.npy"
)

print(
    OUTPUT_DIR
    / "summary.txt"
)