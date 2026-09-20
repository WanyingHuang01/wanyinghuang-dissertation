from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from linear_gaussian import (
    GaussianComponent,
    calc_posterior_cov,
    calc_posterior_mean,
)
from raytracer import SphericalMesh
from tti.traveltimes.traveltimes import calculate_path_direction_vector

from main import (
    calculate_noise_sigma,
    construct_forward_map,
    construct_path_distance_squared_matrix,
    determine_weights,
)


# =============================================================================
# SETTINGS
# =============================================================================

DATA_FILE = Path("data/brett2024_ic_traveltimes.parquet")

RADIAL_RESOLUTION = 10
LATERAL_RESOLUTION = 5
INNER_CORE_RADIUS_KM = 1221.5

# Correct Earth-model prior:
# C_m = 0.01 I, so prior SD = 0.1.
PRIOR_VARIANCE = 0.01

# Exact 72-point hyperparameter grid from the completed run.
TAU_GRID = np.array(
    [
        0.00350,
        0.00375,
        0.00400,
        0.00425,
        0.00450,
        0.00475,
        0.00500,
        0.00525,
        0.00550,
    ],
    dtype=float,
)

ELL_GRID = np.array(
    [
        250.0,
        300.0,
        350.0,
        400.0,
        450.0,
        500.0,
        550.0,
        600.0,
    ],
    dtype=float,
)

# The latest 72-point run was saved in this folder even though the folder name
# still says "coarse40".
HYPERPARAMETER_DIR = Path(
    "outputs/tau_ell_posterior_r10_l5_prior_0p01_coarse40"
)

POSTERIOR_GRID_FILE = (
    HYPERPARAMETER_DIR
    / "tau_ell_posterior.csv"
)

# New output directory: do not mix these results with earlier runs.
OUTPUT_DIR = Path(
    "outputs/earth_posterior_marginalized_r10_l5_prior_0p01_full72"
)

# -------------------------------------------------------------------------
# SPEED / ACCURACY CONTROL
# -------------------------------------------------------------------------
#
# The exact discrete-grid mixture uses all 72 grid points.
#
# A very large fraction of the posterior mass is concentrated in a much
# smaller number of points, so for speed we can retain the highest-weight
# points until this cumulative posterior mass is reached.
#
# 0.9999 = retain at least 99.99% of the hyperparameter posterior mass.
# 1.0    = exact 72-point discrete-grid marginalization.
#
RETAIN_POSTERIOR_MASS = 1

# If True, reuse MAP and IID Earth posterior arrays already produced by the
# completed hyperparameter run instead of recalculating them.
REUSE_EXISTING_MAP_AND_IID = True


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# HELPERS
# =============================================================================


def construct_covariance_inplace(
    path_distance_squared: np.ndarray,
    sigma_squared: np.ndarray,
    tau: float,
    ell: float,
) -> np.ndarray:
    """
    Construct

        C_d = tau^2 K_ell + diag(sigma^2),

    where

        (K_ell)_ij = exp[-d_ij^2 / (2 ell^2)].

    This version avoids creating a separate kernel array and covariance array,
    reducing peak memory compared with

        kernel = exp(...)
        covariance = tau**2 * kernel.
    """
    if tau < 0.0:
        raise ValueError("tau must be non-negative.")
    if ell <= 0.0:
        raise ValueError("ell must be positive.")

    covariance = np.exp(
        -path_distance_squared / (2.0 * ell**2)
    )
    covariance *= tau**2

    diagonal = np.diag_indices_from(covariance)
    covariance[diagonal] += sigma_squared

    # path_distance_squared is symmetric, so this should already be symmetric.
    # Avoid making another full n x n copy merely to symmetrize it.
    return covariance


def conditional_earth_posterior(
    *,
    data: np.ndarray,
    inferred: list[GaussianComponent],
    identity_data: np.ndarray,
    zero_data: np.ndarray,
    covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return conditional Earth posterior mean, covariance and SD.

    The n x n identity matrix and zero vector are supplied once and reused
    across every hyperparameter point.  The old script recreated them at every
    point, which is expensive for n = 7668.
    """
    nuisance = [
        GaussianComponent(
            identity_data,
            zero_data,
            covariance,
        )
    ]

    mean = calc_posterior_mean(
        data,
        inferred,
        nuisance,
    )

    cov = calc_posterior_cov(
        inferred,
        nuisance,
    )

    cov = 0.5 * (cov + cov.T)

    variance = np.diag(cov)
    if np.min(variance) < -1e-10:
        raise RuntimeError(
            "Conditional posterior covariance has materially negative "
            "diagonal entries."
        )

    sd = np.sqrt(
        np.maximum(variance, 0.0)
    )

    return mean, cov, sd


def load_and_validate_hyperparameter_posterior() -> pd.DataFrame:
    """
    Load the already-computed 72-point hyperparameter posterior.

    This avoids rerunning the extremely expensive log-evidence grid before
    Earth-posterior marginalization.
    """
    if not POSTERIOR_GRID_FILE.exists():
        raise FileNotFoundError(
            f"Could not find:\n{POSTERIOR_GRID_FILE}\n\n"
            "Run the completed 72-point tau/ell analysis first."
        )

    df = pd.read_csv(
        POSTERIOR_GRID_FILE
    )

    required = {
        "tau",
        "ell_km",
        "posterior_probability",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns in {POSTERIOR_GRID_FILE}: "
            f"{sorted(missing)}"
        )

    df = df.copy()
    df["tau"] = df["tau"].astype(float).round(8)
    df["ell_km"] = df["ell_km"].astype(float).round(6)
    df["posterior_probability"] = (
        df["posterior_probability"]
        .astype(float)
    )

    expected = pd.MultiIndex.from_product(
        [
            np.round(TAU_GRID, 8),
            np.round(ELL_GRID, 6),
        ],
        names=["tau", "ell_km"],
    )

    df = (
        df[
            df["tau"].isin(np.round(TAU_GRID, 8))
            & df["ell_km"].isin(np.round(ELL_GRID, 6))
        ]
        .drop_duplicates(
            subset=["tau", "ell_km"],
            keep="last",
        )
        .sort_values(["tau", "ell_km"])
        .reset_index(drop=True)
    )

    actual = pd.MultiIndex.from_frame(
        df[["tau", "ell_km"]]
    )

    missing_points = expected.difference(actual)

    if len(missing_points) > 0:
        raise RuntimeError(
            "The hyperparameter posterior file does not contain the complete "
            f"9 x 8 = 72 point grid. Missing {len(missing_points)} points."
        )

    if len(df) != 72:
        raise RuntimeError(
            f"Expected 72 unique grid points, found {len(df)}."
        )

    weight_sum = float(
        df["posterior_probability"].sum()
    )

    if not np.isclose(
        weight_sum,
        1.0,
        rtol=1e-8,
        atol=1e-10,
    ):
        raise RuntimeError(
            "Posterior probabilities should sum to 1, but sum to "
            f"{weight_sum:.16f}."
        )

    # Check the completed run's MAP point.
    map_row = df.loc[
        df["posterior_probability"].idxmax()
    ]

    map_tau = float(map_row["tau"])
    map_ell = float(map_row["ell_km"])

    if not np.isclose(map_tau, 0.00525):
        raise RuntimeError(
            f"Unexpected MAP tau: {map_tau}. Expected 0.00525."
        )

    if not np.isclose(map_ell, 450.0):
        raise RuntimeError(
            f"Unexpected MAP ell: {map_ell}. Expected 450 km."
        )

    logger.info(
        "Verified complete 72-point posterior grid; "
        "MAP tau=%.5f ell=%.0f km.",
        map_tau,
        map_ell,
    )

    return df


def select_high_posterior_mass_points(
    posterior_df: pd.DataFrame,
    retain_mass: float,
) -> pd.DataFrame:
    """
    Select highest-probability grid points until cumulative posterior mass
    reaches retain_mass, then renormalize their weights.

    If retain_mass == 1.0, all 72 points are retained exactly.
    """
    if not (0.0 < retain_mass <= 1.0):
        raise ValueError(
            "RETAIN_POSTERIOR_MASS must be in (0, 1]."
        )

    ranked = (
        posterior_df
        .sort_values(
            "posterior_probability",
            ascending=False,
        )
        .reset_index(drop=True)
        .copy()
    )

    if np.isclose(retain_mass, 1.0):
        selected = ranked.copy()
    else:
        cumulative = (
            ranked["posterior_probability"]
            .cumsum()
            .to_numpy()
        )

        cutoff_index = int(
            np.searchsorted(
                cumulative,
                retain_mass,
                side="left",
            )
        )

        selected = ranked.iloc[
            : cutoff_index + 1
        ].copy()

    retained_mass = float(
        selected["posterior_probability"].sum()
    )

    omitted_mass = 1.0 - retained_mass

    selected[
        "original_posterior_probability"
    ] = selected[
        "posterior_probability"
    ]

    selected[
        "mixture_weight"
    ] = (
        selected["posterior_probability"]
        / retained_mass
    )

    selected[
        "cumulative_original_mass"
    ] = selected[
        "original_posterior_probability"
    ].cumsum()

    logger.info(
        "Selected %d/%d grid points; retained posterior mass = %.10f; "
        "omitted mass = %.3e.",
        len(selected),
        len(posterior_df),
        retained_mass,
        omitted_mass,
    )

    return selected


def load_existing_array(filename: str) -> np.ndarray | None:
    path = HYPERPARAMETER_DIR / filename
    if path.exists():
        logger.info(
            "Reusing existing output: %s",
            path,
        )
        return np.load(path)

    logger.warning(
        "Existing output not found: %s",
        path,
    )
    return None


# =============================================================================
# EARTH-POSTERIOR MARGINALIZATION
# =============================================================================


def marginalize_earth_posterior(
    *,
    data: np.ndarray,
    inferred: list[GaussianComponent],
    n_parameters: int,
    path_distance_squared: np.ndarray,
    sigma_squared: np.ndarray,
    selected_points: pd.DataFrame,
    identity_data: np.ndarray,
    zero_data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute first two moments of the selected discrete Gaussian mixture:

        mu = sum_k w_k mu_k

        C  = sum_k w_k (C_k + mu_k mu_k^T) - mu mu^T.

    selected_points contains renormalized mixture weights.
    """
    marginal_mean = np.zeros(
        n_parameters,
        dtype=float,
    )

    marginal_second_moment = np.zeros(
        (n_parameters, n_parameters),
        dtype=float,
    )

    total_weight = 0.0

    # Process points grouped by ell.  This makes the run easier to monitor and
    # avoids unnecessary dataframe work inside the expensive calculations.
    groups = list(
        selected_points.groupby(
            "ell_km",
            sort=True,
        )
    )

    point_number = 0
    total_points = len(selected_points)

    for ell, group in groups:
        for row in group.itertuples():
            point_number += 1

            tau = float(row.tau)
            weight = float(row.mixture_weight)

            logger.info(
                "Earth mixture %d/%d: tau=%.5f ell=%.0f km "
                "weight=%.8f original_weight=%.8f",
                point_number,
                total_points,
                tau,
                ell,
                weight,
                float(row.original_posterior_probability),
            )

            covariance = construct_covariance_inplace(
                path_distance_squared=path_distance_squared,
                sigma_squared=sigma_squared,
                tau=tau,
                ell=float(ell),
            )

            mean_k, cov_k, _ = conditional_earth_posterior(
                data=data,
                inferred=inferred,
                identity_data=identity_data,
                zero_data=zero_data,
                covariance=covariance,
            )

            marginal_mean += (
                weight * mean_k
            )

            marginal_second_moment += (
                weight
                * (
                    cov_k
                    + np.outer(mean_k, mean_k)
                )
            )

            total_weight += weight

            # Explicitly release large per-point arrays before the next point.
            del covariance
            del mean_k
            del cov_k

    if not np.isclose(
        total_weight,
        1.0,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Renormalized mixture weights sum to "
            f"{total_weight:.16f}, not 1."
        )

    marginal_cov = (
        marginal_second_moment
        - np.outer(
            marginal_mean,
            marginal_mean,
        )
    )

    marginal_cov = (
        0.5
        * (
            marginal_cov
            + marginal_cov.T
        )
    )

    marginal_variance = np.diag(
        marginal_cov
    )

    if np.min(marginal_variance) < -1e-10:
        raise RuntimeError(
            "Marginal covariance has materially negative diagonal entries."
        )

    marginal_sd = np.sqrt(
        np.maximum(
            marginal_variance,
            0.0,
        )
    )

    return (
        marginal_mean,
        marginal_cov,
        marginal_sd,
    )


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # 1. Load and verify existing hyperparameter posterior
    # -------------------------------------------------------------------------
    posterior_df = (
        load_and_validate_hyperparameter_posterior()
    )

    selected_points = (
        select_high_posterior_mass_points(
            posterior_df,
            RETAIN_POSTERIOR_MASS,
        )
    )

    selected_points.to_csv(
        OUTPUT_DIR
        / "selected_hyperparameter_points.csv",
        index=False,
    )

    retained_mass = float(
        selected_points[
            "original_posterior_probability"
        ].sum()
    )

    omitted_mass = (
        1.0
        - retained_mass
    )

    # -------------------------------------------------------------------------
    # 2. Construct the corrected 10 x 5 Earth inverse problem once
    # -------------------------------------------------------------------------
    logger.info(
        "Reading data from %s",
        DATA_FILE,
    )

    df = pd.read_parquet(
        DATA_FILE
    )

    data = (
        df["delta_t"]
        / df["inner_core_travel_time"]
    ).astype(float).to_numpy()

    ic_in = np.stack(
        df["in_location"].to_numpy()
    )

    ic_out = np.stack(
        df["out_location"].to_numpy()
    )

    n_data = len(data)

    sigma = calculate_noise_sigma(
        df["reference_phase"],
        df["inner_core_travel_time"],
    )

    sigma_squared = (
        np.asarray(
            sigma,
            dtype=float,
        )
        ** 2
    )

    path_directions = (
        calculate_path_direction_vector(
            ic_in,
            ic_out,
        )
    )

    mesh = SphericalMesh(
        INNER_CORE_RADIUS_KM,
        RADIAL_RESOLUTION,
        LATERAL_RESOLUTION,
    )

    path_weights = determine_weights(
        mesh,
        ic_in,
        path_directions,
    )

    forward_matrix = construct_forward_map(
        path_directions,
        path_weights,
    )

    if forward_matrix.shape != (
        7668,
        1350,
    ):
        raise RuntimeError(
            "Unexpected forward matrix shape: "
            f"{forward_matrix.shape}. "
            "Expected (7668, 1350) for the corrected 10 x 5 mesh."
        )

    n_parameters = (
        forward_matrix.shape[1]
    )

    inferred = [
        GaussianComponent(
            forward_matrix,
            np.zeros(
                n_parameters,
                dtype=float,
            ),
            PRIOR_VARIANCE
            * np.eye(
                n_parameters,
                dtype=float,
            ),
        )
    ]

    # Construct the path-distance matrix only once.
    logger.info(
        "Constructing path-distance-squared matrix..."
    )

    path_distance_squared = (
        construct_path_distance_squared_matrix(
            ic_in,
            ic_out,
        )
    )

    # Reuse these large nuisance-model arrays across every mixture point.
    logger.info(
        "Constructing reusable data-space identity matrix..."
    )

    identity_data = np.eye(
        n_data,
        dtype=float,
    )

    zero_data = np.zeros(
        n_data,
        dtype=float,
    )

    logger.info(
        "Starting Earth-posterior marginalization with %d selected points.",
        len(selected_points),
    )

    # -------------------------------------------------------------------------
    # 3. Marginalize Earth posterior over selected tau/ell points
    # -------------------------------------------------------------------------
    (
        marginal_mean,
        marginal_cov,
        marginal_sd,
    ) = marginalize_earth_posterior(
        data=data,
        inferred=inferred,
        n_parameters=n_parameters,
        path_distance_squared=path_distance_squared,
        sigma_squared=sigma_squared,
        selected_points=selected_points,
        identity_data=identity_data,
        zero_data=zero_data,
    )

    np.save(
        OUTPUT_DIR
        / "earth_posterior_mean_marginalized.npy",
        marginal_mean,
    )

    np.save(
        OUTPUT_DIR
        / "earth_posterior_covariance_marginalized.npy",
        marginal_cov,
    )

    np.save(
        OUTPUT_DIR
        / "earth_posterior_sd_marginalized.npy",
        marginal_sd,
    )

    # -------------------------------------------------------------------------
    # 4. Reuse already-computed MAP and IID arrays where possible
    # -------------------------------------------------------------------------
    map_mean = None
    map_sd = None
    iid_mean = None
    iid_sd = None

    if REUSE_EXISTING_MAP_AND_IID:
        map_mean = load_existing_array(
            "earth_posterior_mean_MAP.npy"
        )
        map_sd = load_existing_array(
            "earth_posterior_sd_MAP.npy"
        )
        iid_mean = load_existing_array(
            "earth_posterior_mean_block_iid.npy"
        )
        iid_sd = load_existing_array(
            "earth_posterior_sd_block_iid.npy"
        )

    # -------------------------------------------------------------------------
    # 5. Save comparison arrays if matching old arrays are available
    # -------------------------------------------------------------------------
    if (
        map_mean is not None
        and map_sd is not None
        and map_mean.shape == marginal_mean.shape
        and map_sd.shape == marginal_sd.shape
    ):
        np.save(
            OUTPUT_DIR
            / "earth_posterior_mean_difference_marginalized_minus_MAP.npy",
            marginal_mean - map_mean,
        )

        np.save(
            OUTPUT_DIR
            / "earth_posterior_sd_difference_marginalized_minus_MAP.npy",
            marginal_sd - map_sd,
        )

    if (
        iid_mean is not None
        and iid_sd is not None
        and iid_mean.shape == marginal_mean.shape
        and iid_sd.shape == marginal_sd.shape
    ):
        np.save(
            OUTPUT_DIR
            / "earth_posterior_mean_difference_marginalized_minus_block_iid.npy",
            marginal_mean - iid_mean,
        )

        np.save(
            OUTPUT_DIR
            / "earth_posterior_sd_difference_marginalized_minus_block_iid.npy",
            marginal_sd - iid_sd,
        )

    # -------------------------------------------------------------------------
    # 6. Save run summary
    # -------------------------------------------------------------------------
    summary = pd.DataFrame(
        [{
            "radial_resolution": RADIAL_RESOLUTION,
            "lateral_resolution": LATERAL_RESOLUTION,
            "prior_variance": PRIOR_VARIANCE,
            "n_parameters": n_parameters,
            "total_hyperparameter_grid_points": len(posterior_df),
            "selected_grid_points": len(selected_points),
            "requested_retained_posterior_mass": RETAIN_POSTERIOR_MASS,
            "actual_retained_posterior_mass": retained_mass,
            "omitted_posterior_mass": omitted_mass,
            "marginalized_mean_min": float(np.min(marginal_mean)),
            "marginalized_mean_max": float(np.max(marginal_mean)),
            "marginalized_sd_min": float(np.min(marginal_sd)),
            "marginalized_sd_max": float(np.max(marginal_sd)),
        }]
    )

    summary.to_csv(
        OUTPUT_DIR
        / "marginalization_summary.csv",
        index=False,
    )

    print()
    print("=" * 72)
    print("FINAL EARTH-POSTERIOR MARGINALIZATION SUMMARY")
    print("=" * 72)
    print(f"Mesh                         = {RADIAL_RESOLUTION} x {LATERAL_RESOLUTION}")
    print(f"Earth prior variance         = {PRIOR_VARIANCE}")
    print(f"Earth prior SD               = {np.sqrt(PRIOR_VARIANCE)}")
    print(f"Earth parameters             = {n_parameters}")
    print(f"Full tau/ell grid points     = {len(posterior_df)}")
    print(f"Selected mixture points      = {len(selected_points)}")
    print(f"Requested retained mass      = {RETAIN_POSTERIOR_MASS:.8f}")
    print(f"Actual retained mass         = {retained_mass:.10f}")
    print(f"Omitted posterior mass       = {omitted_mass:.3e}")
    print(
        "Marginalized mean range     = "
        f"{np.min(marginal_mean):.8f} to {np.max(marginal_mean):.8f}"
    )
    print(
        "Marginalized SD range       = "
        f"{np.min(marginal_sd):.8f} to {np.max(marginal_sd):.8f}"
    )
    print()
    if RETAIN_POSTERIOR_MASS < 1.0:
        print(
            "NOTE: This is a posterior-mass-truncated approximation to the "
            "72-point discrete-grid mixture."
        )
        print(
            "For an exact 72-point discrete-grid marginalization, set "
            "RETAIN_POSTERIOR_MASS = 1.0."
        )
    else:
        print(
            "This is the exact 72-point discrete-grid marginalization."
        )
    print()
    print("Saved outputs in:")
    print(OUTPUT_DIR)

    logger.info(
        "Finished."
    )


if __name__ == "__main__":
    main()
