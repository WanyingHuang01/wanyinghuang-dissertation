import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from linear_gaussian import (
    GaussianComponent,
    calc_log_evidence,
    calc_posterior_cov,
    calc_posterior_mean,
)

from raytracer import SphericalMesh

from tti.traveltimes.traveltimes import (
    calculate_path_direction_vector,
)

from main import (
    calculate_noise_sigma,
    construct_forward_map,
    construct_path_distance_squared_matrix,
    determine_weights,
)


# ============================================================
# SETTINGS
# ============================================================

DATA_FILE = (
    "data/brett2024_ic_traveltimes.parquet"
)

RADIAL_RESOLUTION = 10
LATERAL_RESOLUTION = 5

OUTPUT_DIR = Path(
    "outputs/tau_ell_posterior_r10_l5_prior_0p01_coarse40"
)

# ------------------------------------------------------------
# Earth-model settings
# ------------------------------------------------------------



PRIOR_VARIANCE = 0.01


# ============================================================
# PRIORS FOR HYPERPARAMETERS
# ============================================================

# tau ~ HalfNormal(0.003)
TAU_PRIOR_SCALE = 0.003


# ell ~ Uniform(50, 800) km
ELL_MIN = 50.0
ELL_MAX = 800.0


# ============================================================
# COARSE EXPLORATORY GRID
# ============================================================
# 8 tau values x 5 ell values = 40 grid points.
# Use this run to locate the posterior mass under the corrected
# Earth-model prior and the 10 x 5 mesh. If needed, refine locally
# before the final Earth-model hyperparameter marginalization.

TAU_GRID = np.arange(0.0035, 0.00551, 0.00025)
ELL_GRID = np.arange(250.0, 601.0, 50.0)
# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s "
        "%(levelname)s "
        "%(name)s: "
        "%(message)s"
    ),
)

logger = logging.getLogger(
    __name__
)


# ============================================================
# PRIORS
# ============================================================


def log_prior_tau(
    tau: float,
    scale: float,
) -> float:
    """Half-normal log prior for tau."""

    if tau < 0.0:
        return -np.inf

    if scale <= 0.0:
        raise ValueError(
            "Half-normal scale must be positive."
        )

    return float(
        0.5 * np.log(
            2.0 / np.pi
        )
        - np.log(
            scale
        )
        - 0.5 * (
            tau / scale
        ) ** 2
    )


def log_prior_ell(
    ell: float,
    lower: float,
    upper: float,
) -> float:
    """Uniform log prior for ell."""

    if upper <= lower:
        raise ValueError(
            "Upper ell bound must exceed lower bound."
        )

    if (
        ell < lower
        or ell > upper
    ):
        return -np.inf

    return float(
        -np.log(
            upper - lower
        )
    )


# ============================================================
# COVARIANCE
# ============================================================


def construct_covariance(
    path_distance_squared: np.ndarray,
    sigma: np.ndarray,
    tau: float,
    ell: float,
) -> np.ndarray:
    """Construct C_epsilon + C_delta."""

    if tau < 0.0:
        raise ValueError(
            "tau must be non-negative."
        )

    if ell <= 0.0:
        raise ValueError(
            "ell must be positive."
        )

    # RBF kernel
    kernel = np.exp(
        -path_distance_squared
        / (
            2.0 * ell**2
        )
    )

    # Correlated component
    covariance = (
        tau**2
        * kernel
    )

    # Add independent observational variance
    diagonal_indices = (
        np.diag_indices_from(
            covariance
        )
    )

    covariance[
        diagonal_indices
    ] += sigma**2

    # Ensure numerical symmetry
    covariance = (
        0.5
        * (
            covariance
            + covariance.T
        )
    )

    return covariance


# ============================================================
# NORMALIZE LOG POSTERIOR FOR A UNIFORM GRID
# ============================================================


def logsumexp_numpy(
    values: np.ndarray,
) -> float:
    """Stable NumPy-only log-sum-exp."""

    values = np.asarray(
        values,
        dtype=float,
    )

    maximum = np.max(
        values
    )

    if not np.isfinite(
        maximum
    ):
        return float(
            maximum
        )

    return float(
        maximum
        + np.log(
            np.sum(
                np.exp(
                    values
                    - maximum
                )
            )
        )
    )


def normalize_log_grid_uniform(
    log_grid: np.ndarray,
    tau_grid: np.ndarray,
    ell_grid: np.ndarray,
):
    """
    Normalize a posterior-density grid evaluated on a uniform grid.

    Because Delta tau and Delta ell are constant, the common cell area
    cancels when the discrete posterior probabilities are normalized.

    The marginalized evidence still includes the constant cell area
    Delta tau * Delta ell.
    """

    if len(tau_grid) < 2 or len(ell_grid) < 2:
        raise ValueError(
            "tau_grid and ell_grid must each contain at least two points."
        )

    delta_tau_values = np.diff(
        tau_grid
    )

    delta_ell_values = np.diff(
        ell_grid
    )

    if not np.allclose(
        delta_tau_values,
        delta_tau_values[0],
    ):
        raise ValueError(
            "TAU_GRID is not uniformly spaced."
        )

    if not np.allclose(
        delta_ell_values,
        delta_ell_values[0],
    ):
        raise ValueError(
            "ELL_GRID is not uniformly spaced."
        )

    delta_tau = float(
        delta_tau_values[0]
    )

    delta_ell = float(
        delta_ell_values[0]
    )

    finite = np.isfinite(
        log_grid
    )

    if not np.any(
        finite
    ):
        raise ValueError(
            "No finite posterior values found."
        )

    log_sum = logsumexp_numpy(
        log_grid[
            finite
        ]
    )

    posterior_probability = np.zeros_like(
        log_grid,
        dtype=float,
    )

    posterior_probability[
        finite
    ] = np.exp(
        log_grid[
            finite
        ]
        - log_sum
    )

    # Numerical approximation to:
    #
    # integral integral
    # p(data | tau, ell) p(tau) p(ell) d tau d ell
    #
    # using the constant uniform-grid cell area.
    log_integral = float(
        log_sum
        + np.log(
            delta_tau
        )
        + np.log(
            delta_ell
        )
    )

    return (
        posterior_probability,
        log_integral,
        delta_tau,
        delta_ell,
    )


# ============================================================
# SAVE EACH GRID POINT
# ============================================================


def append_result(
    output_file: Path,
    tau: float,
    ell: float,
    log_evidence: float,
    log_prior: float,
    log_posterior: float,
) -> None:
    """Save one completed grid point immediately."""

    row = pd.DataFrame(
        [
            {
                "tau": tau,
                "ell_km": ell,
                "log_evidence": (
                    log_evidence
                ),
                "log_prior": (
                    log_prior
                ),
                "log_posterior_unnormalized": (
                    log_posterior
                ),
            }
        ]
    )

    if output_file.exists():

        row.to_csv(
            output_file,
            mode="a",
            header=False,
            index=False,
        )

    else:

        row.to_csv(
            output_file,
            mode="w",
            header=True,
            index=False,
        )


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    """Run Bayesian tau/ell grid inference."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "Analysis settings: radial=%d lateral=%d prior_variance=%.5f "
        "tau_points=%d ell_points=%d total_grid_points=%d",
        RADIAL_RESOLUTION,
        LATERAL_RESOLUTION,
        PRIOR_VARIANCE,
        len(TAU_GRID),
        len(ELL_GRID),
        len(TAU_GRID) * len(ELL_GRID),
    )

    if not np.isclose(PRIOR_VARIANCE, 0.01):
        raise ValueError(
            "This corrected-prior run is expected to use PRIOR_VARIANCE = 0.01."
        )

    if RADIAL_RESOLUTION != 10 or LATERAL_RESOLUTION != 5:
        raise ValueError(
            "This script is configured for the evidence-selected tested mesh "
            "(radial=10, lateral=5)."
        )

    raw_results_file = (
        OUTPUT_DIR
        / "tau_ell_raw_results.csv"
    )


    # ========================================================
    # LOAD DATA
    # ========================================================

    logger.info(
        "Reading data from %s",
        DATA_FILE,
    )

    df = pd.read_parquet(
        DATA_FILE
    )

    data = (
        df["delta_t"]
        / df[
            "inner_core_travel_time"
        ]
    ).astype(float).to_numpy()

    ic_in = np.stack(
        df[
            "in_location"
        ].to_numpy()
    )

    ic_out = np.stack(
        df[
            "out_location"
        ].to_numpy()
    )

    n_data = len(
        data
    )

    logger.info(
        "Number of observations: %d",
        n_data,
    )


    # ========================================================
    # INDEPENDENT OBSERVATIONAL NOISE
    # ========================================================

    sigma = calculate_noise_sigma(
        df[
            "reference_phase"
        ],
        df[
            "inner_core_travel_time"
        ],
    )

    logger.info(
        "Noise sigma range: %.6e to %.6e",
        sigma.min(),
        sigma.max(),
    )


    # ========================================================
    # FORWARD MODEL
    # ========================================================

    path_directions = (
        calculate_path_direction_vector(
            ic_in,
            ic_out,
        )
    )

    mesh = SphericalMesh(
        1221.5,
        RADIAL_RESOLUTION,
        LATERAL_RESOLUTION,
    )

    weights = determine_weights(
        mesh,
        ic_in,
        path_directions,
    )

    forward_matrix = (
        construct_forward_map(
            path_directions,
            weights,
        )
    )

    (
        _,
        n_parameters,
    ) = forward_matrix.shape

    logger.info(
        "Forward matrix shape: %s",
        forward_matrix.shape,
    )


    # ========================================================
    # EARTH MODEL PRIOR
    # ========================================================

    inferred = [
        GaussianComponent(
            forward_matrix,
            np.zeros(
                n_parameters
            ),
            PRIOR_VARIANCE
            * np.eye(
                n_parameters
            ),
        )
    ]


    # ========================================================
    # PATH DISTANCE MATRIX
    # ========================================================

    logger.info(
        "Constructing squared entry+exit path-distance matrix..."
    )

    path_distance_squared = (
        construct_path_distance_squared_matrix(
            ic_in,
            ic_out,
        )
    )

    logger.info(
        "Path-distance matrix shape: %s",
        path_distance_squared.shape,
    )


    # ========================================================
    # CHECK FOR ALREADY COMPLETED GRID POINTS
    # ========================================================

    completed = set()

    if raw_results_file.exists():

        existing = pd.read_csv(
            raw_results_file
        )

        for row in existing.itertuples():

            completed.add(
                (
                    float(
                        row.tau
                    ),
                    float(
                        row.ell_km
                    ),
                )
            )

        logger.info(
            "Found %d completed grid points.",
            len(
                completed
            ),
        )


    # ========================================================
    # GRID LOOP
    # ========================================================

    total_points = (
        len(
            TAU_GRID
        )
        * len(
            ELL_GRID
        )
    )

    point_number = 0


    for tau in TAU_GRID:

        log_p_tau = (
            log_prior_tau(
                tau,
                TAU_PRIOR_SCALE,
            )
        )

        for ell in ELL_GRID:

            point_number += 1

            pair = (
                float(
                    tau
                ),
                float(
                    ell
                ),
            )


            # ------------------------------------------------
            # Skip completed points
            # ------------------------------------------------

            if pair in completed:

                logger.info(
                    "Skipping completed %d/%d: "
                    "tau=%.6f ell=%.1f",
                    point_number,
                    total_points,
                    tau,
                    ell,
                )

                continue


            logger.info(
                "Grid %d/%d: "
                "tau=%.6f ell=%.1f km",
                point_number,
                total_points,
                tau,
                ell,
            )


            # ------------------------------------------------
            # Prior
            # ------------------------------------------------

            log_p_ell = (
                log_prior_ell(
                    ell,
                    ELL_MIN,
                    ELL_MAX,
                )
            )


            # ------------------------------------------------
            # Covariance
            # ------------------------------------------------

            covariance = (
                construct_covariance(
                    path_distance_squared=(
                        path_distance_squared
                    ),
                    sigma=sigma,
                    tau=tau,
                    ell=ell,
                )
            )


            nuisance = [
                GaussianComponent(
                    np.eye(
                        n_data
                    ),
                    np.zeros(
                        n_data
                    ),
                    covariance,
                )
            ]


            # ------------------------------------------------
            # Exact log evidence
            # ------------------------------------------------

            try:

                log_evidence = (
                    calc_log_evidence(
                        data,
                        inferred,
                        nuisance,
                    )
                )

            except ValueError as error:

                logger.error(
                    "FAILED tau=%.6f ell=%.1f: %s",
                    tau,
                    ell,
                    error,
                )

                continue


            # ------------------------------------------------
            # Posterior
            # ------------------------------------------------

            log_prior = (
                log_p_tau
                + log_p_ell
            )

            log_posterior = (
                log_evidence
                + log_prior
            )


            logger.info(
                "log evidence = %.6f",
                log_evidence,
            )

            logger.info(
                "log prior = %.6f",
                log_prior,
            )

            logger.info(
                "unnormalized log posterior = %.6f",
                log_posterior,
            )


            # ------------------------------------------------
            # Save immediately
            # ------------------------------------------------

            append_result(
                raw_results_file,
                tau=tau,
                ell=ell,
                log_evidence=(
                    log_evidence
                ),
                log_prior=(
                    log_prior
                ),
                log_posterior=(
                    log_posterior
                ),
            )


    # ========================================================
    # READ COMPLETE RESULTS
    # ========================================================

    results = pd.read_csv(
        raw_results_file
    )

    results = (
        results
        .drop_duplicates(
            subset=[
                "tau",
                "ell_km",
            ],
            keep="last",
        )
        .sort_values(
            [
                "tau",
                "ell_km",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # CONSTRUCT ARRAYS
    # ========================================================

    grid_shape = (
        len(
            TAU_GRID
        ),
        len(
            ELL_GRID
        ),
    )

    log_evidence_grid = np.full(
        grid_shape,
        np.nan,
    )

    log_prior_grid = np.full(
        grid_shape,
        np.nan,
    )

    log_posterior_grid = np.full(
        grid_shape,
        np.nan,
    )


    for row in results.itertuples():

        tau_matches = np.where(
            np.isclose(
                TAU_GRID,
                row.tau,
            )
        )[0]

        ell_matches = np.where(
            np.isclose(
                ELL_GRID,
                row.ell_km,
            )
        )[0]


        if (
            len(
                tau_matches
            ) == 0
            or len(
                ell_matches
            ) == 0
        ):
            continue


        i = int(
            tau_matches[0]
        )

        j = int(
            ell_matches[0]
        )


        log_evidence_grid[
            i,
            j,
        ] = (
            row.log_evidence
        )

        log_prior_grid[
            i,
            j,
        ] = (
            row.log_prior
        )

        log_posterior_grid[
            i,
            j,
        ] = (
            row.log_posterior_unnormalized
        )


    # ========================================================
    # NORMALIZE POSTERIOR
    # ========================================================

    (
        posterior_weights,
        correlated_log_evidence_marginalized,
        delta_tau,
        delta_ell,
    ) = normalize_log_grid_uniform(
        log_posterior_grid,
        TAU_GRID,
        ELL_GRID,
    )

    print(
        "Uniform grid spacing:"
    )
    print(
        "Delta tau =",
        delta_tau,
    )
    print(
        "Delta ell =",
        delta_ell,
        "km",
    )

    print()
    print(
        "=============================================================="
    )
    print(
        "MARGINALIZED CORRELATED-MODEL EVIDENCE"
    )
    print(
        "=============================================================="
    )
    print(
        "log p(data | M_corr) =",
        correlated_log_evidence_marginalized,
    )


    # ========================================================
    # MARGINAL POSTERIORS
    # ========================================================

    tau_posterior = np.nansum(
        posterior_weights,
        axis=1,
    )

    ell_posterior = np.nansum(
        posterior_weights,
        axis=0,
    )


    # ========================================================
    # MAP VALUE
    # ========================================================

    map_index = np.unravel_index(
        np.nanargmax(
            log_posterior_grid
        ),
        log_posterior_grid.shape,
    )

    map_tau = (
        TAU_GRID[
            map_index[0]
        ]
    )

    map_ell = (
        ELL_GRID[
            map_index[1]
        ]
    )


    # ========================================================
    # SAVE NORMALIZED POSTERIOR
    # ========================================================

    normalized_rows = []


    for i, tau in enumerate(
        TAU_GRID
    ):

        for j, ell in enumerate(
            ELL_GRID
        ):

            if not np.isfinite(
                log_posterior_grid[
                    i,
                    j,
                ]
            ):
                continue


            normalized_rows.append(
                {
                    "tau": tau,
                    "ell_km": ell,
                    "log_evidence": (
                        log_evidence_grid[
                            i,
                            j,
                        ]
                    ),
                    "log_prior": (
                        log_prior_grid[
                            i,
                            j,
                        ]
                    ),
                    "log_posterior": (
                        log_posterior_grid[
                            i,
                            j,
                        ]
                    ),
                    "posterior_probability": (
                        posterior_weights[
                            i,
                            j,
                        ]
                    ),
                }
            )


    normalized_df = pd.DataFrame(
        normalized_rows
    )


    normalized_df.to_csv(
        OUTPUT_DIR
        / "tau_ell_posterior.csv",
        index=False,
    )

    evidence_summary = pd.DataFrame(
        [
            {
                "radial_resolution": RADIAL_RESOLUTION,
                "lateral_resolution": LATERAL_RESOLUTION,
                "prior_variance": PRIOR_VARIANCE,
                "delta_tau": delta_tau,
                "delta_ell_km": delta_ell,
                "marginalized_correlated_log_evidence": (
                    correlated_log_evidence_marginalized
                ),
                "map_tau": map_tau,
                "map_ell_km": map_ell,
            }
        ]
    )

    evidence_summary.to_csv(
        OUTPUT_DIR
        / "correlated_model_evidence_summary.csv",
        index=False,
    )

    # ========================================================
    # POSTERIOR HEATMAP
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(
            8,
            6,
        )
    )


    image = ax.imshow(
        posterior_weights,
        origin="lower",
        aspect="auto",
    )


    ax.set_xticks(
        np.arange(
            len(
                ELL_GRID
            )
        )
    )

    ax.set_xticklabels(
        [
            f"{ell:.0f}"
            for ell in ELL_GRID
        ]
    )


    ax.set_yticks(
        np.arange(
            len(
                TAU_GRID
            )
        )
    )

    ax.set_yticklabels(
        [
            f"{tau:.4f}"
            for tau in TAU_GRID
        ]
    )


    ax.set_xlabel(
        "Correlation length ell (km)"
    )

    ax.set_ylabel(
        "Correlated-error SD tau"
    )

    ax.set_title(
        "Posterior p(tau, ell | data)"
    )


    colorbar = fig.colorbar(
        image,
        ax=ax,
    )

    colorbar.set_label(
        "Posterior probability"
    )


    fig.tight_layout()


    fig.savefig(
        OUTPUT_DIR
        / "tau_ell_posterior_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


    # ========================================================
    # MARGINAL TAU POSTERIOR
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(
            7,
            5,
        )
    )


    ax.plot(
        TAU_GRID,
        tau_posterior,
        marker="o",
    )


    ax.set_xlabel(
        "tau"
    )

    ax.set_ylabel(
        "Posterior probability"
    )

    ax.set_title(
        "Marginal posterior for tau"
    )


    fig.tight_layout()


    fig.savefig(
        OUTPUT_DIR
        / "tau_marginal_posterior.png",
        dpi=300,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


    # ========================================================
    # MARGINAL ELL POSTERIOR
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(
            7,
            5,
        )
    )


    ax.plot(
        ELL_GRID,
        ell_posterior,
        marker="o",
    )


    ax.set_xlabel(
        "Correlation length ell (km)"
    )

    ax.set_ylabel(
        "Posterior probability"
    )

    ax.set_title(
        "Marginal posterior for ell"
    )


    fig.tight_layout()


    fig.savefig(
        OUTPUT_DIR
        / "ell_marginal_posterior.png",
        dpi=300,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


    # ========================================================
    # NEW CHECK 1:
    # LOG EVIDENCE VS TAU
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(
            8,
            6,
        )
    )


    for j, ell in enumerate(
        ELL_GRID
    ):

        ax.plot(
            TAU_GRID,
            log_evidence_grid[
                :,
                j,
            ],
            marker="o",
            label=(
                f"ell={ell:.0f} km"
            ),
        )


    ax.set_xlabel(
        "tau"
    )

    ax.set_ylabel(
        "Log evidence"
    )

    ax.set_title(
        "Log evidence vs tau"
    )


    ax.legend(
        fontsize=8
    )


    fig.tight_layout()


    fig.savefig(
        OUTPUT_DIR
        / "log_evidence_vs_tau.png",
        dpi=300,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


    # ========================================================
    # NEW CHECK 2:
    # LOG EVIDENCE VS ELL
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(
            8,
            6,
        )
    )


    for i, tau in enumerate(
        TAU_GRID
    ):

        ax.plot(
            ELL_GRID,
            log_evidence_grid[
                i,
                :,
            ],
            marker="o",
            label=(
                f"tau={tau:.4f}"
            ),
        )


    ax.set_xlabel(
        "Correlation length ell (km)"
    )

    ax.set_ylabel(
        "Log evidence"
    )

    ax.set_title(
        "Log evidence vs ell"
    )


    ax.legend(
        fontsize=8
    )


    fig.tight_layout()


    fig.savefig(
        OUTPUT_DIR
        / "log_evidence_vs_ell.png",
        dpi=300,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


    # ========================================================
    # NEW CHECK 3:
    # PRINT LOG-EVIDENCE TABLE
    # ========================================================

    print()

    print(
        "=============================================================="
    )

    print(
        "LOG EVIDENCE TABLE"
    )

    print(
        "rows = tau, columns = ell"
    )

    print(
        "=============================================================="
    )


    print(
        f"{'tau':>10}",
        end="",
    )


    for ell in ELL_GRID:

        print(
            f"{ell:>12.0f}",
            end="",
        )


    print()


    for i, tau in enumerate(
        TAU_GRID
    ):

        print(
            f"{tau:10.4f}",
            end="",
        )


        for j in range(
            len(
                ELL_GRID
            )
        ):

            value = (
                log_evidence_grid[
                    i,
                    j,
                ]
            )


            if np.isfinite(
                value
            ):

                print(
                    f"{value:12.1f}",
                    end="",
                )

            else:

                print(
                    f"{'NA':>12}",
                    end="",
                )


        print()


    # ========================================================
    # CHECK WHETHER MAP IS ON A GRID BOUNDARY
    # ========================================================

    print()

    print(
        "=============================================================="
    )

    print(
        "BOUNDARY CHECK"
    )

    print(
        "=============================================================="
    )


    print(
        f"MAP tau = {map_tau:.6f}"
    )

    print(
        f"MAP ell = {map_ell:.1f} km"
    )


    tau_at_lower_boundary = np.isclose(
        map_tau,
        TAU_GRID.min(),
    )

    tau_at_upper_boundary = np.isclose(
        map_tau,
        TAU_GRID.max(),
    )

    ell_at_lower_boundary = np.isclose(
        map_ell,
        ELL_GRID.min(),
    )

    ell_at_upper_boundary = np.isclose(
        map_ell,
        ELL_GRID.max(),
    )


    if (
        tau_at_lower_boundary
        or tau_at_upper_boundary
    ):

        print(
            "WARNING: MAP tau is on a grid boundary."
        )

        print(
            "Consider extending the tau grid."
        )

    else:

        print(
            "MAP tau is inside the grid."
        )


    if (
        ell_at_lower_boundary
        or ell_at_upper_boundary
    ):

        print(
            "WARNING: MAP ell is on a grid boundary."
        )

        print(
            "Consider extending the ell grid."
        )

    else:

        print(
            "MAP ell is inside the grid."
        )


    # ========================================================
    # MAP EARTH-MODEL POSTERIOR
    # ========================================================

    logger.info(
        "Calculating Earth posterior at MAP tau/ell..."
    )


    map_covariance = (
        construct_covariance(
            path_distance_squared=(
                path_distance_squared
            ),
            sigma=sigma,
            tau=map_tau,
            ell=map_ell,
        )
    )


    map_nuisance = [
        GaussianComponent(
            np.eye(
                n_data
            ),
            np.zeros(
                n_data
            ),
            map_covariance,
        )
    ]


    posterior_mean = (
        calc_posterior_mean(
            data,
            inferred,
            map_nuisance,
        )
    )


    posterior_covariance = (
        calc_posterior_cov(
            inferred,
            map_nuisance,
        )
    )


    np.save(
        OUTPUT_DIR
        / "earth_posterior_mean_MAP.npy",
        posterior_mean,
    )


    np.save(
        OUTPUT_DIR
        / "earth_posterior_covariance_MAP.npy",
        posterior_covariance,
    )

    # ------------------------------------------------------------
    # MAP Earth-model posterior standard deviations
    # ------------------------------------------------------------

    posterior_sd = np.sqrt(
        np.diag(
            posterior_covariance
        )
    )

    np.save(
        OUTPUT_DIR
        / "earth_posterior_sd_MAP.npy",
        posterior_sd,
    )

    print()
    print(
        "MAP Earth posterior mean shape =",
        posterior_mean.shape,
    )

    print(
        "MAP Earth posterior covariance shape =",
        posterior_covariance.shape,
    )

    print(
        "MAP Earth posterior SD shape =",
        posterior_sd.shape,
    )

    print(
        "MAP Earth posterior mean range =",
        posterior_mean.min(),
        "to",
        posterior_mean.max(),
    )

    print(
        "MAP Earth posterior SD range =",
        posterior_sd.min(),
        "to",
        posterior_sd.max(),
    )


    # ============================================================
    # BASELINE BLOCK-IID EARTH-MODEL POSTERIOR
    # ============================================================

    logger.info(
        "Calculating baseline block-IID Earth posterior (tau=0)..."
    )

    # When tau = 0, the correlated covariance component vanishes:
    #
    #     C = diag(sigma^2)
    #
    # and ell becomes irrelevant. Any positive ell can therefore
    # be supplied to construct_covariance.
    iid_covariance = (
        construct_covariance(
            path_distance_squared=(
                path_distance_squared
            ),
            sigma=sigma,
            tau=0.0,
            ell=float(
                ELL_GRID[0]
            ),
        )
    )

    iid_nuisance = [
        GaussianComponent(
            np.eye(
                n_data
            ),
            np.zeros(
                n_data
            ),
            iid_covariance,
        )
    ]


    # ------------------------------------------------------------
    # Baseline block-IID log evidence
    # ------------------------------------------------------------

    iid_log_evidence = (
        calc_log_evidence(
            data,
            inferred,
            iid_nuisance,
        )
    )


    # ------------------------------------------------------------
    # Baseline block-IID posterior mean
    # ------------------------------------------------------------

    iid_posterior_mean = (
        calc_posterior_mean(
            data,
            inferred,
            iid_nuisance,
        )
    )


    # ------------------------------------------------------------
    # Baseline block-IID posterior covariance
    # ------------------------------------------------------------

    iid_posterior_covariance = (
        calc_posterior_cov(
            inferred,
            iid_nuisance,
        )
    )


    # ------------------------------------------------------------
    # Baseline block-IID posterior standard deviations
    # ------------------------------------------------------------

    iid_posterior_sd = np.sqrt(
        np.diag(
            iid_posterior_covariance
        )
    )


    # ------------------------------------------------------------
    # Save baseline block-IID results
    # ------------------------------------------------------------

    np.save(
        OUTPUT_DIR
        / "earth_posterior_mean_block_iid.npy",
        iid_posterior_mean,
    )

    np.save(
        OUTPUT_DIR
        / "earth_posterior_covariance_block_iid.npy",
        iid_posterior_covariance,
    )

    np.save(
        OUTPUT_DIR
        / "earth_posterior_sd_block_iid.npy",
        iid_posterior_sd,
    )

    baseline_summary = pd.DataFrame(
        [
            {
                "model": "block_iid",
                "tau": 0.0,
                "ell_km": np.nan,
                "log_evidence": (
                    iid_log_evidence
                ),
                "n_observations": (
                    n_data
                ),
                "n_parameters": (
                    n_parameters
                ),
                "radial_resolution": (
                    RADIAL_RESOLUTION
                ),
                "lateral_resolution": (
                    LATERAL_RESOLUTION
                ),
                "prior_variance": (
                    PRIOR_VARIANCE
                ),
                "posterior_mean_min": (
                    iid_posterior_mean.min()
                ),
                "posterior_mean_max": (
                    iid_posterior_mean.max()
                ),
                "posterior_sd_min": (
                    iid_posterior_sd.min()
                ),
                "posterior_sd_max": (
                    iid_posterior_sd.max()
                ),
            }
        ]
    )

    baseline_summary.to_csv(
        OUTPUT_DIR
        / "block_iid_summary.csv",
        index=False,
    )


    # ------------------------------------------------------------
    # Print baseline block-IID diagnostics
    # ------------------------------------------------------------

    print()
    print(
        "=============================================================="
    )
    print(
        "BASELINE BLOCK-IID MODEL"
    )
    print(
        "=============================================================="
    )

    print(
        f"Block-IID log evidence = {iid_log_evidence:.6f}"
    )

    print(
        "Posterior mean shape =",
        iid_posterior_mean.shape,
    )

    print(
        "Posterior covariance shape =",
        iid_posterior_covariance.shape,
    )

    print(
        "Posterior SD shape =",
        iid_posterior_sd.shape,
    )

    print(
        "Posterior mean range =",
        iid_posterior_mean.min(),
        "to",
        iid_posterior_mean.max(),
    )

    print(
        "Posterior SD range =",
        iid_posterior_sd.min(),
        "to",
        iid_posterior_sd.max(),
    )

    print(
        "Saved block-IID outputs in:",
        OUTPUT_DIR,
    )


    # ------------------------------------------------------------
    # Direct MAP-correlated vs block-IID diagnostic
    # ------------------------------------------------------------

    map_log_evidence = (
        log_evidence_grid[
            map_index
        ]
    )

    delta_log_evidence_map_vs_iid = (
        map_log_evidence
        - iid_log_evidence
    )

    log_bayes_factor_corr_vs_iid = (
        correlated_log_evidence_marginalized
        - iid_log_evidence
    )

    mean_difference = (
        posterior_mean
        - iid_posterior_mean
    )

    sd_difference = (
        posterior_sd
        - iid_posterior_sd
    )

    np.save(
        OUTPUT_DIR
        / "earth_posterior_mean_difference_MAP_minus_block_iid.npy",
        mean_difference,
    )

    np.save(
        OUTPUT_DIR
        / "earth_posterior_sd_difference_MAP_minus_block_iid.npy",
        sd_difference,
    )

    comparison_summary = pd.DataFrame(
        [
            {
                "iid_log_evidence": (
                    iid_log_evidence
                ),
                "map_tau": (
                    map_tau
                ),
                "map_ell_km": (
                    map_ell
                ),
                "map_log_evidence": (
                    map_log_evidence
                ),
                "delta_log_evidence_MAP_minus_IID": (
                    delta_log_evidence_map_vs_iid
                ),
                "marginalized_correlated_log_evidence": (
                    correlated_log_evidence_marginalized
                ),
                "log_bayes_factor_corr_vs_iid": (
                    log_bayes_factor_corr_vs_iid
                ),
                "mean_abs_posterior_mean_difference": (
                    np.mean(
                        np.abs(
                            mean_difference
                        )
                    )
                ),
                "max_abs_posterior_mean_difference": (
                    np.max(
                        np.abs(
                            mean_difference
                        )
                    )
                ),
                "mean_posterior_sd_difference": (
                    np.mean(
                        sd_difference
                    )
                ),
                "max_abs_posterior_sd_difference": (
                    np.max(
                        np.abs(
                            sd_difference
                        )
                    )
                ),
            }
        ]
    )

    comparison_summary.to_csv(
        OUTPUT_DIR
        / "MAP_vs_block_iid_summary.csv",
        index=False,
    )

    print()
    print(
        "=============================================================="
    )
    print(
        "MAP CORRELATED MODEL VS BLOCK-IID"
    )
    print(
        "=============================================================="
    )

    print(
        f"Block-IID log evidence = {iid_log_evidence:.6f}"
    )

    print(
        f"MAP correlated log evidence = {map_log_evidence:.6f}"
    )

    print(
        "Delta log evidence (MAP - IID) =",
        delta_log_evidence_map_vs_iid,
    )

    print(
        "Marginalized correlated log evidence =",
        correlated_log_evidence_marginalized,
    )

    print(
        "Focused-grid log evidence difference (correlated - IID) =",
        log_bayes_factor_corr_vs_iid,
    )

    print(
        "Mean absolute posterior-mean difference =",
        np.mean(
            np.abs(
                mean_difference
            )
        ),
    )

    print(
        "Maximum absolute posterior-mean difference =",
        np.max(
            np.abs(
                mean_difference
            )
        ),
    )

    print(
        "Mean posterior-SD difference =",
        np.mean(
            sd_difference
        ),
    )

    print(
        "Maximum absolute posterior-SD difference =",
        np.max(
            np.abs(
                sd_difference
            )
        ),
    )


    # ============================================================
    # POSTERIOR UNCERTAINTY DIAGNOSTICS FOR TAU AND ELL
    # ============================================================

    # ------------------------------------------------------------
    # Posterior means
    # ------------------------------------------------------------

    posterior_mean_tau = float(
        np.sum(
            TAU_GRID
            * tau_posterior
        )
    )

    posterior_mean_ell = float(
        np.sum(
            ELL_GRID
            * ell_posterior
        )
    )

    print()
    print("==============================================================")
    print("POSTERIOR MEANS")
    print("==============================================================")

    print(
        f"Posterior mean tau = {posterior_mean_tau:.6f}"
    )

    print(
        f"Posterior mean ell = {posterior_mean_ell:.2f} km"
    )


    # ------------------------------------------------------------
    # Posterior standard deviations
    # ------------------------------------------------------------

    posterior_var_tau = float(
        np.sum(
            (
                TAU_GRID
                - posterior_mean_tau
            ) ** 2
            * tau_posterior
        )
    )

    posterior_sd_tau = np.sqrt(
        posterior_var_tau
    )

    posterior_var_ell = float(
        np.sum(
            (
                ELL_GRID
                - posterior_mean_ell
            ) ** 2
            * ell_posterior
        )
    )

    posterior_sd_ell = np.sqrt(
        posterior_var_ell
    )

    print()
    print("Posterior SD tau =", posterior_sd_tau)

    print(
        "Posterior SD ell =",
        posterior_sd_ell,
        "km",
    )


    # ------------------------------------------------------------
    # Posterior covariance between tau and ell
    # ------------------------------------------------------------

    posterior_cov_tau_ell = 0.0

    for i, tau in enumerate(
        TAU_GRID
    ):

        for j, ell in enumerate(
            ELL_GRID
        ):

            posterior_cov_tau_ell += (
                posterior_weights[
                    i,
                    j,
                ]
                * (
                    tau
                    - posterior_mean_tau
                )
                * (
                    ell
                    - posterior_mean_ell
                )
            )


    posterior_corr_tau_ell = (
        posterior_cov_tau_ell
        / (
            posterior_sd_tau
            * posterior_sd_ell
        )
    )

    print()
    print("==============================================================")
    print("TAU / ELL POSTERIOR DEPENDENCE")
    print("==============================================================")

    print(
        "Posterior covariance(tau, ell) =",
        posterior_cov_tau_ell,
    )

    print(
        "Posterior correlation(tau, ell) =",
        posterior_corr_tau_ell,
    )


    # ------------------------------------------------------------
    # Discrete posterior credible intervals
    # ------------------------------------------------------------

    def discrete_credible_interval(
        grid,
        probabilities,
        level=0.95,
    ):
        """Equal-tail credible interval for a discrete posterior."""

        cumulative = np.cumsum(
            probabilities
        )

        lower_probability = (
            0.5
            * (
                1.0
                - level
            )
        )

        upper_probability = (
            1.0
            - lower_probability
        )

        lower_index = np.searchsorted(
            cumulative,
            lower_probability,
        )

        upper_index = np.searchsorted(
            cumulative,
            upper_probability,
        )

        lower_index = min(
            lower_index,
            len(grid) - 1,
        )

        upper_index = min(
            upper_index,
            len(grid) - 1,
        )

        return (
            grid[
                lower_index
            ],
            grid[
                upper_index
            ],
        )


    tau_ci_95 = discrete_credible_interval(
        TAU_GRID,
        tau_posterior,
        level=0.95,
    )

    ell_ci_95 = discrete_credible_interval(
        ELL_GRID,
        ell_posterior,
        level=0.95,
    )

    print()
    print("==============================================================")
    print("95% POSTERIOR CREDIBLE INTERVALS")
    print("==============================================================")

    print(
        "tau 95% credible interval:",
        tau_ci_95,
    )

    print(
        "ell 95% credible interval:",
        ell_ci_95,
        "km",
    )


    # ------------------------------------------------------------
    # Top posterior grid points
    # ------------------------------------------------------------

    top_rows = []

    for i, tau in enumerate(
        TAU_GRID
    ):

        for j, ell in enumerate(
            ELL_GRID
        ):

            top_rows.append(
                {
                    "tau": tau,
                    "ell_km": ell,
                    "posterior_probability": (
                        posterior_weights[
                            i,
                            j,
                        ]
                    ),
                    "log_evidence": (
                        log_evidence_grid[
                            i,
                            j,
                        ]
                    ),
                    "log_posterior": (
                        log_posterior_grid[
                            i,
                            j,
                        ]
                    ),
                }
            )


    top_df = pd.DataFrame(
        top_rows
    )

    top_df = (
        top_df
        .sort_values(
            "posterior_probability",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    print()
    print("==============================================================")
    print("TOP 10 POSTERIOR GRID POINTS")
    print("==============================================================")

    print(
        top_df.head(
            10
        ).to_string(
            index=False
        )
    )


    top_df.to_csv(
        OUTPUT_DIR
        / "tau_ell_ranked_posterior.csv",
        index=False,
    )


    # ------------------------------------------------------------
    # Plot posterior contours
    # ------------------------------------------------------------

    ELL_MESH, TAU_MESH = np.meshgrid(
        ELL_GRID,
        TAU_GRID,
    )

    fig, ax = plt.subplots(
        figsize=(
            8,
            6,
        )
    )

    contours = ax.contour(
        ELL_MESH,
        TAU_MESH,
        posterior_weights,
        levels=8,
    )

    ax.clabel(
        contours,
        inline=True,
        fontsize=8,
    )

    ax.scatter(
        [map_ell],
        [map_tau],
        marker="x",
        s=80,
        label="MAP",
    )

    ax.set_xlabel(
        "Correlation length ell (km)"
    )

    ax.set_ylabel(
        "Correlated-error SD tau"
    )

    ax.set_title(
        "Joint posterior contours for tau and ell"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "tau_ell_posterior_contours.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()

    print(
        "=============================================================="
    )

    print(
        "FINAL SUMMARY"
    )

    print(
        "=============================================================="
    )


    print(
        f"MAP tau = {map_tau:.6f}"
    )

    print(
        f"MAP ell = {map_ell:.1f} km"
    )


    print()

    print(
        "Files saved in:"
    )

    print(
        OUTPUT_DIR
    )


    logger.info(
        "Finished."
    )
    

        # ============================================================
    # METHODOLOGY VERIFICATION CHECKS
    # ============================================================
    
    print()
    print("==============================================================")
    print("METHODOLOGY VERIFICATION")
    print("==============================================================")
    
    # ------------------------------------------------------------
    # 1. Number of observations
    # ------------------------------------------------------------
    
    print()
    print("1. DATA SIZE")
    print("------------------------------")
    
    print("Number of observations n =", len(data))
    
    assert len(data) == len(df), (
        "Mismatch between data vector length and dataframe length."
    )
    
    print("Dataframe rows =", len(df))
    
    
    # ------------------------------------------------------------
    # 2. Verify fractional travel-time definition
    #    d_i = delta_t / inner_core_travel_time
    # ------------------------------------------------------------
    
    print()
    print("2. FRACTIONAL TRAVEL-TIME DATA")
    print("------------------------------")
    
    data_manual = (
        df["delta_t"].astype(float).to_numpy()
        /
        df["inner_core_travel_time"].astype(float).to_numpy()
    )
    
    max_data_difference = np.max(
        np.abs(data - data_manual)
    )
    
    print(
        "Maximum difference between stored data and "
        "delta_t / inner_core_travel_time =",
        max_data_difference,
    )
    
    assert np.allclose(
        data,
        data_manual,
    ), (
        "The data vector is not exactly "
        "delta_t / inner_core_travel_time."
    )
    
    print(
        "Verified: "
        "d_i = delta_t_i / inner_core_travel_time_i"
    )
    
    
    # ------------------------------------------------------------
    # 3. Verify phase-dependent noise levels
    # ------------------------------------------------------------
    
    print()
    print("3. PHASE-DEPENDENT NOISE")
    print("------------------------------")
    
    noise_levels_seconds = {
        "ab": 0.95,
        "bc": 0.63,
        "cd": 0.29,
        "df": 0.95,
    }
    
    print("Noise levels in seconds:")
    for phase, value in noise_levels_seconds.items():
        print(
            f"  {phase}: {value:.2f} s"
        )
    
    sigma_manual = (
        df["reference_phase"]
        .map(noise_levels_seconds)
        .astype(float)
        .to_numpy()
        /
        df["inner_core_travel_time"]
        .astype(float)
        .to_numpy()
    )
    
    if np.any(
        pd.isna(
            df["reference_phase"]
            .map(noise_levels_seconds)
        )
    ):
        unknown_phases = (
            df.loc[
                df["reference_phase"]
                .map(noise_levels_seconds)
                .isna(),
                "reference_phase",
            ]
            .unique()
        )
    
        print(
            "WARNING: unknown reference phases:",
            unknown_phases,
        )
    
    else:
        max_sigma_difference = np.max(
            np.abs(
                sigma - sigma_manual
            )
        )
    
        print(
            "Maximum difference between calculate_noise_sigma "
            "and manual phase-based sigma =",
            max_sigma_difference,
        )
    
        assert np.allclose(
            sigma,
            sigma_manual,
        ), (
            "calculate_noise_sigma does not match "
            "the stated phase-dependent noise model."
        )
    
        print(
            "Verified: sigma_i = "
            "sigma_phase / inner_core_travel_time_i"
        )
    
    
    # ------------------------------------------------------------
    # 4. Show phase counts
    # ------------------------------------------------------------
    
    print()
    print("4. NUMBER OF OBSERVATIONS BY REFERENCE PHASE")
    print("------------------------------")
    
    phase_counts = (
        df["reference_phase"]
        .value_counts()
        .sort_index()
    )
    
    print(phase_counts)
    
    
    # ------------------------------------------------------------
    # 5. Verify spherical mesh settings
    # ------------------------------------------------------------
    
    print()
    print("5. SPHERICAL MESH")
    print("------------------------------")
    
    print(
        "Inner-core radius used by mesh =",
        1221.5,
        "km",
    )
    
    print(
        "Radial resolution =",
        RADIAL_RESOLUTION,
    )
    
    print(
        "Lateral resolution =",
        LATERAL_RESOLUTION,
    )
    
    print(
        "Mesh object =",
        mesh,
    )
    
    
    # ------------------------------------------------------------
    # 6. Verify forward-matrix dimensions
    # ------------------------------------------------------------
    
    print()
    print("6. FORWARD MATRIX")
    print("------------------------------")
    
    print(
        "Forward matrix shape =",
        forward_matrix.shape,
    )
    
    n_observations_forward = (
        forward_matrix.shape[0]
    )
    
    n_parameters_forward = (
        forward_matrix.shape[1]
    )
    
    print(
        "Rows of G =",
        n_observations_forward,
    )
    
    print(
        "Columns of G =",
        n_parameters_forward,
    )
    
    assert (
        n_observations_forward
        == len(data)
    ), (
        "The number of rows of G does not match "
        "the number of observations."
    )
    
    print(
        "Verified: "
        "G has one row per observation."
    )
    
    
    # ------------------------------------------------------------
    # 7. Check whether there are 3 model parameters per segment/cell
    # ------------------------------------------------------------
    
    print()
    print("7. MODEL PARAMETER COUNT")
    print("------------------------------")
    
    print(
        "Total number of model parameters p =",
        n_parameters_forward,
    )
    
    if (
        n_parameters_forward % 3
        == 0
    ):
    
        inferred_number_of_segments = (
            n_parameters_forward // 3
        )
    
        print(
            "p is divisible by 3."
        )
    
        print(
            "If there are exactly 3 parameters "
            "per spatial segment/cell, then:"
        )
    
        print(
            "N = p / 3 =",
            inferred_number_of_segments,
        )
    
        print(
            "Therefore p = 3N =",
            3 * inferred_number_of_segments,
        )
    
    else:
    
        print(
            "WARNING: p is not divisible by 3."
        )
    
        print(
            "Do NOT write p = 3N until "
            "construct_forward_map is inspected."
        )
    
    
    # ------------------------------------------------------------
    # 8. Check entry and exit geometry
    # ------------------------------------------------------------
    
    print()
    print("8. INNER-CORE ENTRY / EXIT LOCATIONS")
    print("------------------------------")
    
    print(
        "ic_in shape =",
        ic_in.shape,
    )
    
    print(
        "ic_out shape =",
        ic_out.shape,
    )
    
    assert (
        ic_in.shape[0]
        == len(data)
    ), (
        "Number of entry locations does not match "
        "number of observations."
    )
    
    assert (
        ic_out.shape[0]
        == len(data)
    ), (
        "Number of exit locations does not match "
        "number of observations."
    )
    
    print(
        "Verified: every observation has "
        "an inner-core entry and exit location."
    )
    
    
    # ------------------------------------------------------------
    # 9. Verify path-distance matrix
    # ------------------------------------------------------------
    
    print()
    print("9. PATH-DISTANCE MATRIX")
    print("------------------------------")
    
    print(
        "Squared path-distance matrix shape =",
        path_distance_squared.shape,
    )
    
    assert (
        path_distance_squared.shape
        == (
            len(data),
            len(data),
        )
    ), (
        "Path-distance matrix does not have "
        "shape n x n."
    )
    
    print(
        "Minimum squared path distance =",
        np.min(
            path_distance_squared
        ),
    )
    
    print(
        "Maximum squared path distance =",
        np.max(
            path_distance_squared
        ),
    )
    
    print(
        "Maximum asymmetry =",
        np.max(
            np.abs(
                path_distance_squared
                - path_distance_squared.T
            )
        ),
    )
    
    print(
        "Maximum diagonal value =",
        np.max(
            np.abs(
                np.diag(
                    path_distance_squared
                )
            )
        ),
    )
    
    
    # ------------------------------------------------------------
    # 10. Verify Earth-model prior covariance
    # ------------------------------------------------------------
    
    print()
    print("10. EARTH-MODEL PRIOR")
    print("------------------------------")
    
    print(
        "PRIOR_VARIANCE =",
        PRIOR_VARIANCE,
    )
    
    print(
        "Prior covariance is "
        "PRIOR_VARIANCE * I_p"
    )
    
    print(
        "Therefore C_m =",
        PRIOR_VARIANCE,
        "* I_p",
    )
    
    print(
        "Prior standard deviation =",
        np.sqrt(
            PRIOR_VARIANCE
        ),
    )
    
    
    # ------------------------------------------------------------
    # 11. Verify covariance hyperparameter priors
    # ------------------------------------------------------------
    
    print()
    print("11. COVARIANCE HYPERPARAMETER PRIORS")
    print("------------------------------")
    
    print(
        "tau ~ HalfNormal(scale =",
        TAU_PRIOR_SCALE,
        ")",
    )
    
    print(
        "ell ~ Uniform(",
        ELL_MIN,
        ",",
        ELL_MAX,
        ") km",
    )
    
    
    # ------------------------------------------------------------
    # 12. Verify current covariance kernel
    # ------------------------------------------------------------
    
    print()
    print("12. CURRENT CORRELATED COVARIANCE KERNEL")
    print("------------------------------")
    
    print(
        "Current implementation uses:"
    )
    
    print(
        "C_delta_ij = tau^2 * "
        "exp(-d_ij^2 / (2 * ell^2))"
    )
    
    print(
        "This is the squared-exponential / RBF kernel."
    )
    
    print()
    print("==============================================================")
    print("END OF METHODOLOGY VERIFICATION")
    print("==============================================================")

if __name__ == "__main__":
    main()