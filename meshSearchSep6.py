"""
Bayesian grid search over spherical-mesh resolutions.

Purpose
-------
Select the Earth-model mesh resolution under the baseline
block-IID observational error model.

For every candidate mesh:
    1. Construct the spherical mesh.
    2. Construct the forward matrix G.
    3. Use the same Earth-model prior:
           m ~ N(0, 0.01 I)
    4. Use the same block-IID observational covariance.
    5. Calculate the exact Bayesian log evidence.
    6. Save the result immediately.

Paper-ready outputs
-------------------
1. mesh_evidence_results.csv
2. mesh_evidence_ranked.csv
3. mesh_log_evidence_heatmap.png
4. mesh_delta_log_evidence_heatmap.png
5. mesh_evidence_profiles.png
6. mesh_top10_table.png
"""

import gc
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from linear_gaussian import (
    GaussianComponent,
    calc_log_evidence,
)

from raytracer import SphericalMesh

from tti.traveltimes.traveltimes import (
    calculate_path_direction_vector,
)

# Only import functions that actually exist in main.py
from main import (
    construct_forward_map,
    determine_weights,
)


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

logger = logging.getLogger(__name__)


# ============================================================
# SETTINGS
# ============================================================

DATA_FILE = Path(
    "data/brett2024_ic_traveltimes.parquet"
)

# New output directory so calculations using a different
# Earth-model prior cannot be accidentally reused.
OUTPUT_DIR = Path(
    "outputs/mesh_grid_search_prior_0p01"
)

RESULTS_FILE = (
    OUTPUT_DIR
    / "mesh_evidence_results.csv"
)

RANKED_FILE = (
    OUTPUT_DIR
    / "mesh_evidence_ranked.csv"
)

LOG_EVIDENCE_HEATMAP_FILE = (
    OUTPUT_DIR
    / "mesh_log_evidence_heatmap.png"
)

DELTA_LOG_EVIDENCE_HEATMAP_FILE = (
    OUTPUT_DIR
    / "mesh_delta_log_evidence_heatmap.png"
)

EVIDENCE_PROFILE_FILE = (
    OUTPUT_DIR
    / "mesh_evidence_profiles.png"
)

TOP10_TABLE_FILE = (
    OUTPUT_DIR
    / "mesh_top10_table.png"
)


# ============================================================
# EARTH-MODEL SETTINGS
# ============================================================

INNER_CORE_RADIUS = 1221.5

N_PARAMETERS_PER_CELL = 3

# Same prior used in the correlated-error analysis:
#
#     m ~ N(0, 0.01 I)
#
PRIOR_VARIANCE = 0.01


# ============================================================
# MESH SEARCH GRID
# ============================================================

# radial resolution = 1, ..., 10
RADIAL_VALUES = range(
    1,
    11,
)

# lateral resolution = 1, ..., 20
LATERAL_VALUES = range(
    1,
    21,
)


# ============================================================
# BLOCK-IID OBSERVATIONAL COVARIANCE
# ============================================================


def construct_Cd(
    reference_phase,
    inner_core_travel_time,
):
    """
    Construct the baseline block-IID observational covariance.

    A different observational standard deviation is used
    for each PKP reference phase.

    Standard deviations are specified in seconds and then
    converted to fractional travel-time units:

        sigma_fractional_i
            = sigma_seconds_i / T_i

    where T_i is the inner-core travel time.

    The covariance matrix is diagonal:

        Cd = diag(sigma_fractional_i^2)

    Parameters
    ----------
    reference_phase
        Phase labels for each observation.

    inner_core_travel_time
        Inner-core travel time for each observation.

    Returns
    -------
    Cd
        Diagonal observational covariance matrix.
    """

    # --------------------------------------------------------
    # Phase-dependent observational uncertainty in seconds
    # --------------------------------------------------------

    phase_sigma_seconds = {
        "ab": 0.95,
        "bc": 0.63,
        "cd": 0.29,
        "df": 0.95,
    }


    reference_phase = np.asarray(
        reference_phase
    ).astype(str)


    inner_core_travel_time = np.asarray(
        inner_core_travel_time,
        dtype=float,
    )


    if (
        len(reference_phase)
        != len(inner_core_travel_time)
    ):

        raise ValueError(
            "reference_phase and inner_core_travel_time "
            "must have the same length."
        )


    if np.any(
        inner_core_travel_time <= 0
    ):

        raise ValueError(
            "All inner-core travel times must be positive."
        )


    n_data = len(
        reference_phase
    )


    sigma_seconds = np.empty(
        n_data,
        dtype=float,
    )


    # --------------------------------------------------------
    # Assign sigma according to phase
    # --------------------------------------------------------

    phase_counts = {
        "ab": 0,
        "bc": 0,
        "cd": 0,
        "df": 0,
    }


    for i, phase in enumerate(
        reference_phase
    ):

        phase_lower = (
            phase
            .strip()
            .lower()
        )

        matched = False


        for (
            phase_type,
            sigma,
        ) in phase_sigma_seconds.items():

            if phase_lower.endswith(
                phase_type
            ):

                sigma_seconds[
                    i
                ] = sigma

                phase_counts[
                    phase_type
                ] += 1

                matched = True

                break


        if not matched:

            raise ValueError(
                "Unknown reference phase "
                f"'{phase}' at observation {i}."
            )


    logger.info(
        "Phase counts: "
        "ab=%d, bc=%d, cd=%d, df=%d",
        phase_counts["ab"],
        phase_counts["bc"],
        phase_counts["cd"],
        phase_counts["df"],
    )


    # --------------------------------------------------------
    # Convert seconds -> fractional travel-time uncertainty
    #
    # sigma_fractional_i
    #     = sigma_seconds_i / T_i
    # --------------------------------------------------------

    sigma_fractional = (
        sigma_seconds
        / inner_core_travel_time
    )


    # --------------------------------------------------------
    # Block-IID covariance
    #
    # Cd = diag(sigma_i^2)
    # --------------------------------------------------------

    Cd = np.diag(
        sigma_fractional**2
    )


    logger.info(
        "Constructed block-IID covariance "
        "with shape %s",
        Cd.shape,
    )


    return Cd


# ============================================================
# LOAD DATA
# ============================================================


def load_data() -> tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Load travel-time data and calculate path directions.

    The observed data are fractional travel-time residuals:

        d_i = delta_t_i / inner_core_travel_time_i
    """

    logger.info(
        "Reading data from %s",
        DATA_FILE,
    )


    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"Could not find {DATA_FILE}. "
            "Make sure you run this script from "
            "the icanilg project folder."
        )


    df = pd.read_parquet(
        DATA_FILE
    )


    # --------------------------------------------------------
    # Fractional travel-time residual
    # --------------------------------------------------------

    data = (
        df["delta_t"]
        / df["inner_core_travel_time"]
    ).astype(
        float
    ).to_numpy()


    # --------------------------------------------------------
    # Entry and exit locations
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Path direction vectors
    # --------------------------------------------------------

    path_directions = (
        calculate_path_direction_vector(
            ic_in,
            ic_out,
        )
    )


    logger.info(
        "Loaded %d observations.",
        len(data),
    )


    logger.info(
        "Data range: %.6e to %.6e",
        np.min(data),
        np.max(data),
    )


    return (
        df,
        data,
        ic_in,
        path_directions,
    )


# ============================================================
# RUN ONE MESH
# ============================================================


def run_one_resolution(
    radial_resolution: int,
    lateral_resolution: int,
    data: np.ndarray,
    ic_in: np.ndarray,
    path_directions: np.ndarray,
    nuisance: list[GaussianComponent],
) -> dict:
    """
    Calculate exact log evidence for one mesh resolution.

    The Earth prior and observational error model remain fixed.
    Only the mesh resolution changes.
    """

    logger.info(
        "============================================================"
    )

    logger.info(
        "Starting radial=%d, lateral=%d",
        radial_resolution,
        lateral_resolution,
    )


    start_time = (
        time.perf_counter()
    )


    # ========================================================
    # Construct spherical mesh
    # ========================================================

    mesh = SphericalMesh(
        INNER_CORE_RADIUS,
        radial_resolution,
        lateral_resolution,
    )


    # ========================================================
    # Calculate ray-path weights
    # ========================================================

    weights = determine_weights(
        mesh,
        ic_in,
        path_directions,
    )


    # ========================================================
    # Construct forward matrix G
    # ========================================================

    forward_matrix = (
        construct_forward_map(
            path_directions,
            weights,
        )
    )


    (
        n_data,
        n_parameters,
    ) = forward_matrix.shape


    # ========================================================
    # Sanity checks
    # ========================================================

    if n_data != len(data):

        raise ValueError(
            f"Forward matrix has {n_data} rows, "
            f"but data contain {len(data)} observations."
        )


    if (
        n_parameters
        % N_PARAMETERS_PER_CELL
        != 0
    ):

        raise ValueError(
            f"Number of parameters {n_parameters} "
            f"is not divisible by "
            f"{N_PARAMETERS_PER_CELL}."
        )


    n_cells = (
        n_parameters
        // N_PARAMETERS_PER_CELL
    )


    logger.info(
        "Forward matrix shape = %s",
        forward_matrix.shape,
    )

    logger.info(
        "Number of mesh cells = %d",
        n_cells,
    )

    logger.info(
        "Number of Earth parameters = %d",
        n_parameters,
    )


    # ========================================================
    # Earth-model prior
    #
    # m ~ N(0, 0.01 I)
    # ========================================================

    prior_mean = np.zeros(
        n_parameters
    )


    prior_covariance = (
        PRIOR_VARIANCE
        * np.eye(
            n_parameters
        )
    )


    inferred = [
        GaussianComponent(
            forward_matrix,
            prior_mean,
            prior_covariance,
        )
    ]


    # ========================================================
    # Exact Bayesian log evidence
    # ========================================================

    log_evidence = (
        calc_log_evidence(
            data,
            inferred,
            nuisance,
        )
    )


    runtime_seconds = (
        time.perf_counter()
        - start_time
    )


    # ========================================================
    # Result
    # ========================================================

    result = {

        "radial_resolution":
            radial_resolution,

        "lateral_resolution":
            lateral_resolution,

        "n_cells":
            n_cells,

        "n_parameters":
            n_parameters,

        "prior_variance":
            PRIOR_VARIANCE,

        "log_evidence":
            float(log_evidence),

        "runtime_seconds":
            runtime_seconds,

        "status":
            "success",

        "error":
            "",
    }


    logger.info(
        "Finished radial=%d, lateral=%d: "
        "cells=%d, parameters=%d, "
        "log evidence=%.6f, runtime=%.2f s",
        radial_resolution,
        lateral_resolution,
        n_cells,
        n_parameters,
        float(log_evidence),
        runtime_seconds,
    )


    # ========================================================
    # Release large objects
    # ========================================================

    del inferred
    del prior_covariance
    del prior_mean
    del forward_matrix
    del weights
    del mesh

    gc.collect()


    return result


# ============================================================
# LOAD PREVIOUS RESULTS
# ============================================================


def load_previous_results() -> pd.DataFrame:
    """
    Load existing results so an interrupted calculation
    can resume.
    """

    if RESULTS_FILE.exists():

        logger.info(
            "Loading existing results from %s",
            RESULTS_FILE,
        )


        results = pd.read_csv(
            RESULTS_FILE
        )


        # ----------------------------------------------------
        # Important safety check
        # ----------------------------------------------------

        if (
            "prior_variance"
            not in results.columns
            and not results.empty
        ):

            raise ValueError(
                "Existing result file does not contain "
                "'prior_variance'. "
                "It may have been generated using an older "
                "version of the code. "
                "Delete the output directory or use a new one."
            )


        if not results.empty:

            previous_prior = (
                pd.to_numeric(
                    results[
                        "prior_variance"
                    ],
                    errors="coerce",
                )
                .dropna()
                .unique()
            )


            if (
                len(previous_prior) > 0
                and not np.allclose(
                    previous_prior,
                    PRIOR_VARIANCE,
                )
            ):

                raise ValueError(
                    "Existing mesh-search results were "
                    "calculated using a different prior "
                    "variance. "
                    "Use a different OUTPUT_DIR."
                )


        return results


    return pd.DataFrame(
        columns=[
            "radial_resolution",
            "lateral_resolution",
            "n_cells",
            "n_parameters",
            "prior_variance",
            "log_evidence",
            "runtime_seconds",
            "status",
            "error",
        ]
    )


# ============================================================
# CHECK FOR COMPLETED RESULT
# ============================================================


def result_already_exists(
    results: pd.DataFrame,
    radial_resolution: int,
    lateral_resolution: int,
) -> bool:
    """
    Return True if this mesh was already calculated
    successfully.
    """

    if results.empty:
        return False


    matching = results[
        (
            results[
                "radial_resolution"
            ]
            == radial_resolution
        )
        &
        (
            results[
                "lateral_resolution"
            ]
            == lateral_resolution
        )
        &
        (
            results[
                "status"
            ]
            == "success"
        )
    ]


    return not matching.empty


# ============================================================
# SAVE RESULT
# ============================================================


def save_result(
    results: pd.DataFrame,
    result: dict,
) -> pd.DataFrame:
    """
    Append one result and immediately save the CSV.
    """

    new_row = pd.DataFrame(
        [result]
    )


    updated = pd.concat(
        [
            results,
            new_row,
        ],
        ignore_index=True,
    )


    updated = (
        updated
        .sort_values(
            [
                "radial_resolution",
                "lateral_resolution",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    updated.to_csv(
        RESULTS_FILE,
        index=False,
    )


    return updated


# ============================================================
# PREPARE SUCCESSFUL RESULTS
# ============================================================


def prepare_successful_results(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep successful numerical results and calculate

        Delta log Z = log Z - max(log Z).
    """

    successful = results[
        results[
            "status"
        ]
        == "success"
    ].copy()


    successful[
        "log_evidence"
    ] = pd.to_numeric(
        successful[
            "log_evidence"
        ],
        errors="coerce",
    )


    successful = (
        successful
        .dropna(
            subset=[
                "log_evidence"
            ]
        )
    )


    if successful.empty:

        return successful


    best_log_evidence = (
        successful[
            "log_evidence"
        ].max()
    )


    successful[
        "delta_log_evidence"
    ] = (
        successful[
            "log_evidence"
        ]
        - best_log_evidence
    )


    return successful


# ============================================================
# LOG-EVIDENCE HEATMAP
# ============================================================


def plot_log_evidence_heatmap(
    results: pd.DataFrame,
) -> None:
    """
    Plot the absolute Bayesian log evidence.
    """

    successful = (
        prepare_successful_results(
            results
        )
    )


    if successful.empty:
        return


    evidence_grid = (
        successful
        .pivot_table(
            index="radial_resolution",
            columns="lateral_resolution",
            values="log_evidence",
            aggfunc="first",
        )
        .sort_index(
            axis=0
        )
        .sort_index(
            axis=1
        )
    )


    evidence_array = (
        evidence_grid
        .to_numpy(
            dtype=float
        )
    )


    fig, ax = plt.subplots(
        figsize=(
            12,
            7,
        )
    )


    image = ax.imshow(
        evidence_array,
        origin="lower",
        aspect="auto",
    )


    ax.set_xticks(
        np.arange(
            len(
                evidence_grid.columns
            )
        )
    )

    ax.set_xticklabels(
        evidence_grid.columns
    )


    ax.set_yticks(
        np.arange(
            len(
                evidence_grid.index
            )
        )
    )

    ax.set_yticklabels(
        evidence_grid.index
    )


    ax.set_xlabel(
        "Lateral resolution"
    )

    ax.set_ylabel(
        "Radial resolution"
    )

    ax.set_title(
        "Bayesian Log Evidence for Mesh Resolution"
    )


    for i in range(
        evidence_grid.shape[0]
    ):

        for j in range(
            evidence_grid.shape[1]
        ):

            value = (
                evidence_array[
                    i,
                    j,
                ]
            )


            if np.isfinite(
                value
            ):

                ax.text(
                    j,
                    i,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )


    colorbar = (
        fig.colorbar(
            image,
            ax=ax,
        )
    )


    colorbar.set_label(
        "Log evidence"
    )


    fig.tight_layout()


    fig.savefig(
        LOG_EVIDENCE_HEATMAP_FILE,
        dpi=300,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


# ============================================================
# DELTA LOG-EVIDENCE HEATMAP
# ============================================================


def plot_delta_log_evidence_heatmap(
    results: pd.DataFrame,
) -> None:
    """
    Plot evidence relative to the best mesh.

        Delta log Z
            = log Z - max(log Z)

    The best mesh therefore has Delta log Z = 0.
    """

    successful = (
        prepare_successful_results(
            results
        )
    )


    if successful.empty:
        return


    delta_grid = (
        successful
        .pivot_table(
            index="radial_resolution",
            columns="lateral_resolution",
            values="delta_log_evidence",
            aggfunc="first",
        )
        .sort_index(
            axis=0
        )
        .sort_index(
            axis=1
        )
    )


    delta_array = (
        delta_grid
        .to_numpy(
            dtype=float
        )
    )


    fig, ax = plt.subplots(
        figsize=(
            12,
            7,
        )
    )


    image = ax.imshow(
        delta_array,
        origin="lower",
        aspect="auto",
    )


    ax.set_xticks(
        np.arange(
            len(
                delta_grid.columns
            )
        )
    )

    ax.set_xticklabels(
        delta_grid.columns
    )


    ax.set_yticks(
        np.arange(
            len(
                delta_grid.index
            )
        )
    )

    ax.set_yticklabels(
        delta_grid.index
    )


    ax.set_xlabel(
        "Lateral resolution"
    )

    ax.set_ylabel(
        "Radial resolution"
    )

    ax.set_title(
        "Log-Evidence Difference Relative to Best Mesh"
    )


    for i in range(
        delta_grid.shape[0]
    ):

        for j in range(
            delta_grid.shape[1]
        ):

            value = (
                delta_array[
                    i,
                    j,
                ]
            )


            if np.isfinite(
                value
            ):

                ax.text(
                    j,
                    i,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )


    colorbar = (
        fig.colorbar(
            image,
            ax=ax,
        )
    )


    colorbar.set_label(
        r"$\Delta \log Z$"
    )


    fig.tight_layout()


    fig.savefig(
        DELTA_LOG_EVIDENCE_HEATMAP_FILE,
        dpi=300,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


# ============================================================
# EVIDENCE PROFILES
# ============================================================


def plot_evidence_profiles(
    results: pd.DataFrame,
) -> None:
    """
    Plot log evidence against lateral resolution,
    with one line for each radial resolution.
    """

    successful = (
        prepare_successful_results(
            results
        )
    )


    if successful.empty:
        return


    fig, ax = plt.subplots(
        figsize=(
            10,
            6,
        )
    )


    radial_values = sorted(
        successful[
            "radial_resolution"
        ].unique()
    )


    for radial in radial_values:

        subset = (
            successful[
                successful[
                    "radial_resolution"
                ]
                == radial
            ]
            .sort_values(
                "lateral_resolution"
            )
        )


        ax.plot(
            subset[
                "lateral_resolution"
            ],
            subset[
                "log_evidence"
            ],
            marker="o",
            label=(
                f"Radial = {int(radial)}"
            ),
        )


    ax.set_xlabel(
        "Lateral resolution"
    )

    ax.set_ylabel(
        "Log evidence"
    )

    ax.set_title(
        "Log Evidence Across Mesh Resolutions"
    )


    ax.legend(
        fontsize=8,
        ncol=2,
    )


    fig.tight_layout()


    fig.savefig(
        EVIDENCE_PROFILE_FILE,
        dpi=300,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


# ============================================================
# TOP-10 TABLE
# ============================================================


def plot_top10_table(
    ranked: pd.DataFrame,
) -> None:
    """
    Save a table figure showing the ten highest-evidence meshes.
    """

    if ranked.empty:
        return


    top10 = (
        ranked
        .head(10)
        .copy()
    )


    display_table = pd.DataFrame(
        {
            "Radial":
                top10[
                    "radial_resolution"
                ].astype(int),

            "Lateral":
                top10[
                    "lateral_resolution"
                ].astype(int),

            "Cells":
                top10[
                    "n_cells"
                ].astype(int),

            "Parameters":
                top10[
                    "n_parameters"
                ].astype(int),

            "log Z":
                top10[
                    "log_evidence"
                ].map(
                    lambda x:
                    f"{x:.2f}"
                ),

            "Delta log Z":
                top10[
                    "delta_log_evidence"
                ].map(
                    lambda x:
                    f"{x:.2f}"
                ),
        }
    )


    fig, ax = plt.subplots(
        figsize=(
            10,
            4.5,
        )
    )


    ax.axis(
        "off"
    )


    table = ax.table(
        cellText=(
            display_table.values
        ),
        colLabels=(
            display_table.columns
        ),
        loc="center",
        cellLoc="center",
    )


    table.auto_set_font_size(
        False
    )

    table.set_fontsize(
        9
    )

    table.scale(
        1.0,
        1.4,
    )


    ax.set_title(
        "Highest-Evidence Mesh Resolutions",
        pad=15,
    )


    fig.tight_layout()


    fig.savefig(
        TOP10_TABLE_FILE,
        dpi=300,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


# ============================================================
# UPDATE FIGURES
# ============================================================


def update_figures(
    results: pd.DataFrame,
) -> None:
    """
    Update all mesh-search figures.
    """

    plot_log_evidence_heatmap(
        results
    )

    plot_delta_log_evidence_heatmap(
        results
    )

    plot_evidence_profiles(
        results
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    """
    Run the complete block-IID mesh-resolution evidence search.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    logger.info(
        "Outputs will be saved to %s",
        OUTPUT_DIR.resolve(),
    )


    # ========================================================
    # Load data
    # ========================================================

    (
        df,
        data,
        ic_in,
        path_directions,
    ) = load_data()


    n_data = len(
        data
    )


    # ========================================================
    # Construct block-IID observational covariance ONCE
    #
    # This remains identical for every mesh resolution.
    # ========================================================

    logger.info(
        "Constructing baseline block-IID "
        "observational covariance..."
    )


    noise_covariance = (
        construct_Cd(
            df[
                "reference_phase"
            ],
            df[
                "inner_core_travel_time"
            ],
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
            noise_covariance,
        )
    ]


    logger.info(
        "Number of observations = %d",
        n_data,
    )

    logger.info(
        "Earth-model prior variance = %.6f",
        PRIOR_VARIANCE,
    )

    logger.info(
        "Earth-model prior SD = %.6f",
        np.sqrt(
            PRIOR_VARIANCE
        ),
    )

    logger.info(
        "Searching radial resolutions %d to %d",
        min(
            RADIAL_VALUES
        ),
        max(
            RADIAL_VALUES
        ),
    )

    logger.info(
        "Searching lateral resolutions %d to %d",
        min(
            LATERAL_VALUES
        ),
        max(
            LATERAL_VALUES
        ),
    )


    # ========================================================
    # Load previous results if this run was interrupted
    # ========================================================

    results = (
        load_previous_results()
    )


    # ========================================================
    # Mesh search
    # ========================================================

    for radial_resolution in RADIAL_VALUES:

        for lateral_resolution in LATERAL_VALUES:


            # ------------------------------------------------
            # Skip already completed meshes
            # ------------------------------------------------

            if result_already_exists(
                results,
                radial_resolution,
                lateral_resolution,
            ):

                logger.info(
                    "Skipping completed "
                    "radial=%d, lateral=%d",
                    radial_resolution,
                    lateral_resolution,
                )

                continue


            # ------------------------------------------------
            # Run mesh
            # ------------------------------------------------

            try:

                result = (
                    run_one_resolution(
                        radial_resolution=(
                            radial_resolution
                        ),
                        lateral_resolution=(
                            lateral_resolution
                        ),
                        data=data,
                        ic_in=ic_in,
                        path_directions=(
                            path_directions
                        ),
                        nuisance=nuisance,
                    )
                )


            except Exception as exc:

                logger.exception(
                    "Failed radial=%d, lateral=%d",
                    radial_resolution,
                    lateral_resolution,
                )


                result = {

                    "radial_resolution":
                        radial_resolution,

                    "lateral_resolution":
                        lateral_resolution,

                    "n_cells":
                        np.nan,

                    "n_parameters":
                        np.nan,

                    "prior_variance":
                        PRIOR_VARIANCE,

                    "log_evidence":
                        np.nan,

                    "runtime_seconds":
                        np.nan,

                    "status":
                        "failed",

                    "error":
                        str(exc),
                }


            # ------------------------------------------------
            # Save immediately
            # ------------------------------------------------

            results = save_result(
                results,
                result,
            )


            # ------------------------------------------------
            # Update figures
            # ------------------------------------------------

            update_figures(
                results
            )


    # ========================================================
    # FINAL RESULTS
    # ========================================================

    successful = (
        prepare_successful_results(
            results
        )
    )


    if successful.empty:

        logger.warning(
            "No successful mesh results."
        )

        return


    # ========================================================
    # Rank meshes
    # ========================================================

    ranked = (
        successful
        .sort_values(
            "log_evidence",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


    ranked.to_csv(
        RANKED_FILE,
        index=False,
    )


    # ========================================================
    # Final figures
    # ========================================================

    update_figures(
        results
    )


    plot_top10_table(
        ranked
    )


    # ========================================================
    # Best mesh
    # ========================================================

    best = ranked.iloc[
        0
    ]


    print()

    print(
        "=============================================================="
    )

    print(
        "MESH SEARCH FINAL SUMMARY"
    )

    print(
        "=============================================================="
    )


    print(
        "Prior variance =",
        PRIOR_VARIANCE,
    )


    print(
        "Prior SD =",
        np.sqrt(
            PRIOR_VARIANCE
        ),
    )


    print(
        "Best radial resolution =",
        int(
            best[
                "radial_resolution"
            ]
        ),
    )


    print(
        "Best lateral resolution =",
        int(
            best[
                "lateral_resolution"
            ]
        ),
    )


    print(
        "Number of cells =",
        int(
            best[
                "n_cells"
            ]
        ),
    )


    print(
        "Number of Earth parameters =",
        int(
            best[
                "n_parameters"
            ]
        ),
    )


    print(
        "Maximum log evidence =",
        float(
            best[
                "log_evidence"
            ]
        ),
    )


    # ========================================================
    # Print top 10
    # ========================================================

    print()

    print(
        "=============================================================="
    )

    print(
        "TOP 10 MESH RESOLUTIONS"
    )

    print(
        "=============================================================="
    )


    print(
        ranked[
            [
                "radial_resolution",
                "lateral_resolution",
                "n_cells",
                "n_parameters",
                "log_evidence",
                "delta_log_evidence",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )


    # ========================================================
    # Print output paths
    # ========================================================

    print()

    print(
        "=============================================================="
    )

    print(
        "OUTPUT FILES"
    )

    print(
        "=============================================================="
    )


    print(
        "Output directory:"
    )

    print(
        OUTPUT_DIR.resolve()
    )


    print()

    print(
        RESULTS_FILE
    )

    print(
        RANKED_FILE
    )

    print(
        LOG_EVIDENCE_HEATMAP_FILE
    )

    print(
        DELTA_LOG_EVIDENCE_HEATMAP_FILE
    )

    print(
        EVIDENCE_PROFILE_FILE
    )

    print(
        TOP10_TABLE_FILE
    )


    logger.info(
        "Mesh grid search complete."
    )


# ============================================================
# RUN
# ============================================================


if __name__ == "__main__":
    main()