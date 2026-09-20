"""
Improved posterior-sensitivity plots for the selected 10 x 5
inner-core parameterisation.

This script compares:

    1. Block-IID posterior
    2. Hyperparameter-marginalized correlated posterior

for:

    delta A / A
    delta C / C
    delta F / F

Selected mesh:
    radial resolution  = 10
    lateral resolution = 5
    cells per shell    = 45
    total cells        = 450
    parameters/cell    = 3
    total parameters   = 1350

Parameter ordering:
    [A1, C1, F1, A2, C2, F2, ...]

The upstream inversion used prior variance 0.01
(prior standard deviation = 0.1).

IMPORTANT:
This script does not recompute the Bayesian inversion.
It loads previously calculated posterior results.

Figure improvements:
    - separate Block-IID and correlated plots;
    - Block-IID shown in blue;
    - correlated posterior shown in orange;
    - larger fonts;
    - stronger alternating radial-shell shading;
    - clearer shell boundaries;
    - radial direction shown as:
          Inner-core centre  ----------------->  ICB
    - larger figure size;
    - 400 dpi output;
    - new output directory.

Run from the project root:

    uv run python analyze_earth_posteriors_r10_l5_full72_improved.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# SETTINGS
# =============================================================================

# Hyperparameter-marginalized correlated posterior.
MARG_DIR = Path(
    "outputs/earth_posterior_marginalized_r10_l5_prior_0p01_full72"
)

# Matched block-IID posterior.
IID_DIR = Path(
    "outputs/tau_ell_posterior_r10_l5_prior_0p01_coarse40"
)

# New output directory.
# Existing figures/results will not be overwritten.
PLOT_DIR = Path(
    "outputs/"
    "posterior_sensitivity_r10_l5_prior_0p01_full72_improved_figures"
)


# =============================================================================
# MESH INFORMATION
# =============================================================================

INNER_CORE_RADIUS_KM = 1221.5

N_RADIAL = 10

# Lateral resolution = 5:
# 5 latitude bands x 9 longitude sectors = 45 cells per shell.
N_LATERAL_CELLS_PER_SHELL = 45

EXPECTED_N_CELLS = (
    N_RADIAL
    * N_LATERAL_CELLS_PER_SHELL
)

# A, C and F for every spatial cell.
EXPECTED_N_PARAMETERS = (
    EXPECTED_N_CELLS * 3
)


# =============================================================================
# PLOT SETTINGS
# =============================================================================

plt.rcParams.update(
    {
        "font.size": 16,
        "axes.titlesize": 19,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
    }
)


# =============================================================================
# FILE HELPERS
# =============================================================================

def first_existing(directory, candidates):
    """
    Return the first existing file from a list of possible filenames.
    """

    for name in candidates:

        path = directory / name

        if path.exists():

            print(f"Using: {path}")

            return path

    return None


def interleave_acf(A, C, F):
    """
    Reconstruct:

        [A1, C1, F1, A2, C2, F2, ...]

    from separate A, C and F arrays.
    """

    A = np.asarray(A)
    C = np.asarray(C)
    F = np.asarray(F)

    if not (
        A.shape
        == C.shape
        == F.shape
    ):

        raise ValueError(
            "A, C and F arrays do not have matching shapes."
        )

    vector = np.empty(
        A.size * 3,
        dtype=float,
    )

    vector[0::3] = A
    vector[1::3] = C
    vector[2::3] = F

    return vector


def split_acf(vector):
    """
    Split:

        [A1, C1, F1, A2, C2, F2, ...]

    into separate A, C and F arrays.
    """

    vector = np.asarray(vector)

    if vector.ndim != 1:

        raise ValueError(
            f"Expected a 1D vector, got {vector.shape}"
        )

    if len(vector) % 3 != 0:

        raise ValueError(
            "Posterior vector length is not divisible by 3."
        )

    return (
        vector[0::3],
        vector[1::3],
        vector[2::3],
    )


# =============================================================================
# LOAD MARGINALIZED CORRELATED POSTERIOR
# =============================================================================

def load_marginalized_mean():
    """
    Load the hyperparameter-marginalized posterior mean.
    """

    full_file = first_existing(
        MARG_DIR,
        [
            "earth_posterior_mean_marginalized.npy",
            "earth_posterior_mean_marginalised.npy",
            "marginalized_earth_posterior_mean.npy",
            "marginalised_earth_posterior_mean.npy",
        ],
    )

    if full_file is not None:

        return np.load(full_file)

    # Try separate A/C/F arrays.
    A_file = first_existing(
        MARG_DIR,
        [
            "marginalized_A_posterior_mean.npy",
            "marginalised_A_posterior_mean.npy",
        ],
    )

    C_file = first_existing(
        MARG_DIR,
        [
            "marginalized_C_posterior_mean.npy",
            "marginalised_C_posterior_mean.npy",
        ],
    )

    F_file = first_existing(
        MARG_DIR,
        [
            "marginalized_F_posterior_mean.npy",
            "marginalised_F_posterior_mean.npy",
        ],
    )

    if (
        A_file is None
        or C_file is None
        or F_file is None
    ):

        raise FileNotFoundError(
            "\nCould not find the marginalized posterior mean.\n"
            f"Searched inside:\n    {MARG_DIR}\n"
        )

    A = np.load(A_file)
    C = np.load(C_file)
    F = np.load(F_file)

    return interleave_acf(
        A,
        C,
        F,
    )


def load_marginalized_sd():
    """
    Load hyperparameter-marginalized posterior standard deviations.
    """

    full_sd_file = first_existing(
        MARG_DIR,
        [
            "earth_posterior_sd_marginalized.npy",
            "earth_posterior_sd_marginalised.npy",
            "marginalized_earth_posterior_sd.npy",
            "marginalised_earth_posterior_sd.npy",
        ],
    )

    if full_sd_file is not None:

        return np.load(full_sd_file)

    # Try covariance matrix.
    covariance_file = first_existing(
        MARG_DIR,
        [
            "earth_posterior_covariance_marginalized.npy",
            "earth_posterior_covariance_marginalised.npy",
            "marginalized_earth_posterior_covariance.npy",
            "marginalised_earth_posterior_covariance.npy",
        ],
    )

    if covariance_file is not None:

        covariance = np.load(
            covariance_file
        )

        return np.sqrt(
            np.maximum(
                np.diag(covariance),
                0.0,
            )
        )

    # Try separate A/C/F SD arrays.
    A_file = first_existing(
        MARG_DIR,
        [
            "marginalized_A_posterior_sd.npy",
            "marginalised_A_posterior_sd.npy",
        ],
    )

    C_file = first_existing(
        MARG_DIR,
        [
            "marginalized_C_posterior_sd.npy",
            "marginalised_C_posterior_sd.npy",
        ],
    )

    F_file = first_existing(
        MARG_DIR,
        [
            "marginalized_F_posterior_sd.npy",
            "marginalised_F_posterior_sd.npy",
        ],
    )

    if (
        A_file is None
        or C_file is None
        or F_file is None
    ):

        raise FileNotFoundError(
            "\nCould not find marginalized posterior SDs.\n"
            f"Searched inside:\n    {MARG_DIR}\n"
        )

    A = np.load(A_file)
    C = np.load(C_file)
    F = np.load(F_file)

    return interleave_acf(
        A,
        C,
        F,
    )


# =============================================================================
# LOAD BLOCK-IID POSTERIOR
# =============================================================================

def load_iid_mean():
    """
    Load block-IID posterior mean.
    """

    mean_file = first_existing(
        IID_DIR,
        [
            "earth_posterior_mean_block_iid.npy",
            "earth_posterior_mean_block_IID.npy",
        ],
    )

    if mean_file is None:

        raise FileNotFoundError(
            "\nCould not find the block-IID posterior mean.\n"
            f"Searched inside:\n    {IID_DIR}\n"
        )

    return np.load(
        mean_file
    )


def load_iid_sd():
    """
    Load block-IID posterior standard deviations.
    """

    sd_file = first_existing(
        IID_DIR,
        [
            "earth_posterior_sd_block_iid.npy",
            "earth_posterior_sd_block_IID.npy",
        ],
    )

    if sd_file is not None:

        return np.load(
            sd_file
        )

    covariance_file = first_existing(
        IID_DIR,
        [
            "earth_posterior_covariance_block_iid.npy",
            "earth_posterior_covariance_block_IID.npy",
        ],
    )

    if covariance_file is None:

        raise FileNotFoundError(
            "\nCould not find the block-IID posterior SD "
            "or covariance matrix.\n"
            f"Searched inside:\n    {IID_DIR}\n"
        )

    covariance = np.load(
        covariance_file
    )

    return np.sqrt(
        np.maximum(
            np.diag(covariance),
            0.0,
        )
    )


# =============================================================================
# RADIAL AXIS
# =============================================================================

def create_radial_axis():
    """
    Construct the grouped radial-shell x-axis.

    There are 450 spatial cells.

    Every 45 consecutive cells belong to one radial shell.

    The ten radial shells are ordered from the centre of the
    inner core toward the inner-core boundary (ICB).

    Therefore radius increases from left to right.
    """

    x = np.arange(
        EXPECTED_N_CELLS
    )

    # Equal radial subdivision from 0 km to 1221.5 km.
    radial_edges = np.linspace(
        0.0,
        INNER_CORE_RADIUS_KM,
        N_RADIAL + 1,
    )

    radial_centres = (
        radial_edges[:-1]
        + radial_edges[1:]
    ) / 2.0

    # Centre of each 45-cell shell on the x-axis.
    tick_positions = (
        np.arange(N_RADIAL)
        * N_LATERAL_CELLS_PER_SHELL
        + (
            N_LATERAL_CELLS_PER_SHELL - 1
        ) / 2.0
    )

    tick_labels = [
        f"{radius:.0f}"
        for radius in radial_centres
    ]

    shell_number = np.repeat(
        np.arange(
            1,
            N_RADIAL + 1,
        ),
        N_LATERAL_CELLS_PER_SHELL,
    )

    shell_radius_per_cell = np.repeat(
        radial_centres,
        N_LATERAL_CELLS_PER_SHELL,
    )

    return (
        x,
        radial_edges,
        radial_centres,
        tick_positions,
        tick_labels,
        shell_number,
        shell_radius_per_cell,
    )


def format_radial_axis(
    ax,
    tick_positions,
    tick_labels,
):
    """
    Apply common formatting to the radial-shell x-axis.

    Alternating shading distinguishes the ten radial shells.

    Radius increases from the inner-core centre toward the ICB.
    """

    # -------------------------------------------------------------------------
    # Alternating radial-shell shading
    # -------------------------------------------------------------------------

    for shell in range(N_RADIAL):

        left = (
            shell
            * N_LATERAL_CELLS_PER_SHELL
            - 0.5
        )

        right = (
            (shell + 1)
            * N_LATERAL_CELLS_PER_SHELL
            - 0.5
        )

        # Shade every second radial shell.
        if shell % 2 == 1:

            ax.axvspan(
                left,
                right,
                color="lightblue",
                alpha=0.30,
                zorder=0,
            )

    # -------------------------------------------------------------------------
    # Radial-shell boundaries
    # -------------------------------------------------------------------------

    for shell in range(1, N_RADIAL):

        boundary = (
            shell
            * N_LATERAL_CELLS_PER_SHELL
            - 0.5
        )

        ax.axvline(
            boundary,
            color="steelblue",
            linewidth=0.9,
            alpha=0.45,
            zorder=1,
        )

    # -------------------------------------------------------------------------
    # Radial-shell labels
    # -------------------------------------------------------------------------

    ax.set_xticks(
        tick_positions
    )

    ax.set_xticklabels(
        tick_labels,
        fontsize=15,
    )

    ax.set_xlabel(
        "Radial shell centre (km)",
        fontsize=18,
        labelpad=10,
    )

    ax.set_xlim(
        -0.8,
        EXPECTED_N_CELLS - 0.2,
    )

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=15,
    )

    # -------------------------------------------------------------------------
    # Radial direction:
    #
    # Inner-core centre ----------------------> ICB
    # -------------------------------------------------------------------------

    ax.annotate(
        "Inner-core centre",
        xy=(0.93, -0.19),
        xytext=(0.12, -0.19),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="->",
            color="black",
            linewidth=1.5,
        ),
        color="black",
        ha="center",
        va="center",
        fontsize=15,
        annotation_clip=False,
    )

    ax.text(
        0.95,
        -0.19,
        "ICB",
        transform=ax.transAxes,
        color="black",
        fontsize=15,
        ha="left",
        va="center",
    )


# =============================================================================
# SEPARATE POSTERIOR ERROR-BAR FIGURES
# =============================================================================

def plot_single_posterior_errorbars(
    x,
    tick_positions,
    tick_labels,
    mean,
    sd,
    parameter,
    model_name,
    filename,
    color,
):
    """
    Plot one posterior separately.

    Points:
        posterior means

    Error bars:
        +/- 1 posterior standard deviation
    """

    fig, ax = plt.subplots(
        figsize=(20, 6.5)
    )

    ax.errorbar(
        x,
        100.0 * mean,
        yerr=100.0 * sd,
        fmt="o",
        linestyle="none",
        markersize=3.6,
        elinewidth=0.8,
        capsize=1.5,
        color=color,
        zorder=3,
    )

    ax.axhline(
        0.0,
        color="black",
        linewidth=1.0,
        zorder=2,
    )

    format_radial_axis(
        ax,
        tick_positions,
        tick_labels,
    )

    ax.set_ylabel(
        rf"Posterior mean of "
        rf"$\delta {parameter}/{parameter}$ (\%)",
        fontsize=18,
        labelpad=10,
    )

    ax.set_title(
        rf"$\delta {parameter}/{parameter}$: "
        f"{model_name} posterior",
        fontsize=19,
        pad=12,
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.90,
        bottom=0.24,
    )

    fig.savefig(
        PLOT_DIR / filename,
        dpi=400,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# POSTERIOR-MEAN DIFFERENCE FIGURES
# =============================================================================

def plot_mean_difference(
    x,
    tick_positions,
    tick_labels,
    difference,
    parameter,
):

    fig, ax = plt.subplots(
        figsize=(16, 6)
    )

    ax.scatter(
        x,
        100.0 * difference,
        s=22,
        color="tab:purple",
        zorder=3,
    )

    ax.axhline(
        0.0,
        color="black",
        linewidth=1.0,
        zorder=2,
    )

    format_radial_axis(
        ax,
        tick_positions,
        tick_labels,
    )

    ax.set_ylabel(
        rf"Change in posterior mean of "
        rf"$\delta {parameter}/{parameter}$"
        "\n(percentage points)",
        fontsize=18,
        labelpad=10,
    )

    ax.set_title(
        rf"$\delta {parameter}/{parameter}$: "
        "correlated minus block-IID posterior mean",
        fontsize=19,
        pad=12,
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.985,
        top=0.90,
        bottom=0.24,
    )

    fig.savefig(
        PLOT_DIR
        / (
            f"{parameter}_posterior_mean_difference_"
            "correlated_minus_IID_improved.png"
        ),
        dpi=400,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# POSTERIOR-SD DIFFERENCE FIGURES
# =============================================================================

def plot_sd_difference(
    x,
    tick_positions,
    tick_labels,
    difference,
    parameter,
):

    fig, ax = plt.subplots(
        figsize=(16, 6)
    )

    ax.scatter(
        x,
        100.0 * difference,
        s=22,
        color="tab:green",
        zorder=3,
    )

    ax.axhline(
        0.0,
        color="black",
        linewidth=1.0,
        zorder=2,
    )

    format_radial_axis(
        ax,
        tick_positions,
        tick_labels,
    )

    ax.set_ylabel(
        rf"Change in posterior SD of "
        rf"$\delta {parameter}/{parameter}$"
        "\n(percentage points)",
        fontsize=18,
        labelpad=10,
    )

    ax.set_title(
        rf"$\delta {parameter}/{parameter}$: "
        "change in posterior uncertainty",
        fontsize=19,
        pad=12,
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.985,
        top=0.90,
        bottom=0.24,
    )

    fig.savefig(
        PLOT_DIR
        / (
            f"{parameter}_posterior_sd_difference_"
            "correlated_minus_IID_improved.png"
        ),
        dpi=400,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# COMBINED POSTERIOR-MEAN SENSITIVITY
# =============================================================================

def plot_combined_mean_difference(
    x,
    tick_positions,
    tick_labels,
    delta_A,
    delta_C,
    delta_F,
):

    fig, ax = plt.subplots(
        figsize=(16, 6.5)
    )

    offsets = {
        "A": -0.18,
        "C": 0.0,
        "F": 0.18,
    }

    ax.scatter(
        x + offsets["A"],
        100.0 * delta_A,
        s=22,
        label=r"$\delta A/A$",
        zorder=3,
    )

    ax.scatter(
        x + offsets["C"],
        100.0 * delta_C,
        s=22,
        label=r"$\delta C/C$",
        zorder=3,
    )

    ax.scatter(
        x + offsets["F"],
        100.0 * delta_F,
        s=22,
        label=r"$\delta F/F$",
        zorder=3,
    )

    ax.axhline(
        0.0,
        color="black",
        linewidth=1.0,
        zorder=2,
    )

    format_radial_axis(
        ax,
        tick_positions,
        tick_labels,
    )

    ax.set_ylabel(
        "Change in posterior mean\n"
        "(correlated - block IID; percentage points)",
        fontsize=18,
        labelpad=10,
    )

    ax.set_title(
        "Sensitivity of posterior means "
        "to the observational error model",
        fontsize=19,
        pad=12,
    )

    ax.legend(
        frameon=True,
        fontsize=15,
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.985,
        top=0.90,
        bottom=0.24,
    )

    fig.savefig(
        PLOT_DIR
        / "posterior_mean_iid_vs_correlated_improved.png",
        dpi=400,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# COMBINED POSTERIOR-SD SENSITIVITY
# =============================================================================

def plot_combined_sd_difference(
    x,
    tick_positions,
    tick_labels,
    delta_A,
    delta_C,
    delta_F,
):

    fig, ax = plt.subplots(
        figsize=(16, 6.5)
    )

    offsets = {
        "A": -0.18,
        "C": 0.0,
        "F": 0.18,
    }

    ax.scatter(
        x + offsets["A"],
        100.0 * delta_A,
        s=22,
        label=r"$\delta A/A$",
        zorder=3,
    )

    ax.scatter(
        x + offsets["C"],
        100.0 * delta_C,
        s=22,
        label=r"$\delta C/C$",
        zorder=3,
    )

    ax.scatter(
        x + offsets["F"],
        100.0 * delta_F,
        s=22,
        label=r"$\delta F/F$",
        zorder=3,
    )

    ax.axhline(
        0.0,
        color="black",
        linewidth=1.0,
        zorder=2,
    )

    format_radial_axis(
        ax,
        tick_positions,
        tick_labels,
    )

    ax.set_ylabel(
        "Change in posterior SD\n"
        "(correlated - block IID; percentage points)",
        fontsize=18,
        labelpad=10,
    )

    ax.set_title(
        "Change in posterior uncertainty "
        "under correlated observational errors",
        fontsize=19,
        pad=12,
    )

    ax.legend(
        frameon=True,
        fontsize=15,
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.985,
        top=0.90,
        bottom=0.24,
    )

    fig.savefig(
        PLOT_DIR
        / "posterior_sd_difference_correlated_minus_iid_improved.png",
        dpi=400,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 72)
    print("LOADING POSTERIOR RESULTS")
    print("=" * 72)

    # -------------------------------------------------------------------------
    # Load posterior results
    # -------------------------------------------------------------------------

    marg_mean = load_marginalized_mean()
    marg_sd = load_marginalized_sd()

    iid_mean = load_iid_mean()
    iid_sd = load_iid_sd()

    # -------------------------------------------------------------------------
    # Dimension checks
    # -------------------------------------------------------------------------

    if not (
        marg_mean.shape
        == marg_sd.shape
        == iid_mean.shape
        == iid_sd.shape
    ):

        raise ValueError(
            "\nPosterior arrays do not have matching shapes:\n"
            f"marg_mean = {marg_mean.shape}\n"
            f"marg_sd   = {marg_sd.shape}\n"
            f"iid_mean  = {iid_mean.shape}\n"
            f"iid_sd    = {iid_sd.shape}\n"
        )

    if len(marg_mean) != EXPECTED_N_PARAMETERS:

        raise ValueError(
            f"\nExpected {EXPECTED_N_PARAMETERS} parameters "
            f"for the 10 x 5 mesh, but found {len(marg_mean)}."
        )

    print()
    print(
        "Posterior vector length =",
        len(marg_mean),
    )

    print(
        "Number of spatial cells =",
        len(marg_mean) // 3,
    )

    print(
        "Parameter ordering = "
        "[A1, C1, F1, A2, C2, F2, ...]"
    )

    # -------------------------------------------------------------------------
    # Radial axis
    # -------------------------------------------------------------------------

    (
        x,
        radial_edges,
        radial_centres,
        tick_positions,
        tick_labels,
        shell_number,
        shell_radius_per_cell,
    ) = create_radial_axis()

    print()
    print("Radial shell edges (km):")

    print(
        np.round(
            radial_edges,
            2,
        )
    )

    print()
    print("Radial shell centres (km):")

    print(
        np.round(
            radial_centres,
            2,
        )
    )

    print()
    print(
        "Radius increases from the inner-core centre "
        "toward the ICB."
    )

    # -------------------------------------------------------------------------
    # Split A/C/F
    # -------------------------------------------------------------------------

    (
        marg_A_mean,
        marg_C_mean,
        marg_F_mean,
    ) = split_acf(
        marg_mean
    )

    (
        marg_A_sd,
        marg_C_sd,
        marg_F_sd,
    ) = split_acf(
        marg_sd
    )

    (
        iid_A_mean,
        iid_C_mean,
        iid_F_mean,
    ) = split_acf(
        iid_mean
    )

    (
        iid_A_sd,
        iid_C_sd,
        iid_F_sd,
    ) = split_acf(
        iid_sd
    )

    # -------------------------------------------------------------------------
    # Differences
    # -------------------------------------------------------------------------

    delta_A_mean = (
        marg_A_mean
        - iid_A_mean
    )

    delta_C_mean = (
        marg_C_mean
        - iid_C_mean
    )

    delta_F_mean = (
        marg_F_mean
        - iid_F_mean
    )

    delta_A_sd = (
        marg_A_sd
        - iid_A_sd
    )

    delta_C_sd = (
        marg_C_sd
        - iid_C_sd
    )

    delta_F_sd = (
        marg_F_sd
        - iid_F_sd
    )

    # =========================================================================
    # SAVE ARRAYS
    # =========================================================================

    np.save(
        PLOT_DIR / "marginalized_A_posterior_mean.npy",
        marg_A_mean,
    )

    np.save(
        PLOT_DIR / "marginalized_C_posterior_mean.npy",
        marg_C_mean,
    )

    np.save(
        PLOT_DIR / "marginalized_F_posterior_mean.npy",
        marg_F_mean,
    )

    np.save(
        PLOT_DIR / "marginalized_A_posterior_sd.npy",
        marg_A_sd,
    )

    np.save(
        PLOT_DIR / "marginalized_C_posterior_sd.npy",
        marg_C_sd,
    )

    np.save(
        PLOT_DIR / "marginalized_F_posterior_sd.npy",
        marg_F_sd,
    )

    np.save(
        PLOT_DIR
        / "delta_A_mean_marginalized_correlated_minus_IID.npy",
        delta_A_mean,
    )

    np.save(
        PLOT_DIR
        / "delta_C_mean_marginalized_correlated_minus_IID.npy",
        delta_C_mean,
    )

    np.save(
        PLOT_DIR
        / "delta_F_mean_marginalized_correlated_minus_IID.npy",
        delta_F_mean,
    )

    np.save(
        PLOT_DIR
        / "delta_A_sd_marginalized_correlated_minus_IID.npy",
        delta_A_sd,
    )

    np.save(
        PLOT_DIR
        / "delta_C_sd_marginalized_correlated_minus_IID.npy",
        delta_C_sd,
    )

    np.save(
        PLOT_DIR
        / "delta_F_sd_marginalized_correlated_minus_IID.npy",
        delta_F_sd,
    )

    # =========================================================================
    # SUMMARY BY PARAMETER
    # =========================================================================

    comparison_rows = []

    for (
        name,
        iid_m,
        corr_m,
        iid_s,
        corr_s,
    ) in [

        (
            "A",
            iid_A_mean,
            marg_A_mean,
            iid_A_sd,
            marg_A_sd,
        ),

        (
            "C",
            iid_C_mean,
            marg_C_mean,
            iid_C_sd,
            marg_C_sd,
        ),

        (
            "F",
            iid_F_mean,
            marg_F_mean,
            iid_F_sd,
            marg_F_sd,
        ),
    ]:

        dm = (
            corr_m
            - iid_m
        )

        ds = (
            corr_s
            - iid_s
        )

        comparison_rows.append(
            {
                "parameter":
                    name,

                "iid_mean_posterior_sd":
                    float(
                        np.mean(iid_s)
                    ),

                "correlated_mean_posterior_sd":
                    float(
                        np.mean(corr_s)
                    ),

                "mean_abs_posterior_mean_change":
                    float(
                        np.mean(
                            np.abs(dm)
                        )
                    ),

                "max_abs_posterior_mean_change":
                    float(
                        np.max(
                            np.abs(dm)
                        )
                    ),

                "mean_posterior_sd_change":
                    float(
                        np.mean(ds)
                    ),

                "mean_abs_posterior_sd_change":
                    float(
                        np.mean(
                            np.abs(ds)
                        )
                    ),

                "max_abs_posterior_sd_change":
                    float(
                        np.max(
                            np.abs(ds)
                        )
                    ),

                "fraction_cells_sd_increased":
                    float(
                        np.mean(
                            ds > 0
                        )
                    ),

                "fraction_cells_sd_decreased":
                    float(
                        np.mean(
                            ds < 0
                        )
                    ),
            }
        )

    comparison_summary = pd.DataFrame(
        comparison_rows
    )

    comparison_summary.to_csv(
        PLOT_DIR
        / "IID_vs_marginalized_correlated_by_parameter.csv",
        index=False,
    )

    # =========================================================================
    # PER-CELL TABLE
    # =========================================================================

    comparison_per_cell = pd.DataFrame(
        {
            "cell_index":
                x,

            "radial_shell":
                shell_number,

            "radial_shell_centre_km":
                shell_radius_per_cell,

            "A_mean_IID":
                iid_A_mean,

            "A_mean_correlated":
                marg_A_mean,

            "A_mean_change":
                delta_A_mean,

            "C_mean_IID":
                iid_C_mean,

            "C_mean_correlated":
                marg_C_mean,

            "C_mean_change":
                delta_C_mean,

            "F_mean_IID":
                iid_F_mean,

            "F_mean_correlated":
                marg_F_mean,

            "F_mean_change":
                delta_F_mean,

            "A_sd_IID":
                iid_A_sd,

            "A_sd_correlated":
                marg_A_sd,

            "A_sd_change":
                delta_A_sd,

            "C_sd_IID":
                iid_C_sd,

            "C_sd_correlated":
                marg_C_sd,

            "C_sd_change":
                delta_C_sd,

            "F_sd_IID":
                iid_F_sd,

            "F_sd_correlated":
                marg_F_sd,

            "F_sd_change":
                delta_F_sd,
        }
    )

    comparison_per_cell.to_csv(
        PLOT_DIR
        / "IID_vs_marginalized_correlated_by_cell.csv",
        index=False,
    )

    # =========================================================================
    # OVERALL SUMMARY
    # =========================================================================

    all_delta_mean = (
        marg_mean
        - iid_mean
    )

    all_delta_sd = (
        marg_sd
        - iid_sd
    )

    overall_summary = {

        "mean_absolute_change_in_posterior_mean":
            float(
                np.mean(
                    np.abs(
                        all_delta_mean
                    )
                )
            ),

        "maximum_absolute_change_in_posterior_mean":
            float(
                np.max(
                    np.abs(
                        all_delta_mean
                    )
                )
            ),

        "mean_change_in_posterior_sd":
            float(
                np.mean(
                    all_delta_sd
                )
            ),

        "mean_absolute_change_in_posterior_sd":
            float(
                np.mean(
                    np.abs(
                        all_delta_sd
                    )
                )
            ),

        "maximum_absolute_change_in_posterior_sd":
            float(
                np.max(
                    np.abs(
                        all_delta_sd
                    )
                )
            ),
    }

    pd.DataFrame(
        [overall_summary]
    ).to_csv(
        PLOT_DIR
        / "overall_error_model_sensitivity_summary.csv",
        index=False,
    )

    print()
    print("=" * 72)
    print("IID VS MARGINALIZED CORRELATED POSTERIOR")
    print("=" * 72)
    print()

    print(
        comparison_summary.to_string(
            index=False
        )
    )

    print()
    print(
        f"Overall across all "
        f"{EXPECTED_N_PARAMETERS} parameters:"
    )

    for key, value in overall_summary.items():

        print(
            f"{key} = {value:.8f}"
        )

    # =========================================================================
    # SEPARATE ERROR-BAR FIGURES
    # =========================================================================

    # A -----------------------------------------------------------------------

    plot_single_posterior_errorbars(
        x,
        tick_positions,
        tick_labels,
        iid_A_mean,
        iid_A_sd,
        "A",
        "block-IID",
        "A_block_IID_errorbars_improved.png",
        color="tab:blue",
    )

    plot_single_posterior_errorbars(
        x,
        tick_positions,
        tick_labels,
        marg_A_mean,
        marg_A_sd,
        "A",
        "correlated, marginalized",
        "A_correlated_marginalized_errorbars_improved.png",
        color="tab:orange",
    )

    # C -----------------------------------------------------------------------

    plot_single_posterior_errorbars(
        x,
        tick_positions,
        tick_labels,
        iid_C_mean,
        iid_C_sd,
        "C",
        "block-IID",
        "C_block_IID_errorbars_improved.png",
        color="tab:blue",
    )

    plot_single_posterior_errorbars(
        x,
        tick_positions,
        tick_labels,
        marg_C_mean,
        marg_C_sd,
        "C",
        "correlated, marginalized",
        "C_correlated_marginalized_errorbars_improved.png",
        color="tab:orange",
    )

    # F -----------------------------------------------------------------------

    plot_single_posterior_errorbars(
        x,
        tick_positions,
        tick_labels,
        iid_F_mean,
        iid_F_sd,
        "F",
        "block-IID",
        "F_block_IID_errorbars_improved.png",
        color="tab:blue",
    )

    plot_single_posterior_errorbars(
        x,
        tick_positions,
        tick_labels,
        marg_F_mean,
        marg_F_sd,
        "F",
        "correlated, marginalized",
        "F_correlated_marginalized_errorbars_improved.png",
        color="tab:orange",
    )

    # =========================================================================
    # INDIVIDUAL DIFFERENCE FIGURES
    # =========================================================================

    plot_mean_difference(
        x,
        tick_positions,
        tick_labels,
        delta_A_mean,
        "A",
    )

    plot_mean_difference(
        x,
        tick_positions,
        tick_labels,
        delta_C_mean,
        "C",
    )

    plot_mean_difference(
        x,
        tick_positions,
        tick_labels,
        delta_F_mean,
        "F",
    )

    plot_sd_difference(
        x,
        tick_positions,
        tick_labels,
        delta_A_sd,
        "A",
    )

    plot_sd_difference(
        x,
        tick_positions,
        tick_labels,
        delta_C_sd,
        "C",
    )

    plot_sd_difference(
        x,
        tick_positions,
        tick_labels,
        delta_F_sd,
        "F",
    )

    # =========================================================================
    # COMBINED SENSITIVITY FIGURES
    # =========================================================================

    plot_combined_mean_difference(
        x,
        tick_positions,
        tick_labels,
        delta_A_mean,
        delta_C_mean,
        delta_F_mean,
    )

    plot_combined_sd_difference(
        x,
        tick_positions,
        tick_labels,
        delta_A_sd,
        delta_C_sd,
        delta_F_sd,
    )

    # =========================================================================
    # FINISH
    # =========================================================================

    print()
    print("=" * 72)
    print("FINISHED")
    print("=" * 72)
    print()

    print(
        "Improved figures and summaries saved in:"
    )

    print(
        PLOT_DIR
    )


if __name__ == "__main__":
    main()