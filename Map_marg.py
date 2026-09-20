"""
Final spatial plotting script for the full hyperparameter-marginalized
correlated-error Earth posterior.

Final analysis:
    Mesh: 10 radial x 5 lateral
    Cells: 450
    Parameters: 1350
    Earth prior variance: 0.01
    Hyperparameter marginalization: full 72-point discrete grid

Figures produced:
    1. Equatorial cross-sections of posterior mean and SD
    2. Meridional cross-sections of posterior mean and SD
    3. Fixed-radius maps for delta A/A
    4. Fixed-radius maps for delta C/C
    5. Fixed-radius maps for delta F/F

All fractional perturbations are displayed in percent.
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Polygon, Wedge, Circle

from raytracer import SphericalMesh


# ============================================================
# OPTIONAL CARTOPY IMPORT
# ============================================================

try:
    import cartopy.crs as ccrs

    HAS_CARTOPY = True

except ImportError:
    HAS_CARTOPY = False


# ============================================================
# SETTINGS
# ============================================================

OUTPUT_DIR = Path(
    "outputs/"
    "earth_posterior_marginalized_r10_l5_prior_0p01_full72"
)

FIGURE_DIR = (
    OUTPUT_DIR
    / "earth_spatial_figures_marginalized"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


INNER_CORE_RADIUS_KM = 1221.5

RADIAL_RESOLUTION = 10
LATERAL_RESOLUTION = 5


# Meridional plane:
# +90 degrees on one side,
# +270 degrees on the opposite side.

PHI_PLANE_DEG = 90.0


# ============================================================
# VISUAL SETTINGS
# ============================================================

# Hide internal mesh boundaries.
#
# IMPORTANT:
# The inferred values remain piecewise constant within cells.
# This option changes only the visual appearance and does not
# smooth the inferred Earth model.

DRAW_INTERNAL_BOUNDARIES = False


# Posterior mean:
# blue  = negative perturbation
# white = approximately zero
# red   = positive perturbation

MEAN_CMAP_NAME = "RdBu_r"


# Posterior SD:
# light = lower uncertainty
# dark  = higher uncertainty

SD_CMAP_NAME = "YlGnBu"


# ============================================================
# LOAD FULL MARGINALIZED POSTERIOR
# ============================================================

posterior_mean_path = (
    OUTPUT_DIR
    / "earth_posterior_mean_marginalized.npy"
)

posterior_sd_path = (
    OUTPUT_DIR
    / "earth_posterior_sd_marginalized.npy"
)


if not posterior_mean_path.exists():
    raise FileNotFoundError(
        f"Posterior mean file not found:\n"
        f"{posterior_mean_path}"
    )


if not posterior_sd_path.exists():
    raise FileNotFoundError(
        f"Posterior SD file not found:\n"
        f"{posterior_sd_path}"
    )


posterior_mean = np.load(
    posterior_mean_path
)

posterior_sd = np.load(
    posterior_sd_path
)


# ============================================================
# VALIDATE POSTERIOR ARRAYS
# ============================================================

if posterior_mean.ndim != 1:
    raise ValueError(
        "Posterior mean must be a one-dimensional array."
    )


if posterior_sd.ndim != 1:
    raise ValueError(
        "Posterior SD must be a one-dimensional array."
    )


if posterior_mean.size % 3 != 0:
    raise ValueError(
        "Posterior mean length must be divisible by 3."
    )


if posterior_sd.size != posterior_mean.size:
    raise ValueError(
        "Posterior mean and posterior SD must have "
        "the same length."
    )


if not np.all(np.isfinite(posterior_mean)):
    raise ValueError(
        "Posterior mean contains non-finite values."
    )


if not np.all(np.isfinite(posterior_sd)):
    raise ValueError(
        "Posterior SD contains non-finite values."
    )


if np.any(posterior_sd < 0.0):
    raise ValueError(
        "Posterior SD contains negative values."
    )


n_cells = (
    posterior_mean.size
    // 3
)


print()
print("=" * 70)
print("FULL HYPERPARAMETER-MARGINALIZED EARTH POSTERIOR")
print("=" * 70)

print(
    "Number of spatial cells =",
    n_cells,
)

print(
    "Number of parameters    =",
    posterior_mean.size,
)

print(
    "Posterior mean range    =",
    f"{posterior_mean.min():.8f}",
    "to",
    f"{posterior_mean.max():.8f}",
)

print(
    "Posterior SD range      =",
    f"{posterior_sd.min():.8f}",
    "to",
    f"{posterior_sd.max():.8f}",
)


# ============================================================
# PARAMETER ORDER
#
# [A1, C1, F1,
#  A2, C2, F2,
#  ...]
#
# Values are fractional perturbations.
# Multiply by 100 to display them as percentages.
# ============================================================

A_mean = (
    posterior_mean[0::3]
    * 100.0
)

C_mean = (
    posterior_mean[1::3]
    * 100.0
)

F_mean = (
    posterior_mean[2::3]
    * 100.0
)


A_sd = (
    posterior_sd[0::3]
    * 100.0
)

C_sd = (
    posterior_sd[1::3]
    * 100.0
)

F_sd = (
    posterior_sd[2::3]
    * 100.0
)


mean_fields = [
    A_mean,
    C_mean,
    F_mean,
]


sd_fields = [
    A_sd,
    C_sd,
    F_sd,
]


parameter_titles = [
    r"$\delta A/A$",
    r"$\delta C/C$",
    r"$\delta F/F$",
]


# ============================================================
# RECREATE FINAL 10 x 5 MESH
# ============================================================

mesh = SphericalMesh(
    INNER_CORE_RADIUS_KM,
    RADIAL_RESOLUTION,
    LATERAL_RESOLUTION,
)


radial_edges = np.asarray(
    mesh.radial_edges,
    dtype=float,
)


theta_edges = np.asarray(
    mesh.theta_edges,
    dtype=float,
)


phi_centres = np.asarray(
    mesh.phi_centres,
    dtype=float,
)


labels = list(
    mesh.labels
)


# ============================================================
# VALIDATE MESH AGAINST POSTERIOR
# ============================================================

if len(labels) != n_cells:
    raise ValueError(
        f"Mesh contains {len(labels)} cells, "
        f"but posterior contains {n_cells} cells."
    )


expected_parameters = (
    3
    * mesh.n_cells
)


if posterior_mean.size != expected_parameters:
    raise ValueError(
        f"Expected {expected_parameters} parameters "
        f"for the {RADIAL_RESOLUTION} x "
        f"{LATERAL_RESOLUTION} mesh, but found "
        f"{posterior_mean.size}."
    )


print()
print("=" * 70)
print("MESH")
print("=" * 70)

print(
    "n_radial =",
    mesh.n_radial,
)

print(
    "n_lat    =",
    mesh.n_lat,
)

print(
    "n_lon    =",
    mesh.n_lon,
)

print(
    "n_cells  =",
    mesh.n_cells,
)


print(
    "radial edges =",
    radial_edges,
)


print(
    "theta edges (deg) =",
    np.degrees(
        theta_edges
    ),
)


print(
    "phi centres (deg) =",
    np.degrees(
        phi_centres
    ),
)


# ============================================================
# PARSE CELL LABELS
# ============================================================

pattern = re.compile(
    r"r(\d+)_lat(\d+)_lon(\d+)"
)


cell_lookup = {}


for cell_index, label in enumerate(
    labels
):

    match = pattern.fullmatch(
        label
    )

    if match is None:
        raise ValueError(
            f"Could not parse mesh label: {label}"
        )


    radial_index = int(
        match.group(1)
    )

    lat_index = int(
        match.group(2)
    )

    lon_index = int(
        match.group(3)
    )


    cell_lookup[
        (
            radial_index,
            lat_index,
            lon_index,
        )
    ] = cell_index


# ============================================================
# COMMON COLOR SCALES
# ============================================================

# Use one common symmetric scale for the posterior means
# of delta A/A, delta C/C, and delta F/F.

mean_limit = max(
    np.max(
        np.abs(A_mean)
    ),
    np.max(
        np.abs(C_mean)
    ),
    np.max(
        np.abs(F_mean)
    ),
)


mean_norm = TwoSlopeNorm(
    vmin=-mean_limit,
    vcenter=0.0,
    vmax=mean_limit,
)


# Use one common scale for all posterior SD fields.

sd_limit = max(
    np.max(A_sd),
    np.max(C_sd),
    np.max(F_sd),
)


sd_norm = Normalize(
    vmin=0.0,
    vmax=sd_limit,
)


mean_cmap = plt.get_cmap(
    MEAN_CMAP_NAME
)


sd_cmap = plt.get_cmap(
    SD_CMAP_NAME
)


print()
print(
    "Mean scale:",
    -mean_limit,
    "to",
    mean_limit,
    "%",
)

print(
    "SD scale:",
    0.0,
    "to",
    sd_limit,
    "%",
)


# ============================================================
# INTERNAL BOUNDARY STYLE
# ============================================================

if DRAW_INTERNAL_BOUNDARIES:

    CELL_EDGE_COLOR = "0.75"
    CELL_EDGE_WIDTH = 0.25

else:

    CELL_EDGE_COLOR = "none"
    CELL_EDGE_WIDTH = 0.0


# ============================================================
# CLEAN AXIS STYLE
# ============================================================

def clean_circle_axis(
    ax,
):

    ax.set_aspect(
        "equal"
    )

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlabel("")
    ax.set_ylabel("")


    for spine in ax.spines.values():

        spine.set_visible(
            False
        )


# ============================================================
# OUTER INNER-CORE BOUNDARY
# ============================================================

def add_outer_boundary(
    ax,
):

    boundary = Circle(
        (0.0, 0.0),
        INNER_CORE_RADIUS_KM,
        fill=False,
        edgecolor="black",
        linewidth=1.1,
        zorder=20,
    )

    ax.add_patch(
        boundary
    )


# ============================================================
# EQUATORIAL CROSS-SECTION
# ============================================================

theta_equator = (
    np.pi
    / 2.0
)


equatorial_lat_index = None


for lat_index in range(
    mesh.n_lat
):

    theta_lower = (
        theta_edges[
            lat_index
        ]
    )

    theta_upper = (
        theta_edges[
            lat_index + 1
        ]
    )


    if (
        theta_lower
        <= theta_equator
        <= theta_upper
    ):

        equatorial_lat_index = (
            lat_index
        )

        break


if equatorial_lat_index is None:
    raise RuntimeError(
        "Could not find equatorial latitude band."
    )


print()
print(
    "Equatorial latitude band =",
    equatorial_lat_index,
)


# ============================================================
# DRAW EQUATORIAL FIELD
# ============================================================

def draw_equatorial_field(
    ax,
    values,
    norm,
    cmap,
):

    phi_offset = float(
        mesh.phi_offset
    )


    for radial_index in range(
        mesh.n_radial
    ):

        r_inner = (
            radial_edges[
                radial_index
            ]
        )

        r_outer = (
            radial_edges[
                radial_index + 1
            ]
        )


        shell_width = (
            r_outer
            - r_inner
        )


        for lon_index in range(
            mesh.n_lon
        ):

            cell_index = (
                cell_lookup[
                    (
                        radial_index,
                        equatorial_lat_index,
                        lon_index,
                    )
                ]
            )


            value = (
                values[
                    cell_index
                ]
            )


            phi_centre = (
                phi_centres[
                    lon_index
                ]
            )


            phi_lower = (
                phi_centre
                - phi_offset
            )

            phi_upper = (
                phi_centre
                + phi_offset
            )


            patch = Wedge(
                center=(
                    0.0,
                    0.0,
                ),
                r=r_outer,
                theta1=np.degrees(
                    phi_lower
                ),
                theta2=np.degrees(
                    phi_upper
                ),
                width=shell_width,
                facecolor=cmap(
                    norm(
                        value
                    )
                ),
                edgecolor=CELL_EDGE_COLOR,
                linewidth=CELL_EDGE_WIDTH,
                antialiased=True,
                zorder=2,
            )


            ax.add_patch(
                patch
            )


    add_outer_boundary(
        ax
    )


    limit = (
        INNER_CORE_RADIUS_KM
        * 1.15
    )


    ax.set_xlim(
        -limit,
        limit,
    )

    ax.set_ylim(
        -limit,
        limit,
    )


    clean_circle_axis(
        ax
    )


# ============================================================
# EQUATORIAL LABELS
# ============================================================

def add_equatorial_labels(
    ax,
):

    R = (
        INNER_CORE_RADIUS_KM
    )


    r_label = (
        R
        * 1.065
    )


    ax.text(
        0.0,
        r_label,
        r"$90^\circ$E",
        ha="center",
        va="bottom",
        fontsize=9,
    )


    ax.text(
        0.0,
        -r_label,
        r"$90^\circ$W",
        ha="center",
        va="top",
        fontsize=9,
    )


    ax.text(
        -r_label,
        0.0,
        r"$180^\circ$",
        ha="right",
        va="center",
        fontsize=9,
    )


    ax.text(
        r_label,
        0.0,
        r"$0^\circ$",
        ha="left",
        va="center",
        fontsize=9,
    )


# ============================================================
# CREATE EQUATORIAL FIGURE
# ============================================================

def make_equatorial_figure():

    fig = plt.figure(
        figsize=(10.5, 9.0)
    )


    gs = fig.add_gridspec(
        nrows=4,
        ncols=3,
        height_ratios=[
            1.0,
            0.045,
            1.0,
            0.045,
        ],
        hspace=0.52,
        wspace=0.18,
        left=0.04,
        right=0.96,
        bottom=0.06,
        top=0.90,
    )


    axes_mean = []
    axes_sd = []


    # --------------------------------------------------------
    # CREATE AXES
    # --------------------------------------------------------

    for column in range(3):

        axes_mean.append(
            fig.add_subplot(
                gs[0, column]
            )
        )

        axes_sd.append(
            fig.add_subplot(
                gs[2, column]
            )
        )


    # --------------------------------------------------------
    # POSTERIOR MEAN ROW
    # --------------------------------------------------------

    for column in range(3):

        draw_equatorial_field(
            ax=axes_mean[column],
            values=mean_fields[column],
            norm=mean_norm,
            cmap=mean_cmap,
        )

        add_equatorial_labels(
            axes_mean[column]
        )

        axes_mean[column].set_title(
            "Mean "
            + parameter_titles[column],
            fontsize=12,
            pad=10,
        )


    # --------------------------------------------------------
    # POSTERIOR MEAN COLORBAR
    # --------------------------------------------------------

    mean_scalar = ScalarMappable(
        norm=mean_norm,
        cmap=mean_cmap,
    )

    mean_scalar.set_array([])


    cax_mean = fig.add_subplot(
        gs[1, :]
    )


    cbar_mean = fig.colorbar(
        mean_scalar,
        cax=cax_mean,
        orientation="horizontal",
    )


    cbar_mean.set_label(
        "Posterior mean (%)",
        labelpad=2,
    )


    # --------------------------------------------------------
    # POSTERIOR SD ROW
    # --------------------------------------------------------

    for column in range(3):

        draw_equatorial_field(
            ax=axes_sd[column],
            values=sd_fields[column],
            norm=sd_norm,
            cmap=sd_cmap,
        )

        add_equatorial_labels(
            axes_sd[column]
        )

        axes_sd[column].set_title(
            parameter_titles[column]
            + " SD",
            fontsize=12,
            pad=10,
        )


    # --------------------------------------------------------
    # POSTERIOR SD COLORBAR
    # --------------------------------------------------------

    sd_scalar = ScalarMappable(
        norm=sd_norm,
        cmap=sd_cmap,
    )

    sd_scalar.set_array([])


    cax_sd = fig.add_subplot(
        gs[3, :]
    )


    cbar_sd = fig.colorbar(
        sd_scalar,
        cax=cax_sd,
        orientation="horizontal",
    )


    cbar_sd.set_label(
        "Posterior SD (%)",
        labelpad=4,
    )


    fig.suptitle(
        "Equatorial plane cross-sections",
        fontsize=15,
        fontweight="bold",
        y=0.97,
    )


    return (
        fig,
        (
            axes_mean,
            axes_sd,
        ),
    )


# ============================================================
# MERIDIONAL CROSS-SECTION
# ============================================================

phi_positive = np.radians(
    PHI_PLANE_DEG
)


phi_negative = (
    phi_positive
    + np.pi
) % (
    2.0
    * np.pi
)


# ============================================================
# ANGULAR DISTANCE
# ============================================================

def angular_difference(
    angle1,
    angle2,
):

    return np.abs(
        np.angle(
            np.exp(
                1j
                * (
                    angle1
                    - angle2
                )
            )
        )
    )


# ============================================================
# FIND LONGITUDE SECTOR
# ============================================================

def find_lon_index(
    phi,
):

    differences = np.asarray(
        [
            angular_difference(
                phi,
                centre,
            )
            for centre
            in phi_centres
        ]
    )


    return int(
        np.argmin(
            differences
        )
    )


lon_positive = find_lon_index(
    phi_positive
)


lon_negative = find_lon_index(
    phi_negative
)


print(
    "Meridional positive-side longitude sector =",
    lon_positive,
    "(",
    f"{np.degrees(phi_centres[lon_positive]):.1f} deg",
    ")",
)

print(
    "Meridional negative-side longitude sector =",
    lon_negative,
    "(",
    f"{np.degrees(phi_centres[lon_negative]):.1f} deg",
    ")",
)


# ============================================================
# MERIDIONAL POLYGON
# ============================================================

def make_meridional_polygon(
    r_inner,
    r_outer,
    theta_lower,
    theta_upper,
    horizontal_sign,
    n_arc_points=100,
):

    theta_outer = np.linspace(
        theta_lower,
        theta_upper,
        n_arc_points,
    )


    x_outer = (
        horizontal_sign
        * r_outer
        * np.sin(
            theta_outer
        )
    )


    z_outer = (
        r_outer
        * np.cos(
            theta_outer
        )
    )


    theta_inner = np.linspace(
        theta_upper,
        theta_lower,
        n_arc_points,
    )


    x_inner = (
        horizontal_sign
        * r_inner
        * np.sin(
            theta_inner
        )
    )


    z_inner = (
        r_inner
        * np.cos(
            theta_inner
        )
    )


    x = np.concatenate(
        [
            x_outer,
            x_inner,
        ]
    )


    z = np.concatenate(
        [
            z_outer,
            z_inner,
        ]
    )


    return np.column_stack(
        [
            x,
            z,
        ]
    )


# ============================================================
# DRAW MERIDIONAL FIELD
# ============================================================

def draw_meridional_field(
    ax,
    values,
    norm,
    cmap,
):

    for radial_index in range(
        mesh.n_radial
    ):

        r_inner = (
            radial_edges[
                radial_index
            ]
        )

        r_outer = (
            radial_edges[
                radial_index + 1
            ]
        )


        for lat_index in range(
            mesh.n_lat
        ):

            theta_lower = (
                theta_edges[
                    lat_index
                ]
            )

            theta_upper = (
                theta_edges[
                    lat_index + 1
                ]
            )


            # ------------------------------------------------
            # POSITIVE SIDE
            # ------------------------------------------------

            positive_cell_index = (
                cell_lookup[
                    (
                        radial_index,
                        lat_index,
                        lon_positive,
                    )
                ]
            )


            positive_value = (
                values[
                    positive_cell_index
                ]
            )


            positive_vertices = (
                make_meridional_polygon(
                    r_inner=r_inner,
                    r_outer=r_outer,
                    theta_lower=theta_lower,
                    theta_upper=theta_upper,
                    horizontal_sign=1.0,
                )
            )


            positive_patch = Polygon(
                positive_vertices,
                closed=True,
                facecolor=cmap(
                    norm(
                        positive_value
                    )
                ),
                edgecolor=CELL_EDGE_COLOR,
                linewidth=CELL_EDGE_WIDTH,
                antialiased=True,
                zorder=2,
            )


            ax.add_patch(
                positive_patch
            )


            # ------------------------------------------------
            # NEGATIVE SIDE
            # ------------------------------------------------

            negative_cell_index = (
                cell_lookup[
                    (
                        radial_index,
                        lat_index,
                        lon_negative,
                    )
                ]
            )


            negative_value = (
                values[
                    negative_cell_index
                ]
            )


            negative_vertices = (
                make_meridional_polygon(
                    r_inner=r_inner,
                    r_outer=r_outer,
                    theta_lower=theta_lower,
                    theta_upper=theta_upper,
                    horizontal_sign=-1.0,
                )
            )


            negative_patch = Polygon(
                negative_vertices,
                closed=True,
                facecolor=cmap(
                    norm(
                        negative_value
                    )
                ),
                edgecolor=CELL_EDGE_COLOR,
                linewidth=CELL_EDGE_WIDTH,
                antialiased=True,
                zorder=2,
            )


            ax.add_patch(
                negative_patch
            )


    add_outer_boundary(
        ax
    )


    limit = (
        INNER_CORE_RADIUS_KM
        * 1.15
    )


    ax.set_xlim(
        -limit,
        limit,
    )

    ax.set_ylim(
        -limit,
        limit,
    )


    clean_circle_axis(
        ax
    )


# ============================================================
# MERIDIONAL LABELS
# ============================================================

def add_meridional_labels(
    ax,
):

    R = (
        INNER_CORE_RADIUS_KM
    )


    r_label = (
        R
        * 1.065
    )


    ax.text(
        0.0,
        r_label,
        "N",
        ha="center",
        va="bottom",
        fontsize=9,
    )


    ax.text(
        0.0,
        -r_label,
        "S",
        ha="center",
        va="top",
        fontsize=9,
    )


    ax.text(
        -r_label,
        0.0,
        r"$90^\circ$W",
        ha="right",
        va="center",
        fontsize=9,
    )


    ax.text(
        r_label,
        0.0,
        r"$90^\circ$E",
        ha="left",
        va="center",
        fontsize=9,
    )


# ============================================================
# CREATE MERIDIONAL FIGURE
# ============================================================

def make_meridional_figure():

    fig = plt.figure(
        figsize=(10.5, 9.0)
    )


    gs = fig.add_gridspec(
        nrows=4,
        ncols=3,
        height_ratios=[
            1.0,
            0.045,
            1.0,
            0.045,
        ],
        hspace=0.52,
        wspace=0.18,
        left=0.04,
        right=0.96,
        bottom=0.06,
        top=0.90,
    )


    axes_mean = []
    axes_sd = []


    for column in range(3):

        axes_mean.append(
            fig.add_subplot(
                gs[0, column]
            )
        )

        axes_sd.append(
            fig.add_subplot(
                gs[2, column]
            )
        )


    # --------------------------------------------------------
    # POSTERIOR MEAN ROW
    # --------------------------------------------------------

    for column in range(3):

        draw_meridional_field(
            ax=axes_mean[column],
            values=mean_fields[column],
            norm=mean_norm,
            cmap=mean_cmap,
        )

        add_meridional_labels(
            axes_mean[column]
        )

        axes_mean[column].set_title(
            "Mean "
            + parameter_titles[column],
            fontsize=12,
            pad=10,
        )


    # --------------------------------------------------------
    # POSTERIOR MEAN COLORBAR
    # --------------------------------------------------------

    mean_scalar = ScalarMappable(
        norm=mean_norm,
        cmap=mean_cmap,
    )

    mean_scalar.set_array([])


    cax_mean = fig.add_subplot(
        gs[1, :]
    )


    cbar_mean = fig.colorbar(
        mean_scalar,
        cax=cax_mean,
        orientation="horizontal",
    )


    cbar_mean.set_label(
        "Posterior mean (%)",
        labelpad=2,
    )


    # --------------------------------------------------------
    # POSTERIOR SD ROW
    # --------------------------------------------------------

    for column in range(3):

        draw_meridional_field(
            ax=axes_sd[column],
            values=sd_fields[column],
            norm=sd_norm,
            cmap=sd_cmap,
        )

        add_meridional_labels(
            axes_sd[column]
        )

        axes_sd[column].set_title(
            parameter_titles[column]
            + " SD",
            fontsize=12,
            pad=10,
        )


    # --------------------------------------------------------
    # POSTERIOR SD COLORBAR
    # --------------------------------------------------------

    sd_scalar = ScalarMappable(
        norm=sd_norm,
        cmap=sd_cmap,
    )

    sd_scalar.set_array([])


    cax_sd = fig.add_subplot(
        gs[3, :]
    )


    cbar_sd = fig.colorbar(
        sd_scalar,
        cax=cax_sd,
        orientation="horizontal",
    )


    cbar_sd.set_label(
        "Posterior SD (%)",
        labelpad=4,
    )


    fig.suptitle(
        "Meridional plane cross-sections",
        fontsize=15,
        fontweight="bold",
        y=0.97,
    )


    return (
        fig,
        (
            axes_mean,
            axes_sd,
        ),
    )


# ============================================================
# FIXED-RADIUS MAPS
# ============================================================

def find_radial_cell_for_radius(
    target_radius_km,
):

    radial_index = np.searchsorted(
        radial_edges,
        target_radius_km,
        side="right",
    ) - 1


    radial_index = int(
        np.clip(
            radial_index,
            0,
            mesh.n_radial - 1,
        )
    )


    return radial_index


# ============================================================
# BUILD LAT-LON FIELD
# ============================================================

def make_lat_lon_field(
    values,
    radial_index,
):

    # --------------------------------------------------------
    # LATITUDE EDGES
    #
    # latitude = 90 degrees - colatitude
    # --------------------------------------------------------

    lat_edges = (
        90.0
        - np.degrees(
            theta_edges
        )
    )


    # Reverse so latitude increases south -> north.

    lat_edges = (
        lat_edges[::-1]
    )


    # --------------------------------------------------------
    # LONGITUDE EDGES
    #
    # IMPORTANT:
    # Construct these automatically from SphericalMesh.
    #
    # This replaces the old hard-coded:
    #
    # [-60, 60, 180, 300]
    #
    # which was valid only for the old L=2 mesh.
    #
    # For the final L=5 mesh:
    #
    # centres:
    # 0, 40, 80, ..., 320 degrees
    #
    # edges:
    # -20, 20, 60, ..., 340 degrees
    # --------------------------------------------------------

    phi_centres_deg = np.degrees(
        phi_centres
    )


    phi_offset_deg = np.degrees(
        float(
            mesh.phi_offset
        )
    )


    lon_edges = np.concatenate(
        [
            phi_centres_deg
            - phi_offset_deg,
            [
                phi_centres_deg[-1]
                + phi_offset_deg
            ],
        ]
    )


    # --------------------------------------------------------
    # BUILD FIELD
    # --------------------------------------------------------

    field = np.zeros(
        (
            mesh.n_lat,
            mesh.n_lon,
        ),
        dtype=float,
    )


    for lat_index in range(
        mesh.n_lat
    ):

        for lon_index in range(
            mesh.n_lon
        ):

            cell_index = (
                cell_lookup[
                    (
                        radial_index,
                        lat_index,
                        lon_index,
                    )
                ]
            )


            field[
                lat_index,
                lon_index,
            ] = values[
                cell_index
            ]


    # Reverse latitude dimension to match lat_edges.

    field = field[
        ::-1,
        :
    ]


    # --------------------------------------------------------
    # VALIDATE PCOLORMESH DIMENSIONS
    # --------------------------------------------------------

    expected_lon_edges = (
        field.shape[1]
        + 1
    )

    expected_lat_edges = (
        field.shape[0]
        + 1
    )


    if lon_edges.size != expected_lon_edges:

        raise ValueError(
            "Longitude-edge mismatch: "
            f"field has {field.shape[1]} longitude cells "
            f"but lon_edges has {lon_edges.size} values. "
            f"Expected {expected_lon_edges}."
        )


    if lat_edges.size != expected_lat_edges:

        raise ValueError(
            "Latitude-edge mismatch: "
            f"field has {field.shape[0]} latitude cells "
            f"but lat_edges has {lat_edges.size} values. "
            f"Expected {expected_lat_edges}."
        )


    return (
        lon_edges,
        lat_edges,
        field,
    )


# ============================================================
# DRAW FIXED-RADIUS MAP
# ============================================================

def draw_fixed_radius_map(
    ax,
    values,
    radial_index,
    norm,
    cmap,
):

    (
        lon_edges,
        lat_edges,
        field,
    ) = make_lat_lon_field(
        values=values,
        radial_index=radial_index,
    )


    mesh_artist = ax.pcolormesh(
        lon_edges,
        lat_edges,
        field,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        shading="flat",
        edgecolors="none",
        linewidth=0.0,
        antialiased=False,
        rasterized=True,
        snap=True,
    )


    # Minimize thin projection seams between neighboring cells.

    try:
        mesh_artist.set_edgecolor(
            "face"
        )

    except Exception:
        pass


    ax.coastlines(
        resolution="110m",
        linewidth=0.65,
        color="black",
    )


    ax.set_global()


# ============================================================
# RADII TO DISPLAY
# ============================================================

target_radii = [
    INNER_CORE_RADIUS_KM - 1.0,
    800.0,
    400.0,
]


radius_titles = [
    "ICB",
    "800-km radius",
    "400-km radius",
]


radial_indices = [
    find_radial_cell_for_radius(
        radius
    )
    for radius
    in target_radii
]


print()
print("=" * 70)
print("FIXED-RADIUS LAYERS")
print("=" * 70)


for target_radius, radial_index in zip(
    target_radii,
    radial_indices,
):

    print(
        f"{target_radius:.1f} km "
        f"-> radial cell {radial_index}: "
        f"{radial_edges[radial_index]:.1f}"
        f"–"
        f"{radial_edges[radial_index + 1]:.1f} km"
    )


# ============================================================
# CREATE FIXED-RADIUS MAPS FOR ONE PARAMETER
# ============================================================

def make_fixed_radius_figure(
    values_mean,
    values_sd,
    parameter_title,
    filename_stub,
):

    if not HAS_CARTOPY:

        print()
        print(
            "Cartopy is not installed."
        )

        print(
            "Skipping geographic fixed-radius maps."
        )

        return None


    fig = plt.figure(
        figsize=(12.0, 8.0)
    )


    gs = fig.add_gridspec(
        nrows=4,
        ncols=3,
        height_ratios=[
            1.0,
            0.055,
            1.0,
            0.055,
        ],
        hspace=0.28,
        wspace=0.06,
        left=0.03,
        right=0.97,
        bottom=0.06,
        top=0.90,
    )


    axes_mean = []
    axes_sd = []


    # --------------------------------------------------------
    # CREATE MAP AXES
    # --------------------------------------------------------

    for column in range(3):

        axes_mean.append(
            fig.add_subplot(
                gs[0, column],
                projection=ccrs.Robinson(
                    central_longitude=0.0
                ),
            )
        )


        axes_sd.append(
            fig.add_subplot(
                gs[2, column],
                projection=ccrs.Robinson(
                    central_longitude=0.0
                ),
            )
        )


    # --------------------------------------------------------
    # POSTERIOR MEAN MAPS
    # --------------------------------------------------------

    for column in range(3):

        draw_fixed_radius_map(
            ax=axes_mean[column],
            values=values_mean,
            radial_index=radial_indices[column],
            norm=mean_norm,
            cmap=mean_cmap,
        )


        axes_mean[column].set_title(
            radius_titles[column],
            fontsize=12,
            pad=8,
        )


    # --------------------------------------------------------
    # POSTERIOR MEAN COLORBAR
    # --------------------------------------------------------

    mean_scalar = ScalarMappable(
        norm=mean_norm,
        cmap=mean_cmap,
    )

    mean_scalar.set_array([])


    cax_mean = fig.add_subplot(
        gs[1, :]
    )


    cbar_mean = fig.colorbar(
        mean_scalar,
        cax=cax_mean,
        orientation="horizontal",
    )


    cbar_mean.set_label(
        "Posterior mean of "
        + parameter_title
        + " (%)",
        labelpad=4,
    )


    # --------------------------------------------------------
    # POSTERIOR SD MAPS
    # --------------------------------------------------------

    for column in range(3):

        draw_fixed_radius_map(
            ax=axes_sd[column],
            values=values_sd,
            radial_index=radial_indices[column],
            norm=sd_norm,
            cmap=sd_cmap,
        )


    # --------------------------------------------------------
    # POSTERIOR SD COLORBAR
    # --------------------------------------------------------

    sd_scalar = ScalarMappable(
        norm=sd_norm,
        cmap=sd_cmap,
    )

    sd_scalar.set_array([])


    cax_sd = fig.add_subplot(
        gs[3, :]
    )


    cbar_sd = fig.colorbar(
        sd_scalar,
        cax=cax_sd,
        orientation="horizontal",
    )


    cbar_sd.set_label(
        "Posterior SD of "
        + parameter_title
        + " (%)",
        labelpad=4,
    )


    fig.suptitle(
        "Fixed-radius maps of "
        + parameter_title,
        fontsize=15,
        fontweight="bold",
        y=0.97,
    )


    output_path = (
        FIGURE_DIR
        / (
            filename_stub
            + "_fixed_radius_maps.png"
        )
    )


    fig.savefig(
        output_path,
        dpi=400,
        bbox_inches="tight",
        facecolor="white",
    )


    print(
        "Saved:",
        output_path,
    )


    return (
        fig,
        (
            axes_mean,
            axes_sd,
        ),
    )


# ============================================================
# GENERATE EQUATORIAL FIGURE
# ============================================================

print()
print("=" * 70)
print("CREATING EQUATORIAL FIGURE")
print("=" * 70)


fig_eq, axes_eq = (
    make_equatorial_figure()
)


equatorial_path = (
    FIGURE_DIR
    / "correlated_ACF_equatorial_clean.png"
)


fig_eq.savefig(
    equatorial_path,
    dpi=400,
    bbox_inches="tight",
    facecolor="white",
)


print(
    "Saved:",
    equatorial_path,
)


# ============================================================
# GENERATE MERIDIONAL FIGURE
# ============================================================

print()
print("=" * 70)
print("CREATING MERIDIONAL FIGURE")
print("=" * 70)


fig_mer, axes_mer = (
    make_meridional_figure()
)


meridional_path = (
    FIGURE_DIR
    / "correlated_ACF_meridional_clean.png"
)


fig_mer.savefig(
    meridional_path,
    dpi=400,
    bbox_inches="tight",
    facecolor="white",
)


print(
    "Saved:",
    meridional_path,
)


# ============================================================
# GENERATE FIXED-RADIUS MAPS
# ============================================================

if HAS_CARTOPY:

    print()
    print("=" * 70)
    print("CREATING FIXED-RADIUS MAPS")
    print("=" * 70)


    make_fixed_radius_figure(
        values_mean=A_mean,
        values_sd=A_sd,
        parameter_title=r"$\delta A/A$",
        filename_stub="A",
    )


    make_fixed_radius_figure(
        values_mean=C_mean,
        values_sd=C_sd,
        parameter_title=r"$\delta C/C$",
        filename_stub="C",
    )


    make_fixed_radius_figure(
        values_mean=F_mean,
        values_sd=F_sd,
        parameter_title=r"$\delta F/F$",
        filename_stub="F",
    )


else:

    print()
    print("=" * 70)
    print("CARTOPY NOT FOUND")
    print("=" * 70)

    print(
        "Equatorial and meridional figures "
        "were still created."
    )

    print()

    print(
        "To create the geographic maps, run:"
    )

    print(
        "uv add cartopy"
    )


# ============================================================
# FINISH
# ============================================================

print()
print("=" * 70)
print("DONE")
print("=" * 70)


print()
print(
    "Equatorial:"
)

print(
    equatorial_path
)


print()
print(
    "Meridional:"
)

print(
    meridional_path
)


print()
print(
    "All figures saved in:"
)

print(
    FIGURE_DIR
)


plt.show()