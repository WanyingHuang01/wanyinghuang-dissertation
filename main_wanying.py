# """Solve the IC anisotropy problem modelled as a purely linear Gaussian system."""

# import logging

# import numpy as np
# import pandas as pd
# from linear_gaussian import (
#     GaussianComponent,
#     calc_log_evidence,
#     calc_posterior_cov,
#     calc_posterior_mean,
# )
# from raytracer import SphericalMesh
# from tti.elastic.voigt import (
#     gradient_C_wrt_A,
#     gradient_C_wrt_C,
#     gradient_C_wrt_F,
#     gradient_C_wrt_L,
#     gradient_C_wrt_N,
# )
# from tti.traveltimes.parametrisations import LinearParametriser
# from tti.traveltimes.traveltimes import (
#     calculate_path_direction_vector,
#     calculate_relative_traveltime_voigt,
# )


# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s %(levelname)s %(name)s: %(message)s",
# )
# logger = logging.getLogger(__name__)


# class NoAnglesParametriser(LinearParametriser):
#     """Only P-wave Love parameters A, C, and F."""

#     n_model_params_per_segment = 3

#     transformation = np.array(
#         [
#             [1, 0, 0],
#             [0, 1, 0],
#             [0, 0, 1],
#             [0, 0, 0],
#             [0, 0, 0],
#             [0, 0, 0],
#             [0, 0, 0],
#         ],
#         dtype=float,
#     )


# def lonlatrad_to_xyz(
#     lonlatrad: np.ndarray,
# ) -> np.ndarray:
#     """Convert longitude, latitude, and radius to Cartesian coordinates.

#     Parameters
#     ----------
#     lonlatrad
#         Array with shape (..., 3), containing longitude in degrees,
#         latitude in degrees, and radius in kilometres.

#     Returns
#     -------
#     np.ndarray
#         Cartesian coordinates with shape (..., 3), in kilometres.
#     """

#     lon = np.radians(lonlatrad[..., 0])
#     lat = np.radians(lonlatrad[..., 1])
#     radius = lonlatrad[..., 2]

#     x = radius * np.cos(lat) * np.cos(lon)
#     y = radius * np.cos(lat) * np.sin(lon)
#     z = radius * np.sin(lat)

#     return np.stack(
#         [x, y, z],
#         axis=-1,
#     )


# def determine_weights(
#     mesh: SphericalMesh,
#     ic_in: np.ndarray,
#     path_directions: np.ndarray,
# ) -> np.ndarray:
#     """Determine the fraction of each path travelled through each mesh region."""

#     segment_distances = mesh.ray_distances_per_region(
#         lonlatrad_to_xyz(ic_in),
#         path_directions,
#     )

#     total_distances = segment_distances.sum(
#         axis=1,
#     )

#     weights = (
#         segment_distances
#         / total_distances[:, None]
#     )

#     # Shape:
#     # (1, number of segments, number of paths)
#     weights_for_calculator = weights.T[None, ...]

#     logger.debug(
#         "determine_weights: segment_distances.shape=%s "
#         "total_distances.shape=%s",
#         segment_distances.shape,
#         total_distances.shape,
#     )

#     logger.debug(
#         "determine_weights: computed weights for %d paths "
#         "and %d segments",
#         ic_in.shape[0],
#         segment_distances.shape[1],
#     )

#     return weights_for_calculator


# def construct_forward_map(
#     path_directions: np.ndarray,
#     weights: np.ndarray,
# ) -> np.ndarray:
#     """Construct the linear map from A, C, F to fractional travel time."""

#     # Map the five Love-parameter derivative tensors
#     # (A, C, F, L, N) to the three inferred parameters
#     # (A, C, F).
#     transformation = np.array(
#         [
#             [1.0, 0.0, 0.0],  # dC/dA
#             [0.0, 1.0, 0.0],  # dC/dC
#             [0.0, 0.0, 1.0],  # dC/dF
#             [0.0, 0.0, 0.0],  # dC/dL, not inferred
#             [0.0, 0.0, 0.0],  # dC/dN, not inferred
#         ],
#         dtype=float,
#     )

#     n_segments = weights.shape[1]
#     n_paths = weights.shape[2]
#     n_parameters_per_segment = 3

#     derivative_tensors = np.stack(
#         [
#             gradient_C_wrt_A(),
#             gradient_C_wrt_C(),
#             gradient_C_wrt_F(),
#             gradient_C_wrt_L(),
#             gradient_C_wrt_N(),
#         ],
#         axis=0,
#     )

#     # Shape:
#     # (n_segments, 5, 6, 6)
#     derivative_tensors_broadcast = np.broadcast_to(
#         derivative_tensors[None, ...],
#         (
#             n_segments,
#             *derivative_tensors.shape,
#         ),
#     )

#     # Shape:
#     # (n_segments, 5, n_paths)
#     travel_time_derivatives = (
#         calculate_relative_traveltime_voigt(
#             path_directions,
#             derivative_tensors_broadcast,
#             normalisation=0.5,
#         )
#     )

#     logger.debug(
#         "construct_forward_map: weights.shape=%s "
#         "path_directions.shape=%s",
#         weights.shape,
#         path_directions.shape,
#     )

#     logger.debug(
#         "construct_forward_map: derivative_tensors.shape=%s "
#         "travel_time_derivatives.shape=%s",
#         derivative_tensors.shape,
#         travel_time_derivatives.shape,
#     )

#     # Contract over the five Love-parameter derivative tensors.
#     #
#     # Initial result:
#     # (n_segments, n_paths, 3)
#     #
#     # After transpose:
#     # (n_segments, 3, n_paths)
#     travel_time_model = np.tensordot(
#         travel_time_derivatives,
#         transformation,
#         axes=([1], [0]),
#     ).transpose(0, 2, 1)

#     segment_weights = weights[0, :, :]

#     weighted_travel_time_model = (
#         segment_weights[:, None, :]
#         * travel_time_model
#     )

#     # Final shape:
#     # (n_paths, n_segments * 3)
#     forward_matrix = weighted_travel_time_model.reshape(
#         n_segments * n_parameters_per_segment,
#         n_paths,
#     ).T

#     logger.debug(
#         "construct_forward_map: returning forward matrix "
#         "with shape %s",
#         forward_matrix.shape,
#     )

#     return forward_matrix


# def construct_Cd(
#     ref_phase: pd.Series,
#     ic_tt: pd.Series,
# ) -> np.ndarray:
#     """Construct the observational covariance used by the current model.

#     The phase-dependent noise levels are divided by inner-core travel
#     time to convert them to fractional travel-time units.

#     Important
#     ---------
#     This version preserves the original implementation and places
#     sigma directly on the diagonal. To match the original MCMC
#     likelihood exactly, use ``np.diag(sigma**2)`` instead.
#     """

#     noise_levels: dict[str, float] = {
#         "ab": 0.95,
#         "bc": 0.63,
#         "cd": 0.29,
#         "df": 0.95,
#     }

#     sigma = (
#         ref_phase.map(noise_levels)
#         / ic_tt
#     ).astype(float).to_numpy()

#     return np.diag(sigma**2)


# def diagnose_log_evidence(
#     data: np.ndarray,
#     inferred: list[GaussianComponent],
#     nuisance: list[GaussianComponent],
#     log_evidence: float,
# ) -> None:
#     """Print the three terms making up the Gaussian log evidence."""

#     forward_matrix = inferred[0].A

#     marginal_mean = (
#         forward_matrix @ inferred[0].mu
#         + nuisance[0].A @ nuisance[0].mu
#     )

#     marginal_covariance = (
#         forward_matrix
#         @ inferred[0].C
#         @ forward_matrix.T
#         + nuisance[0].A
#         @ nuisance[0].C
#         @ nuisance[0].A.T
#     )

#     difference = data - marginal_mean

#     # C = L L^T
#     cholesky_factor = np.linalg.cholesky(
#         marginal_covariance
#     )

#     # -M/2 log(2 pi)
#     normalisation_term = (
#         -0.5
#         * len(data)
#         * np.log(2.0 * np.pi)
#     )

#     # log |C| = 2 sum(log(diag(L)))
#     log_determinant = (
#         2.0
#         * np.sum(
#             np.log(
#                 np.diag(cholesky_factor)
#             )
#         )
#     )

#     log_determinant_term = (
#         -0.5 * log_determinant
#     )

#     # Solve L y = d - mu.
#     solved_difference = np.linalg.solve(
#         cholesky_factor,
#         difference,
#     )

#     # (d - mu)^T C^{-1} (d - mu) = y^T y
#     quadratic_form = float(
#         solved_difference
#         @ solved_difference
#     )

#     quadratic_term = (
#         -0.5 * quadratic_form
#     )

#     reconstructed_log_evidence = (
#         normalisation_term
#         + log_determinant_term
#         + quadratic_term
#     )

#     logger.info(
#         "Evidence normalisation term: %.6f",
#         normalisation_term,
#     )

#     logger.info(
#         "log|C_marginal|: %.6f",
#         log_determinant,
#     )

#     logger.info(
#         "Evidence log-determinant term: %.6f",
#         log_determinant_term,
#     )

#     logger.info(
#         "Quadratic form: %.6f",
#         quadratic_form,
#     )

#     logger.info(
#         "Evidence quadratic term: %.6f",
#         quadratic_term,
#     )

#     logger.info(
#         "Evidence reconstructed from terms: %.6f",
#         reconstructed_log_evidence,
#     )

#     logger.info(
#         "Difference from calc_log_evidence: %.12e",
#         reconstructed_log_evidence
#         - log_evidence,
#     )


# def main() -> None:
#     """Run the linear-Gaussian inner-core inversion."""

#     data_file = (
#         "data/brett2024_ic_traveltimes.parquet"
#     )

#     logger.info(
#         "Reading data from %s",
#         data_file,
#     )

#     df = pd.read_parquet(
#         data_file
#     )

#     data = (
#         df["delta_t"]
#         / df["inner_core_travel_time"]
#     ).astype(float).to_numpy()

#     ic_in = np.stack(
#         df["in_location"].to_numpy()
#     )

#     ic_out = np.stack(
#         df["out_location"].to_numpy()
#     )

#     path_directions = (
#         calculate_path_direction_vector(
#             ic_in,
#             ic_out,
#         )
#     )

#     logger.debug(
#         "Loaded data: n_obs=%d "
#         "ic_in.shape=%s ic_out.shape=%s",
#         data.shape[0],
#         ic_in.shape,
#         ic_out.shape,
#     )

#     # Original mesh resolution:
#     mesh = SphericalMesh(
#         1221.5,
#         4,
#         5,
#     )

#     weights = determine_weights(
#         mesh,
#         ic_in,
#         path_directions,
#     )

#     logger.info(
#         "Constructing forward matrix A"
#     )

#     forward_matrix = construct_forward_map(
#         path_directions,
#         weights,
#     )

#     n_data, n_parameters = (
#         forward_matrix.shape
#     )

#     logger.info(
#         "Forward matrix A shape: %s",
#         forward_matrix.shape,
#     )

#     inferred = [
#         GaussianComponent(
#             forward_matrix,
#             np.zeros(n_parameters),
#             0.1 * np.eye(n_parameters),
#         )
#     ]

#     nuisance = [
#         GaussianComponent(
#             np.eye(n_data),
#             np.zeros(n_data),
#             construct_Cd(
#                 df["reference_phase"],
#                 df["inner_core_travel_time"],
#             ),
#         )
#     ]

#     logger.info(
#         "Running inference: prior cov shape=%s "
#         "noise cov shape=%s",
#         inferred[0].C.shape,
#         nuisance[0].C.shape,
#     )

#     noise_covariance_diagonal = np.diag(
#         nuisance[0].C
#     )

#     logger.info(
#         "Noise covariance diagonal minimum: %.6e",
#         noise_covariance_diagonal.min(),
#     )

#     logger.info(
#         "Noise covariance diagonal maximum: %.6e",
#         noise_covariance_diagonal.max(),
#     )

#     posterior_covariance = calc_posterior_cov(
#         inferred,
#         nuisance,
#     )

#     logger.info(
#         "Posterior covariance shape: %s",
#         posterior_covariance.shape,
#     )

#     posterior_mean = calc_posterior_mean(
#         data,
#         inferred,
#         nuisance,
#     )

#     logger.info(
#         "Posterior mean shape: %s",
#         posterior_mean.shape,
#     )

#     log_evidence = calc_log_evidence(
#         data,
#         inferred,
#         nuisance,
#     )

#     logger.info(
#         "Log-evidence: %.12f",
#         log_evidence,
#     )

#     logger.info(
#         "Posterior mean: %s",
#         posterior_mean,
#     )

#     logger.info(
#         "Posterior standard deviations: %s",
#         np.sqrt(
#             np.diag(
#                 posterior_covariance
#             )
#         ),
#     )

#     # This must come after log_evidence has been calculated.
#     diagnose_log_evidence(
#         data=data,
#         inferred=inferred,
#         nuisance=nuisance,
#         log_evidence=log_evidence,
#     )


# if __name__ == "__main__":
#     main()



# """Solve the IC anisotropy problem as a linear Gaussian system.

# Model:

#     d = Gm + delta + epsilon

# where

#     m       = inferred Earth-model parameters
#     epsilon = independent observational noise
#     delta   = correlated nuisance error

# We assume

#     epsilon ~ N(0, C_epsilon)
#     delta   ~ N(0, C_delta)

# and delta is analytically marginalised, giving

#     d | m ~ N(Gm, C_epsilon + C_delta).

# The correlated covariance is based on similarity between ray paths.
# For two rays i and j, define

#     d_ij = 0.5 * (
#         distance(entry_i, entry_j)
#         + distance(exit_i, exit_j)
#     )

# and then

#     C_delta[i,j]
#         = tau^2 * exp(
#             -d_ij^2 / (2 * ell^2)
#         ).
# """

# import logging

# import numpy as np
# import pandas as pd

# from linear_gaussian import (
#     GaussianComponent,
#     calc_log_evidence,
#     calc_posterior_cov,
#     calc_posterior_mean,
# )

# from raytracer import SphericalMesh

# from tti.elastic.voigt import (
#     gradient_C_wrt_A,
#     gradient_C_wrt_C,
#     gradient_C_wrt_F,
#     gradient_C_wrt_L,
#     gradient_C_wrt_N,
# )

# from tti.traveltimes.parametrisations import (
#     LinearParametriser,
# )

# from tti.traveltimes.traveltimes import (
#     calculate_path_direction_vector,
#     calculate_relative_traveltime_voigt,
# )


# # ============================================================
# # USER SETTINGS
# # ============================================================

# USE_CORRELATED_ERROR = True

# # Standard deviation of correlated fractional travel-time error.
# CORRELATED_ERROR_TAU = 0.002

# # Correlation length in km.
# CORRELATION_LENGTH_KM = 200.0

# # Prior covariance multiplier.
# #
# # Keep this at the value you currently want to test.
# PRIOR_VARIANCE = 0.1

# # Mesh resolution.
# RADIAL_RESOLUTION = 4
# LATERAL_RESOLUTION = 5


# # ============================================================
# # LOGGING
# # ============================================================

# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s %(levelname)s %(name)s: %(message)s",
# )

# logger = logging.getLogger(__name__)


# # ============================================================
# # PARAMETRISATION
# # ============================================================


# class NoAnglesParametriser(LinearParametriser):
#     """Only P-wave Love parameters A, C, and F."""

#     n_model_params_per_segment = 3

#     transformation = np.array(
#         [
#             [1, 0, 0],
#             [0, 1, 0],
#             [0, 0, 1],
#             [0, 0, 0],
#             [0, 0, 0],
#             [0, 0, 0],
#             [0, 0, 0],
#         ],
#         dtype=float,
#     )


# # ============================================================
# # COORDINATE CONVERSION
# # ============================================================


# def lonlatrad_to_xyz(
#     lonlatrad: np.ndarray,
# ) -> np.ndarray:
#     """Convert longitude, latitude, radius to Cartesian xyz.

#     Parameters
#     ----------
#     lonlatrad
#         Array with shape (..., 3).

#         Columns:
#             longitude in degrees
#             latitude in degrees
#             radius in km

#     Returns
#     -------
#     np.ndarray
#         Cartesian coordinates in km.
#     """

#     lon = np.radians(
#         lonlatrad[..., 0]
#     )

#     lat = np.radians(
#         lonlatrad[..., 1]
#     )

#     radius = lonlatrad[..., 2]

#     x = (
#         radius
#         * np.cos(lat)
#         * np.cos(lon)
#     )

#     y = (
#         radius
#         * np.cos(lat)
#         * np.sin(lon)
#     )

#     z = (
#         radius
#         * np.sin(lat)
#     )

#     return np.stack(
#         [x, y, z],
#         axis=-1,
#     )


# # ============================================================
# # RAY WEIGHTS
# # ============================================================


# def determine_weights(
#     mesh: SphericalMesh,
#     ic_in: np.ndarray,
#     path_directions: np.ndarray,
# ) -> np.ndarray:
#     """Determine fractional distance of each path in each mesh cell."""

#     segment_distances = (
#         mesh.ray_distances_per_region(
#             lonlatrad_to_xyz(ic_in),
#             path_directions,
#         )
#     )

#     total_distances = (
#         segment_distances.sum(
#             axis=1,
#         )
#     )

#     weights = (
#         segment_distances
#         / total_distances[:, None]
#     )

#     # Shape:
#     # (1, number_of_segments, number_of_paths)
#     weights_for_calculator = (
#         weights.T[None, ...]
#     )

#     logger.debug(
#         "determine_weights: "
#         "segment_distances.shape=%s "
#         "total_distances.shape=%s",
#         segment_distances.shape,
#         total_distances.shape,
#     )

#     logger.debug(
#         "determine_weights: "
#         "computed weights for %d paths "
#         "and %d segments",
#         ic_in.shape[0],
#         segment_distances.shape[1],
#     )

#     return weights_for_calculator


# # ============================================================
# # FORWARD MAP
# # ============================================================


# def construct_forward_map(
#     path_directions: np.ndarray,
#     weights: np.ndarray,
# ) -> np.ndarray:
#     """Construct linear map from A, C, F to fractional travel time."""

#     transformation = np.array(
#         [
#             [1.0, 0.0, 0.0],  # A
#             [0.0, 1.0, 0.0],  # C
#             [0.0, 0.0, 1.0],  # F
#             [0.0, 0.0, 0.0],  # L not inferred
#             [0.0, 0.0, 0.0],  # N not inferred
#         ],
#         dtype=float,
#     )

#     n_segments = weights.shape[1]
#     n_paths = weights.shape[2]
#     n_parameters_per_segment = 3

#     derivative_tensors = np.stack(
#         [
#             gradient_C_wrt_A(),
#             gradient_C_wrt_C(),
#             gradient_C_wrt_F(),
#             gradient_C_wrt_L(),
#             gradient_C_wrt_N(),
#         ],
#         axis=0,
#     )

#     derivative_tensors_broadcast = (
#         np.broadcast_to(
#             derivative_tensors[None, ...],
#             (
#                 n_segments,
#                 *derivative_tensors.shape,
#             ),
#         )
#     )

#     travel_time_derivatives = (
#         calculate_relative_traveltime_voigt(
#             path_directions,
#             derivative_tensors_broadcast,
#             normalisation=0.5,
#         )
#     )

#     logger.debug(
#         "construct_forward_map: "
#         "weights.shape=%s "
#         "path_directions.shape=%s",
#         weights.shape,
#         path_directions.shape,
#     )

#     logger.debug(
#         "construct_forward_map: "
#         "derivative_tensors.shape=%s "
#         "travel_time_derivatives.shape=%s",
#         derivative_tensors.shape,
#         travel_time_derivatives.shape,
#     )

#     travel_time_model = np.tensordot(
#         travel_time_derivatives,
#         transformation,
#         axes=([1], [0]),
#     ).transpose(
#         0,
#         2,
#         1,
#     )

#     segment_weights = (
#         weights[0, :, :]
#     )

#     weighted_travel_time_model = (
#         segment_weights[:, None, :]
#         * travel_time_model
#     )

#     forward_matrix = (
#         weighted_travel_time_model.reshape(
#             n_segments
#             * n_parameters_per_segment,
#             n_paths,
#         ).T
#     )

#     logger.debug(
#         "construct_forward_map: "
#         "returning forward matrix "
#         "with shape %s",
#         forward_matrix.shape,
#     )

#     return forward_matrix


# # ============================================================
# # INDEPENDENT OBSERVATIONAL NOISE
# # ============================================================


# def calculate_noise_sigma(
#     ref_phase: pd.Series,
#     ic_tt: pd.Series,
# ) -> np.ndarray:
#     """Calculate fractional observational standard deviation."""

#     noise_levels: dict[str, float] = {
#         "ab": 0.95,
#         "bc": 0.63,
#         "cd": 0.29,
#         "df": 0.95,
#     }

#     sigma = (
#         ref_phase.map(noise_levels)
#         / ic_tt
#     ).astype(float).to_numpy()

#     if np.any(
#         ~np.isfinite(sigma)
#     ):
#         raise ValueError(
#             "Non-finite sigma values found."
#         )

#     if np.any(
#         sigma <= 0.0
#     ):
#         raise ValueError(
#             "All sigma values must be positive."
#         )

#     return sigma


# def construct_independent_covariance(
#     ref_phase: pd.Series,
#     ic_tt: pd.Series,
# ) -> np.ndarray:
#     """Construct independent covariance C_epsilon."""

#     sigma = calculate_noise_sigma(
#         ref_phase,
#         ic_tt,
#     )

#     return np.diag(
#         sigma**2
#     )


# # ============================================================
# # PAIRWISE DISTANCE
# # ============================================================


# def calculate_pairwise_distances(
#     xyz: np.ndarray,
# ) -> np.ndarray:
#     """Calculate Euclidean distance between every pair of xyz points."""

#     squared_norms = np.sum(
#         xyz**2,
#         axis=1,
#     )

#     distance_squared = (
#         squared_norms[:, None]
#         + squared_norms[None, :]
#         - 2.0 * (
#             xyz @ xyz.T
#         )
#     )

#     # Remove tiny negative values caused by floating-point error.
#     distance_squared = np.maximum(
#         distance_squared,
#         0.0,
#     )

#     return np.sqrt(
#         distance_squared
#     )


# # ============================================================
# # CORRELATED COVARIANCE
# # ============================================================


# def construct_correlated_covariance(
#     ic_in: np.ndarray,
#     ic_out: np.ndarray,
#     tau: float,
#     correlation_length_km: float,
# ) -> np.ndarray:
#     """Construct C_delta from entry and exit-point similarity.

#     For paths i and j:

#         d_ij = 0.5 * (
#             ||entry_i - entry_j||
#             + ||exit_i - exit_j||
#         )

#     and

#         C_delta[i,j]
#             = tau^2
#               * exp(
#                   -d_ij^2
#                   / (2 * correlation_length_km^2)
#               )

#     Parameters
#     ----------
#     ic_in
#         Inner-core entry locations.

#     ic_out
#         Inner-core exit locations.

#     tau
#         Standard deviation of correlated fractional error.

#     correlation_length_km
#         Correlation scale in km.

#     Returns
#     -------
#     np.ndarray
#         Correlated covariance matrix.
#     """

#     if tau < 0.0:
#         raise ValueError(
#             "tau must be non-negative."
#         )

#     if correlation_length_km <= 0.0:
#         raise ValueError(
#             "correlation_length_km must be positive."
#         )

#     # Convert entry and exit positions to Cartesian coordinates.
#     xyz_in = lonlatrad_to_xyz(
#         ic_in
#     )

#     xyz_out = lonlatrad_to_xyz(
#         ic_out
#     )

#     # Pairwise entry-point distances.
#     distance_in = (
#         calculate_pairwise_distances(
#             xyz_in
#         )
#     )

#     # Pairwise exit-point distances.
#     distance_out = (
#         calculate_pairwise_distances(
#             xyz_out
#         )
#     )

#     # Define overall path-to-path distance.
#     path_distance = (
#         0.5
#         * (
#             distance_in
#             + distance_out
#         )
#     )

#     # Gaussian / RBF covariance kernel.
#     covariance = (
#         tau**2
#         * np.exp(
#             -(
#                 path_distance**2
#             )
#             / (
#                 2.0
#                 * correlation_length_km**2
#             )
#         )
#     )

#     logger.info(
#         "Constructed correlated covariance "
#         "using entry + exit path distance"
#     )

#     logger.info(
#         "tau = %.6f",
#         tau,
#     )

#     logger.info(
#         "correlation length = %.1f km",
#         correlation_length_km,
#     )

#     logger.info(
#         "Correlated covariance shape: %s",
#         covariance.shape,
#     )

#     logger.info(
#         "C_delta diagonal: %.6e",
#         covariance[0, 0],
#     )

#     return covariance


# # ============================================================
# # TOTAL NUISANCE COVARIANCE
# # ============================================================


# def construct_total_covariance(
#     ref_phase: pd.Series,
#     ic_tt: pd.Series,
#     ic_in: np.ndarray,
#     ic_out: np.ndarray,
#     use_correlated_error: bool,
#     tau: float,
#     correlation_length_km: float,
# ) -> np.ndarray:
#     """Construct C_total = C_epsilon + C_delta."""

#     sigma = calculate_noise_sigma(
#         ref_phase,
#         ic_tt,
#     )

#     n_data = len(
#         sigma
#     )

#     if use_correlated_error:

#         covariance = (
#             construct_correlated_covariance(
#                 ic_in=ic_in,
#                 ic_out=ic_out,
#                 tau=tau,
#                 correlation_length_km=(
#                     correlation_length_km
#                 ),
#             )
#         )

#         # Add independent observational variance
#         # to diagonal.
#         diagonal_indices = (
#             np.diag_indices(
#                 n_data
#             )
#         )

#         covariance[
#             diagonal_indices
#         ] += sigma**2

#     else:

#         covariance = np.diag(
#             sigma**2
#         )

#     return covariance


# # ============================================================
# # LOG-EVIDENCE DIAGNOSTIC
# # ============================================================


# def diagnose_log_evidence(
#     data: np.ndarray,
#     inferred: list[GaussianComponent],
#     nuisance: list[GaussianComponent],
#     log_evidence: float,
# ) -> None:
#     """Print components of Gaussian log evidence."""

#     forward_matrix = (
#         inferred[0].A
#     )

#     marginal_mean = (
#         forward_matrix
#         @ inferred[0].mu
#         + nuisance[0].A
#         @ nuisance[0].mu
#     )

#     marginal_covariance = (
#         forward_matrix
#         @ inferred[0].C
#         @ forward_matrix.T
#         + nuisance[0].A
#         @ nuisance[0].C
#         @ nuisance[0].A.T
#     )

#     difference = (
#         data
#         - marginal_mean
#     )

#     cholesky_factor = (
#         np.linalg.cholesky(
#             marginal_covariance
#         )
#     )

#     # Term 1:
#     #
#     # -M/2 log(2 pi)
#     normalisation_term = (
#         -0.5
#         * len(data)
#         * np.log(
#             2.0 * np.pi
#         )
#     )

#     # Term 2:
#     #
#     # -1/2 log |C|
#     log_determinant = (
#         2.0
#         * np.sum(
#             np.log(
#                 np.diag(
#                     cholesky_factor
#                 )
#             )
#         )
#     )

#     log_determinant_term = (
#         -0.5
#         * log_determinant
#     )

#     # Term 3:
#     #
#     # -1/2 (d-mu)^T C^-1 (d-mu)
#     solved_difference = (
#         np.linalg.solve(
#             cholesky_factor,
#             difference,
#         )
#     )

#     quadratic_form = float(
#         solved_difference
#         @ solved_difference
#     )

#     quadratic_term = (
#         -0.5
#         * quadratic_form
#     )

#     reconstructed_log_evidence = (
#         normalisation_term
#         + log_determinant_term
#         + quadratic_term
#     )

#     logger.info(
#         "Evidence normalisation term: %.6f",
#         normalisation_term,
#     )

#     logger.info(
#         "log|C_marginal|: %.6f",
#         log_determinant,
#     )

#     logger.info(
#         "Evidence log-determinant term: %.6f",
#         log_determinant_term,
#     )

#     logger.info(
#         "Quadratic form: %.6f",
#         quadratic_form,
#     )

#     logger.info(
#         "Evidence quadratic term: %.6f",
#         quadratic_term,
#     )

#     logger.info(
#         "Evidence reconstructed from terms: %.6f",
#         reconstructed_log_evidence,
#     )

#     logger.info(
#         "Difference from calc_log_evidence: %.12e",
#         reconstructed_log_evidence
#         - log_evidence,
#     )


# # ============================================================
# # MAIN
# # ============================================================


# def main() -> None:
#     """Run the linear-Gaussian inner-core inversion."""

#     data_file = (
#         "data/brett2024_ic_traveltimes.parquet"
#     )

#     logger.info(
#         "Reading data from %s",
#         data_file,
#     )

#     df = pd.read_parquet(
#         data_file
#     )

#     # --------------------------------------------------------
#     # DATA
#     # --------------------------------------------------------

#     data = (
#         df["delta_t"]
#         / df["inner_core_travel_time"]
#     ).astype(float).to_numpy()

#     ic_in = np.stack(
#         df["in_location"].to_numpy()
#     )

#     ic_out = np.stack(
#         df["out_location"].to_numpy()
#     )

#     path_directions = (
#         calculate_path_direction_vector(
#             ic_in,
#             ic_out,
#         )
#     )

#     logger.debug(
#         "Loaded data: "
#         "n_obs=%d "
#         "ic_in.shape=%s "
#         "ic_out.shape=%s",
#         data.shape[0],
#         ic_in.shape,
#         ic_out.shape,
#     )

#     # --------------------------------------------------------
#     # EARTH MODEL MESH
#     # --------------------------------------------------------

#     mesh = SphericalMesh(
#         1221.5,
#         RADIAL_RESOLUTION,
#         LATERAL_RESOLUTION,
#     )

#     weights = determine_weights(
#         mesh,
#         ic_in,
#         path_directions,
#     )

#     # --------------------------------------------------------
#     # FORWARD MATRIX
#     # --------------------------------------------------------

#     logger.info(
#         "Constructing forward matrix A"
#     )

#     forward_matrix = (
#         construct_forward_map(
#             path_directions,
#             weights,
#         )
#     )

#     (
#         n_data,
#         n_parameters,
#     ) = forward_matrix.shape

#     logger.info(
#         "Forward matrix A shape: %s",
#         forward_matrix.shape,
#     )

#     # --------------------------------------------------------
#     # PRIOR
#     # --------------------------------------------------------

#     inferred = [
#         GaussianComponent(
#             forward_matrix,
#             np.zeros(
#                 n_parameters
#             ),
#             PRIOR_VARIANCE
#             * np.eye(
#                 n_parameters
#             ),
#         )
#     ]

#     # --------------------------------------------------------
#     # TOTAL NOISE / NUISANCE COVARIANCE
#     # --------------------------------------------------------

#     logger.info(
#         "Constructing nuisance covariance..."
#     )

#     logger.info(
#         "USE_CORRELATED_ERROR = %s",
#         USE_CORRELATED_ERROR,
#     )

#     if USE_CORRELATED_ERROR:

#         logger.info(
#             "tau = %.6f",
#             CORRELATED_ERROR_TAU,
#         )

#         logger.info(
#             "correlation length = %.1f km",
#             CORRELATION_LENGTH_KM,
#         )

#     total_covariance = (
#         construct_total_covariance(
#             ref_phase=(
#                 df["reference_phase"]
#             ),
#             ic_tt=(
#                 df[
#                     "inner_core_travel_time"
#                 ]
#             ),
#             ic_in=ic_in,
#             ic_out=ic_out,
#             use_correlated_error=(
#                 USE_CORRELATED_ERROR
#             ),
#             tau=(
#                 CORRELATED_ERROR_TAU
#             ),
#             correlation_length_km=(
#                 CORRELATION_LENGTH_KM
#             ),
#         )
#     )

#     # --------------------------------------------------------
#     # NUISANCE COMPONENT
#     # --------------------------------------------------------

#     nuisance = [
#         GaussianComponent(
#             np.eye(
#                 n_data
#             ),
#             np.zeros(
#                 n_data
#             ),
#             total_covariance,
#         )
#     ]

#     logger.info(
#         "Running inference: "
#         "prior cov shape=%s "
#         "noise cov shape=%s",
#         inferred[0].C.shape,
#         nuisance[0].C.shape,
#     )

#     # --------------------------------------------------------
#     # COVARIANCE CHECKS
#     # --------------------------------------------------------

#     covariance_diagonal = (
#         np.diag(
#             nuisance[0].C
#         )
#     )

#     logger.info(
#         "Total covariance diagonal minimum: %.6e",
#         covariance_diagonal.min(),
#     )

#     logger.info(
#         "Total covariance diagonal maximum: %.6e",
#         covariance_diagonal.max(),
#     )

#     logger.info(
#         "Example covariance C[0,1]: %.6e",
#         nuisance[0].C[
#             0,
#             1,
#         ],
#     )

#     logger.info(
#         "Example covariance C[0,100]: %.6e",
#         nuisance[0].C[
#             0,
#             100,
#         ],
#     )

#     # --------------------------------------------------------
#     # POSTERIOR COVARIANCE
#     # --------------------------------------------------------

#     posterior_covariance = (
#         calc_posterior_cov(
#             inferred,
#             nuisance,
#         )
#     )

#     logger.info(
#         "Posterior covariance shape: %s",
#         posterior_covariance.shape,
#     )

#     # --------------------------------------------------------
#     # POSTERIOR MEAN
#     # --------------------------------------------------------

#     posterior_mean = (
#         calc_posterior_mean(
#             data,
#             inferred,
#             nuisance,
#         )
#     )

#     logger.info(
#         "Posterior mean shape: %s",
#         posterior_mean.shape,
#     )

#     # --------------------------------------------------------
#     # LOG EVIDENCE
#     # --------------------------------------------------------

#     log_evidence = (
#         calc_log_evidence(
#             data,
#             inferred,
#             nuisance,
#         )
#     )

#     logger.info(
#         "Log-evidence: %.12f",
#         log_evidence,
#     )

#     logger.info(
#         "Posterior mean: %s",
#         posterior_mean,
#     )

#     logger.info(
#         "Posterior standard deviations: %s",
#         np.sqrt(
#             np.diag(
#                 posterior_covariance
#             )
#         ),
#     )

#     # --------------------------------------------------------
#     # EVIDENCE DIAGNOSTIC
#     # --------------------------------------------------------

#     diagnose_log_evidence(
#         data=data,
#         inferred=inferred,
#         nuisance=nuisance,
#         log_evidence=log_evidence,
#     )


# if __name__ == "__main__":
#     main()


"""Solve the IC anisotropy problem as a linear Gaussian system.

Model:

    d = Gm + delta + epsilon

where

    m       = inferred Earth-model parameters
    epsilon = independent observational noise
    delta   = correlated nuisance error

We assume

    epsilon ~ N(0, C_epsilon)
    delta   ~ N(0, C_delta)

and marginalise delta analytically:

    d | m ~ N(Gm, C_epsilon + C_delta).

For two ray paths i and j, define

    d_ij^2 =
        0.5 * (
            ||entry_i - entry_j||^2
            +
            ||exit_i - exit_j||^2
        )

and

    C_delta[i,j]
        = tau^2 * exp(
            -d_ij^2 / (2 * ell^2)
        ).

This is a valid Gaussian/RBF covariance kernel.
"""

import logging

import numpy as np
import pandas as pd

from linear_gaussian import (
    GaussianComponent,
    calc_log_evidence,
    calc_posterior_cov,
    calc_posterior_mean,
)

from raytracer import SphericalMesh

from tti.elastic.voigt import (
    gradient_C_wrt_A,
    gradient_C_wrt_C,
    gradient_C_wrt_F,
    gradient_C_wrt_L,
    gradient_C_wrt_N,
)

from tti.traveltimes.parametrisations import (
    LinearParametriser,
)

from tti.traveltimes.traveltimes import (
    calculate_path_direction_vector,
    calculate_relative_traveltime_voigt,
)


# ============================================================
# USER SETTINGS
# ============================================================

USE_CORRELATED_ERROR = True

CORRELATED_ERROR_TAU = 0.002

CORRELATION_LENGTH_KM = 200.0

PRIOR_VARIANCE = 0.1

RADIAL_RESOLUTION = 4
LATERAL_RESOLUTION = 5


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# PARAMETRISATION
# ============================================================


class NoAnglesParametriser(LinearParametriser):
    """Only P-wave Love parameters A, C, and F."""

    n_model_params_per_segment = 3

    transformation = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=float,
    )


# ============================================================
# COORDINATE CONVERSION
# ============================================================


def lonlatrad_to_xyz(
    lonlatrad: np.ndarray,
) -> np.ndarray:
    """Convert longitude, latitude, radius to Cartesian xyz."""

    lon = np.radians(
        lonlatrad[..., 0]
    )

    lat = np.radians(
        lonlatrad[..., 1]
    )

    radius = lonlatrad[..., 2]

    x = (
        radius
        * np.cos(lat)
        * np.cos(lon)
    )

    y = (
        radius
        * np.cos(lat)
        * np.sin(lon)
    )

    z = (
        radius
        * np.sin(lat)
    )

    return np.stack(
        [x, y, z],
        axis=-1,
    )


# ============================================================
# RAY WEIGHTS
# ============================================================


def determine_weights(
    mesh: SphericalMesh,
    ic_in: np.ndarray,
    path_directions: np.ndarray,
) -> np.ndarray:
    """Determine fractional distance of each path in each mesh cell."""

    segment_distances = (
        mesh.ray_distances_per_region(
            lonlatrad_to_xyz(ic_in),
            path_directions,
        )
    )

    total_distances = (
        segment_distances.sum(
            axis=1,
        )
    )

    weights = (
        segment_distances
        / total_distances[:, None]
    )

    weights_for_calculator = (
        weights.T[None, ...]
    )

    logger.debug(
        "determine_weights: "
        "segment_distances.shape=%s "
        "total_distances.shape=%s",
        segment_distances.shape,
        total_distances.shape,
    )

    logger.debug(
        "determine_weights: "
        "computed weights for %d paths "
        "and %d segments",
        ic_in.shape[0],
        segment_distances.shape[1],
    )

    return weights_for_calculator


# ============================================================
# FORWARD MAP
# ============================================================


def construct_forward_map(
    path_directions: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Construct linear map from A, C, F to fractional travel time."""

    transformation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    n_segments = weights.shape[1]
    n_paths = weights.shape[2]
    n_parameters_per_segment = 3

    derivative_tensors = np.stack(
        [
            gradient_C_wrt_A(),
            gradient_C_wrt_C(),
            gradient_C_wrt_F(),
            gradient_C_wrt_L(),
            gradient_C_wrt_N(),
        ],
        axis=0,
    )

    derivative_tensors_broadcast = (
        np.broadcast_to(
            derivative_tensors[None, ...],
            (
                n_segments,
                *derivative_tensors.shape,
            ),
        )
    )

    travel_time_derivatives = (
        calculate_relative_traveltime_voigt(
            path_directions,
            derivative_tensors_broadcast,
            normalisation=0.5,
        )
    )

    travel_time_model = np.tensordot(
        travel_time_derivatives,
        transformation,
        axes=([1], [0]),
    ).transpose(
        0,
        2,
        1,
    )

    segment_weights = (
        weights[0, :, :]
    )

    weighted_travel_time_model = (
        segment_weights[:, None, :]
        * travel_time_model
    )

    forward_matrix = (
        weighted_travel_time_model.reshape(
            n_segments
            * n_parameters_per_segment,
            n_paths,
        ).T
    )

    logger.debug(
        "construct_forward_map: returning forward matrix "
        "with shape %s",
        forward_matrix.shape,
    )

    return forward_matrix


# ============================================================
# OBSERVATIONAL NOISE
# ============================================================


def calculate_noise_sigma(
    ref_phase: pd.Series,
    ic_tt: pd.Series,
) -> np.ndarray:
    """Calculate fractional observational standard deviations."""

    noise_levels: dict[str, float] = {
        "ab": 0.95,
        "bc": 0.63,
        "cd": 0.29,
        "df": 0.95,
    }

    sigma = (
        ref_phase.map(noise_levels)
        / ic_tt
    ).astype(float).to_numpy()

    if np.any(
        ~np.isfinite(sigma)
    ):
        raise ValueError(
            "Non-finite sigma values found."
        )

    if np.any(
        sigma <= 0.0
    ):
        raise ValueError(
            "All sigma values must be positive."
        )

    return sigma


def construct_independent_covariance(
    ref_phase: pd.Series,
    ic_tt: pd.Series,
) -> np.ndarray:
    """Construct diagonal observational covariance C_epsilon."""

    sigma = calculate_noise_sigma(
        ref_phase,
        ic_tt,
    )

    return np.diag(
        sigma**2
    )


# ============================================================
# VALID ENTRY + EXIT PATH DISTANCE
# ============================================================


def construct_path_distance_squared_matrix(
    ic_in: np.ndarray,
    ic_out: np.ndarray,
) -> np.ndarray:
    """Construct average squared entry/exit path distance.

    d_ij^2 =
        0.5 * (
            ||entry_i - entry_j||^2
            +
            ||exit_i - exit_j||^2
        )
    """

    xyz_in = lonlatrad_to_xyz(
        ic_in
    )

    xyz_out = lonlatrad_to_xyz(
        ic_out
    )

    in_norms = np.sum(
        xyz_in**2,
        axis=1,
    )

    distance_in_squared = (
        in_norms[:, None]
        + in_norms[None, :]
        - 2.0 * (
            xyz_in @ xyz_in.T
        )
    )

    distance_in_squared = np.maximum(
        distance_in_squared,
        0.0,
    )

    out_norms = np.sum(
        xyz_out**2,
        axis=1,
    )

    distance_out_squared = (
        out_norms[:, None]
        + out_norms[None, :]
        - 2.0 * (
            xyz_out @ xyz_out.T
        )
    )

    distance_out_squared = np.maximum(
        distance_out_squared,
        0.0,
    )

    path_distance_squared = (
        0.5
        * (
            distance_in_squared
            + distance_out_squared
        )
    )

    return path_distance_squared


# ============================================================
# CORRELATED COVARIANCE
# ============================================================


def construct_correlated_covariance(
    path_distance_squared: np.ndarray,
    tau: float,
    correlation_length_km: float,
) -> np.ndarray:
    """Construct valid Gaussian/RBF correlated covariance."""

    if tau < 0.0:
        raise ValueError(
            "tau must be non-negative."
        )

    if correlation_length_km <= 0.0:
        raise ValueError(
            "correlation_length_km must be positive."
        )

    covariance = (
        tau**2
        * np.exp(
            -path_distance_squared
            / (
                2.0
                * correlation_length_km**2
            )
        )
    )

    covariance = (
        0.5
        * (
            covariance
            + covariance.T
        )
    )

    return covariance


# ============================================================
# TOTAL COVARIANCE
# ============================================================


def construct_total_covariance(
    ref_phase: pd.Series,
    ic_tt: pd.Series,
    path_distance_squared: np.ndarray,
    use_correlated_error: bool,
    tau: float,
    correlation_length_km: float,
) -> np.ndarray:
    """Construct C_total = C_epsilon + C_delta."""

    sigma = calculate_noise_sigma(
        ref_phase,
        ic_tt,
    )

    if use_correlated_error:

        covariance = (
            construct_correlated_covariance(
                path_distance_squared=(
                    path_distance_squared
                ),
                tau=tau,
                correlation_length_km=(
                    correlation_length_km
                ),
            )
        )

        diagonal_indices = (
            np.diag_indices_from(
                covariance
            )
        )

        covariance[
            diagonal_indices
        ] += sigma**2

    else:

        covariance = np.diag(
            sigma**2
        )

    covariance = (
        0.5
        * (
            covariance
            + covariance.T
        )
    )

    return covariance


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    """Run the linear-Gaussian inversion."""

    data_file = (
        "data/brett2024_ic_traveltimes.parquet"
    )

    logger.info(
        "Reading data from %s",
        data_file,
    )

    df = pd.read_parquet(
        data_file
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

    forward_matrix = construct_forward_map(
        path_directions,
        weights,
    )

    n_data, n_parameters = (
        forward_matrix.shape
    )

    logger.info(
        "Forward matrix A shape: %s",
        forward_matrix.shape,
    )

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

    path_distance_squared = (
        construct_path_distance_squared_matrix(
            ic_in,
            ic_out,
        )
    )

    total_covariance = (
        construct_total_covariance(
            ref_phase=(
                df["reference_phase"]
            ),
            ic_tt=(
                df[
                    "inner_core_travel_time"
                ]
            ),
            path_distance_squared=(
                path_distance_squared
            ),
            use_correlated_error=(
                USE_CORRELATED_ERROR
            ),
            tau=(
                CORRELATED_ERROR_TAU
            ),
            correlation_length_km=(
                CORRELATION_LENGTH_KM
            ),
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
            total_covariance,
        )
    ]

    posterior_covariance = (
        calc_posterior_cov(
            inferred,
            nuisance,
        )
    )

    posterior_mean = (
        calc_posterior_mean(
            data,
            inferred,
            nuisance,
        )
    )

    log_evidence = (
        calc_log_evidence(
            data,
            inferred,
            nuisance,
        )
    )

    logger.info(
        "Log-evidence: %.12f",
        log_evidence,
    )

    logger.info(
        "Posterior covariance shape: %s",
        posterior_covariance.shape,
    )

    logger.info(
        "Posterior mean shape: %s",
        posterior_mean.shape,
    )

    logger.info(
        "Total covariance diagonal min: %.6e",
        np.diag(
            total_covariance
        ).min(),
    )

    logger.info(
        "Total covariance diagonal max: %.6e",
        np.diag(
            total_covariance
        ).max(),
    )

    logger.info(
        "Example C[0,1]: %.6e",
        total_covariance[
            0,
            1,
        ],
    )


if __name__ == "__main__":
    main()
