from __future__ import annotations

from typing import Protocol

import stim
import numpy as np
import numpy.typing as npt
from beliefmatching import detector_error_model_to_check_matrices


class DetectorErrorModelTask(Protocol):
    """Protocol for objects exposing a Stim detector error model."""

    @property
    def detector_error_model(self) -> stim.DetectorErrorModel: ...


def make_layout_only_dem(
    num_detectors: int,
    num_observables: int,
) -> stim.DetectorErrorModel:
    """Create a minimal DEM carrying detector and observable dimensions."""

    terms: list[str] = []
    if num_detectors:
        terms.append(" ".join(f"D{i}" for i in range(int(num_detectors))))
    if num_observables:
        terms.append(" ".join(f"L{i}" for i in range(int(num_observables))))
    if not terms:
        raise ValueError("Need at least one detector or observable.")
    return stim.DetectorErrorModel("\n".join(f"error(0.5) {term}" for term in terms))


def matrix_to_dem(
    check_matrix: np.ndarray,
    observables_matrix: np.ndarray,
    priors: np.ndarray,
) -> stim.DetectorErrorModel:
    """Convert binary detector/observable matrices into a Stim DEM."""

    check = np.asarray(check_matrix, dtype=np.uint8)
    observables = np.asarray(observables_matrix, dtype=np.uint8)
    prior_arr = np.asarray(priors, dtype=np.float64)
    if check.ndim != 2 or observables.ndim != 2:
        raise ValueError("check_matrix and observables_matrix must be 2D.")
    if check.shape[1] != observables.shape[1] or check.shape[1] != len(prior_arr):
        raise ValueError("Matrices and priors must describe the same errors.")

    lines: list[str] = []
    for col, prior in enumerate(prior_arr):
        det_targets = [f"D{i}" for i in np.flatnonzero(check[:, col])]
        obs_targets = [f"L{i}" for i in np.flatnonzero(observables[:, col])]
        if not det_targets and not obs_targets:
            continue
        safe_prior = float(np.clip(prior, 1e-12, 1.0 - 1e-12))
        lines.append(f"error({safe_prior:.16g}) " + " ".join(det_targets + obs_targets))
    if not lines:
        raise ValueError("Matrix reduction produced an empty DEM.")
    return stim.DetectorErrorModel("\n".join(lines))


def detector_error_model_matrices(
    task_or_dem: DetectorErrorModelTask | stim.DetectorErrorModel,
) -> dict[str, npt.NDArray[np.float64] | npt.NDArray[np.int64]]:
    """Extract check matrices, observable matrices, and priors from a DEM."""

    dem = (
        task_or_dem
        if isinstance(task_or_dem, stim.DetectorErrorModel)
        else task_or_dem.detector_error_model
    )
    dem_matrix = detector_error_model_to_check_matrices(
        dem,
        allow_undecomposed_hyperedges=True,
    )
    return {
        "H": dem_matrix.check_matrix.toarray().astype(np.int64),
        "O": dem_matrix.observables_matrix.toarray().astype(np.int64),
        "priors": np.asarray(dem_matrix.priors, dtype=np.float64),
    }
