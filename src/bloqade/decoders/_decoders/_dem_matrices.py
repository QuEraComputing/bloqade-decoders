"""Convert a detector error model into check matrices for LDPC decoders.

Adapted from ``detector_error_model_to_check_matrices`` in beliefmatching
(https://github.com/oscarhiggott/BeliefMatching, Apache-2.0), keeping only
the hyperedge check matrix, observables matrix, and priors.
"""

from typing import cast
from dataclasses import dataclass

import stim
import numpy as np
from scipy.sparse import csc_matrix


@dataclass(frozen=True)
class DemMatrices:
    """Check-matrix representation of a detector error model.

    Attributes:
        check_matrix: ``(num_detectors, num_errors)`` matrix; column ``j``
            marks the detectors flipped by error mechanism ``j``.
        observables_matrix: ``(num_observables, num_errors)`` matrix; column
            ``j`` marks the observables flipped by error mechanism ``j``.
        priors: Probability of each error mechanism.
    """

    check_matrix: csc_matrix
    observables_matrix: csc_matrix
    priors: np.ndarray


def _xor_sets(target_lists: list[list[int]]) -> frozenset[int]:
    out: set[int] = set()
    for targets in target_lists:
        out ^= set(targets)
    return frozenset(out)


def _columns_to_csc(columns: list[frozenset[int]], num_rows: int) -> csc_matrix:
    row_ind = np.fromiter((row for col in columns for row in col), dtype=np.int64)
    col_ind = np.fromiter(
        (j for j, col in enumerate(columns) for _ in col), dtype=np.int64
    )
    data = np.ones(len(row_ind), dtype=np.uint8)
    return csc_matrix((data, (row_ind, col_ind)), shape=(num_rows, len(columns)))


def dem_to_check_matrices(dem: stim.DetectorErrorModel) -> DemMatrices:
    """Build check matrices and priors from a detector error model.

    Each error instruction's components (split by ``^`` separators) are
    XOR'd together. Instructions flipping the same set of detectors are
    merged into one column with probability ``p1 (1 - p2) + p2 (1 - p1)``;
    the merged column keeps the observables of the last such instruction.

    Args:
        dem: The detector error model.

    Returns:
        The check matrix, observables matrix, and priors.
    """
    column_of: dict[frozenset[int], int] = {}
    det_columns: list[frozenset[int]] = []
    obs_columns: list[frozenset[int]] = []
    priors: list[float] = []

    for instruction in dem.flattened():
        assert isinstance(instruction, stim.DemInstruction)
        if instruction.type in ("detector", "logical_observable"):
            continue
        if instruction.type != "error":
            raise NotImplementedError(
                f"Unsupported DEM instruction type: {instruction.type}"
            )
        p = instruction.args_copy()[0]
        dets: list[list[int]] = [[]]
        obs: list[list[int]] = [[]]
        for target in instruction.targets_copy():
            t = cast(stim.DemTarget, target)
            if t.is_relative_detector_id():
                dets[-1].append(t.val)
            elif t.is_logical_observable_id():
                obs[-1].append(t.val)
            elif t.is_separator():
                dets.append([])
                obs.append([])
        det_set = _xor_sets(dets)
        obs_set = _xor_sets(obs)

        if det_set not in column_of:
            column_of[det_set] = len(det_columns)
            det_columns.append(det_set)
            obs_columns.append(obs_set)
            priors.append(0.0)
        j = column_of[det_set]
        obs_columns[j] = obs_set
        priors[j] = priors[j] * (1 - p) + p * (1 - priors[j])

    return DemMatrices(
        check_matrix=_columns_to_csc(det_columns, dem.num_detectors),
        observables_matrix=_columns_to_csc(obs_columns, dem.num_observables),
        priors=np.array(priors, dtype=np.float64),
    )
