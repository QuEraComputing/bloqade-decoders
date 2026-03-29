from __future__ import annotations

import numpy as np


def pack_boolean_array(arr: np.ndarray) -> np.ndarray:
    """Pack a boolean array into int64 (each row becomes one int)."""
    data_len = arr.shape[1]
    return np.sum(arr << np.arange(data_len), axis=1)  # type: ignore[arg-type]


def unpack_boolean_array(arr: np.ndarray, data_len: int) -> np.ndarray:
    """Unpack int64 array back to boolean array."""
    return (arr.reshape(-1, 1) & (1 << np.arange(data_len))) > 0


def shots_to_counts(shots: np.ndarray) -> np.ndarray:
    """Convert boolean shots array to histogram of counts."""
    packed_shots = pack_boolean_array(shots)
    counts = np.bincount(packed_shots, minlength=2 ** shots.shape[1])
    return counts


def det_obs_shots_to_counts(det_shots: np.ndarray, obs_shots: np.ndarray) -> np.ndarray:
    """Convert detector and observable shots into combined counts."""
    shots = np.concatenate([det_shots, obs_shots], axis=1)
    counts = shots_to_counts(shots)
    return counts
