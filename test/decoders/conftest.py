"""Shared test helpers for sinter decoder tests."""

import stim
import numpy as np


def simple_dem() -> stim.DetectorErrorModel:
    return stim.DetectorErrorModel(
        """
        error(0.1) D0 L0
        error(0.1) D1 L0
        """
    )


def repetition_circuit() -> stim.Circuit:
    return stim.Circuit(
        """
        R 0 1 2
        X_ERROR(0.1) 0 1 2
        MZZ 0 1
        DETECTOR rec[-1]
        MZZ 1 2
        DETECTOR rec[-1]
        M 0 1 2
        OBSERVABLE_INCLUDE(0) rec[-1] rec[-2] rec[-3]
    """
    )


def pack_dets(det_shots: np.ndarray) -> np.ndarray:
    return np.packbits(det_shots.astype(np.uint8), axis=1, bitorder="little")


def unpack_obs(packed_obs: np.ndarray, num_obs: int) -> np.ndarray:
    unpacked = np.unpackbits(packed_obs, axis=1, bitorder="little")
    return unpacked[:, :num_obs].astype(bool)
