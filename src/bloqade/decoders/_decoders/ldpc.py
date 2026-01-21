from typing import Literal

import stim
import numpy as np
import numpy.typing as npt
from beliefmatching import detector_error_model_to_check_matrices
from ldpc.bplsd_decoder import BpLsdDecoder as LdpcBpLsdDecoder
from ldpc.bposd_decoder import BpOsdDecoder as LdpcBpOsdDecoder
from ldpc.belief_find_decoder import BeliefFindDecoder as LdpcBeliefFindDecoder

from .base import BaseDecoder


class BeliefFindDecoder(BaseDecoder):
    """Belief propagation + union-find decoder.

    Arguments match ldpc.BeliefFindDecoder exactly, except a stim.DetectorErrorModel
    is expected in place of the parity check matrix and error channel priors.
    Defaults are used if not specified. The information below is taken straight from the
    original ldpc.BeliefFindDecoder docstring.

    Args:
        dem (stim.DetectorErrorModel): The detector error model describing the error structure.
        max_iter (int): Maximum number of BP iterations, by default 0.
        bp_method (Literal["product_sum", "minimum_sum"]): BP method to use, by default 'minimum_sum'.
        ms_scaling_factor (float): Scaling factor used in the minimum sum method,
            by default 1.0.
        schedule (Literal["parallel", "serial"]): Update schedule, by default 'parallel'.
        omp_thread_count (int): Number of OpenMP threads for parallel decoding,
            by default 1.
        random_schedule_seed (int): Seed for random serial schedule order,
            by default 0.
        serial_schedule_order (Optional[list[int]]): List specifying the serial schedule
            order. Must be of length equal to the block length of the code, by default None.
        uf_method (Literal["inversion", "peeling"]): Method to solve the local decoding problem in each
            cluster. The 'peeling' method is only suitable for LDPC codes with point-like
            syndromes; 'inversion' can be applied to any parity check matrix, by default 'inversion'.
        bits_per_step (Optional[int]): Number of bits added to cluster in each step of the UFD algorithm.
            If not provided, this is set to the block length of the code.
    """

    def __init__(
        self,
        dem: stim.DetectorErrorModel,
        max_iter: int = 0,
        bp_method: Literal["product_sum", "minimum_sum"] = "minimum_sum",
        ms_scaling_factor: float = 1.0,
        schedule: Literal["parallel", "serial"] = "parallel",
        omp_thread_count: int = 1,
        random_schedule_seed: int = 0,
        serial_schedule_order: list[int] | None = None,
        uf_method: Literal["inversion", "peeling"] = "inversion",
        bits_per_step: int = 0,
    ):
        self._dem = dem
        dem_matrix = detector_error_model_to_check_matrices(
            dem, allow_undecomposed_hyperedges=True
        )
        self._observable_matrix = dem_matrix.observables_matrix

        self._decoder = LdpcBeliefFindDecoder(
            dem_matrix.check_matrix,
            error_channel=list(dem_matrix.priors),
            max_iter=max_iter,
            bp_method=bp_method,
            ms_scaling_factor=ms_scaling_factor,
            schedule=schedule,
            omp_thread_count=omp_thread_count,
            random_schedule_seed=random_schedule_seed,
            serial_schedule_order=serial_schedule_order,
            uf_method=uf_method,
            bits_per_step=bits_per_step,
        )

    def _decode(self, detector_bits: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
        estimated_error = self._decoder.decode(detector_bits)
        return estimated_error @ self._observable_matrix.T % 2


class BpLsdDecoder(BaseDecoder):
    """Belief propagation + localized statistics decoder (BP+LSD).

    Arguments match ldpc.BpLsdDecoder exactly, except a stim.DetectorErrorModel
    is expected in place of the parity check matrix and error channel priors.
    Defaults are used if not specified.

    Args:
        dem (stim.DetectorErrorModel): The detector error model describing the error structure.
        max_iter (int): The maximum number of iterations for the decoding algorithm,
            by default 0.
        bp_method (Literal["product_sum", "minimum_sum"]): The belief propagation method used,
            by default 'minimum_sum'.
        ms_scaling_factor (float): The scaling factor used in the minimum sum method,
            by default 1.0.
        schedule (Literal["parallel", "serial"]): The scheduling method used,
            by default 'parallel'.
        omp_thread_count (int): The number of OpenMP threads used for parallel
            decoding, by default 1.
        random_schedule_seed (int): Seed for random serial schedule order,
            by default 0.
        serial_schedule_order (Optional[list[int]]): A list of integers specifying the serial
            schedule order. Must be of length equal to the block length of the code,
            by default None.
        bits_per_step (Optional[int]): Specifies the number of bits added to the cluster in
            each step. If no value is provided, this is set to the block length of the code.
        lsd_order (int): The order of the LSD applied to each cluster. Must be
            greater than or equal to 0, by default 0.
        lsd_method (Literal["LSD_0", "LSD_E", "LSD_CS"]): The LSD method applied to each cluster,
            by default 'LSD_0'.
    """

    def __init__(
        self,
        dem: stim.DetectorErrorModel,
        max_iter: int = 0,
        bp_method: Literal["product_sum", "minimum_sum"] = "minimum_sum",
        ms_scaling_factor: float = 1.0,
        schedule: Literal["parallel", "serial"] = "parallel",
        omp_thread_count: int = 1,
        random_schedule_seed: int = 0,
        serial_schedule_order: list[int] | None = None,
        bits_per_step: int = 0,
        lsd_order: int = 0,
        lsd_method: Literal["LSD_0", "LSD_E", "LSD_CS"] = "LSD_0",
    ):
        self._dem = dem
        dem_matrix = detector_error_model_to_check_matrices(
            dem, allow_undecomposed_hyperedges=True
        )
        self._observable_matrix = dem_matrix.observables_matrix

        self._decoder = LdpcBpLsdDecoder(
            dem_matrix.check_matrix,
            error_channel=list(dem_matrix.priors),
            max_iter=max_iter,
            bp_method=bp_method,
            ms_scaling_factor=ms_scaling_factor,
            schedule=schedule,
            omp_thread_count=omp_thread_count,
            random_schedule_seed=random_schedule_seed,
            serial_schedule_order=serial_schedule_order,
            bits_per_step=bits_per_step,
            lsd_order=lsd_order,
            lsd_method=lsd_method,
        )

    def _decode(self, detector_bits: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
        estimated_error = self._decoder.decode(detector_bits)
        return estimated_error @ self._observable_matrix.T % 2


class BpOsdDecoder(BaseDecoder):
    """Belief propagation and Ordered Statistic Decoding (OSD) decoder for binary linear codes.

    Arguments match ldpc.BpOsdDecoder exactly, except a stim.DetectorErrorModel
    is expected in place of the parity check matrix and error channel priors.
    Defaults are used if not specified. The information below is taken straight from the
    original ldpc.BpOsdDecoder docstring.

    Args:
        dem (stim.DetectorErrorModel): The detector error model describing the error structure.
        max_iter (int): The maximum number of iterations for the decoding algorithm,
            by default 0.
        bp_method (Literal["product_sum", "minimum_sum"]): The belief propagation method used,
            by default 'minimum_sum'.
        ms_scaling_factor (float): The scaling factor used in the minimum sum method,
            by default 1.0.
        schedule (Literal["parallel", "serial"]): The scheduling method used,
            by default 'parallel'.
        omp_thread_count (int): The number of OpenMP threads used for parallel
            decoding, by default 1.
        serial_schedule_order (list[int] | None): A list of integers that specify the
            serial schedule order. Must be of length equal to the block length of the code,
            by default None.
        osd_method (Literal["OSD_0", "OSD_E", "OSD_CS"]): The OSD method used,
            by default 'OSD_0'.
        osd_order (int): The OSD order, by default 0.
    """

    def __init__(
        self,
        dem: stim.DetectorErrorModel,
        max_iter: int = 0,
        bp_method: Literal["product_sum", "minimum_sum"] = "minimum_sum",
        ms_scaling_factor: float = 1.0,
        schedule: Literal["parallel", "serial"] = "parallel",
        omp_thread_count: int = 1,
        random_serial_schedule: bool = False,
        serial_schedule_order: list[int] | None = None,
        osd_method: Literal["OSD_0", "OSD_E", "OSD_CS"] = "OSD_0",
        osd_order: int = 0,
    ):
        self._dem = dem
        dem_matrix = detector_error_model_to_check_matrices(
            dem, allow_undecomposed_hyperedges=True
        )
        self._observable_matrix = dem_matrix.observables_matrix

        self._decoder = LdpcBpOsdDecoder(
            dem_matrix.check_matrix,
            error_channel=list(dem_matrix.priors),
            max_iter=max_iter,
            bp_method=bp_method,
            ms_scaling_factor=ms_scaling_factor,
            schedule=schedule,
            omp_thread_count=omp_thread_count,
            serial_schedule_order=serial_schedule_order,
            osd_method=osd_method,
            osd_order=osd_order,
        )

    def _decode(self, detector_bits: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
        estimated_error = self._decoder.decode(detector_bits)
        return estimated_error @ self._observable_matrix.T % 2
