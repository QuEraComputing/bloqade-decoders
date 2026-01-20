from typing import Optional

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
        max_iter (Optional[int]): Maximum number of BP iterations, by default 0.
        bp_method (Optional[str]): BP method to use. Must be one of
            {'product_sum', 'minimum_sum'}, by default 'minimum_sum'.
        ms_scaling_factor (Optional[float]): Scaling factor used in the minimum sum method,
            by default 1.0.
        schedule (Optional[str]): Update schedule. Must be one of {'parallel', 'serial'},
            by default 'parallel'.
        omp_thread_count (Optional[int]): Number of OpenMP threads for parallel decoding,
            by default 1.
        random_schedule_seed (Optional[int]): Seed for random serial schedule order,
            by default 0.
        serial_schedule_order (Optional[list[int]]): List specifying the serial schedule
            order. Must be of length equal to the block length of the code, by default None.
        uf_method (Optional[str]): Method to solve the local decoding problem in each
            cluster. Must be one of {'inversion', 'peeling'}. The 'peeling' method is only
            suitable for LDPC codes with point-like syndromes; 'inversion' can be applied
            to any parity check matrix, by default 'inversion'.
        bits_per_step (Optional[int]): Number of bits added to cluster in each step of the UFD algorithm.
            If not provided, this is set to the block length of the code.
    """

    def __init__(
        self,
        dem: stim.DetectorErrorModel,
        max_iter: Optional[int] = None,
        bp_method: Optional[str] = None,
        ms_scaling_factor: Optional[float] = None,
        schedule: Optional[str] = None,
        omp_thread_count: Optional[int] = None,
        random_schedule_seed: Optional[int] = None,
        serial_schedule_order: Optional[list[int]] = None,
        uf_method: Optional[str] = None,
        bits_per_step: Optional[int] = None,
    ):
        self._dem = dem
        dem_matrix = detector_error_model_to_check_matrices(dem)
        self._observable_matrix = dem_matrix.observables_matrix

        decoder_kwargs: dict = {}
        if max_iter is not None:
            decoder_kwargs["max_iter"] = max_iter
        if bp_method is not None:
            decoder_kwargs["bp_method"] = bp_method
        if ms_scaling_factor is not None:
            decoder_kwargs["ms_scaling_factor"] = ms_scaling_factor
        if schedule is not None:
            decoder_kwargs["schedule"] = schedule
        if omp_thread_count is not None:
            decoder_kwargs["omp_thread_count"] = omp_thread_count
        if random_schedule_seed is not None:
            decoder_kwargs["random_schedule_seed"] = random_schedule_seed
        if serial_schedule_order is not None:
            decoder_kwargs["serial_schedule_order"] = serial_schedule_order
        if uf_method is not None:
            decoder_kwargs["uf_method"] = uf_method
        if bits_per_step is not None:
            decoder_kwargs["bits_per_step"] = bits_per_step

        self._decoder = LdpcBeliefFindDecoder(
            dem_matrix.check_matrix,
            error_channel=list(dem_matrix.priors),
            **decoder_kwargs,
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
        max_iter (Optional[int]): The maximum number of iterations for the decoding algorithm,
            by default 0.
        bp_method (Optional[str]): The belief propagation method used. Must be one of
            {'product_sum', 'minimum_sum'}, by default 'minimum_sum'.
        ms_scaling_factor (Optional[float]): The scaling factor used in the minimum sum method,
            by default 1.0.
        schedule (Optional[str]): The scheduling method used. Must be one of
            {'parallel', 'serial'}, by default 'parallel'.
        omp_thread_count (Optional[int]): The number of OpenMP threads used for parallel
            decoding, by default 1.
        random_schedule_seed (Optional[int]): Seed for random serial schedule order,
            by default 0.
        serial_schedule_order (Optional[list[int]]): A list of integers specifying the serial
            schedule order. Must be of length equal to the block length of the code,
            by default None.
        bits_per_step (Optional[int]): Specifies the number of bits added to the cluster in
            each step. If no value is provided, this is set to the block length of the code.
        lsd_order (Optional[int]): The order of the LSD applied to each cluster. Must be
            greater than or equal to 0, by default 0.
        lsd_method (Optional[str]): The LSD method applied to each cluster. Must be one of
            {'LSD_0', 'LSD_E', 'LSD_CS'}, by default 'LSD_0'.
    """

    def __init__(
        self,
        dem: stim.DetectorErrorModel,
        max_iter: Optional[int] = None,
        bp_method: Optional[str] = None,
        ms_scaling_factor: Optional[float] = None,
        schedule: Optional[str] = None,
        omp_thread_count: Optional[int] = None,
        random_schedule_seed: Optional[int] = None,
        serial_schedule_order: Optional[list[int]] = None,
        bits_per_step: Optional[int] = None,
        lsd_order: Optional[int] = None,
        lsd_method: Optional[str] = None,
    ):
        self._dem = dem
        dem_matrix = detector_error_model_to_check_matrices(dem)
        self._observable_matrix = dem_matrix.observables_matrix

        decoder_kwargs: dict = {}
        if max_iter is not None:
            decoder_kwargs["max_iter"] = max_iter
        if bp_method is not None:
            decoder_kwargs["bp_method"] = bp_method
        if ms_scaling_factor is not None:
            decoder_kwargs["ms_scaling_factor"] = ms_scaling_factor
        if schedule is not None:
            decoder_kwargs["schedule"] = schedule
        if omp_thread_count is not None:
            decoder_kwargs["omp_thread_count"] = omp_thread_count
        if random_schedule_seed is not None:
            decoder_kwargs["random_schedule_seed"] = random_schedule_seed
        if serial_schedule_order is not None:
            decoder_kwargs["serial_schedule_order"] = serial_schedule_order
        if bits_per_step is not None:
            decoder_kwargs["bits_per_step"] = bits_per_step
        if lsd_order is not None:
            decoder_kwargs["lsd_order"] = lsd_order
        if lsd_method is not None:
            decoder_kwargs["lsd_method"] = lsd_method

        self._decoder = LdpcBpLsdDecoder(
            dem_matrix.check_matrix,
            error_channel=list(dem_matrix.priors),
            **decoder_kwargs,
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
        max_iter (Optional[int]): The maximum number of iterations for the decoding algorithm,
            by default 0.
        bp_method (Optional[str]): The belief propagation method used. Must be one of
            {'product_sum', 'minimum_sum'}, by default 'minimum_sum'.
        ms_scaling_factor (Optional[float]): The scaling factor used in the minimum sum method,
            by default 1.0.
        schedule (Optional[str]): The scheduling method used. Must be one of
            {'parallel', 'serial'}, by default 'parallel'.
        omp_thread_count (Optional[int]): The number of OpenMP threads used for parallel
            decoding, by default 1.
        random_serial_schedule (Optional[int]): Whether to use a random serial schedule order,
            by default False.
        serial_schedule_order (Optional[List[int]]): A list of integers that specify the
            serial schedule order. Must be of length equal to the block length of the code,
            by default None.
        osd_method (str): The OSD method used. Must be one of {'OSD_0', 'OSD_E', 'OSD_CS'}.
        osd_order (int): The OSD order, by default 0.
    """

    def __init__(
        self,
        dem: stim.DetectorErrorModel,
        bp_method: Optional[str] = None,
        max_iter: Optional[int] = None,
        ms_scaling_factor: Optional[float] = None,
        schedule: Optional[str] = None,
        omp_thread_count: Optional[int] = None,
        random_serial_schedule: Optional[bool] = None,
        serial_schedule_order: Optional[list[int]] = None,
        osd_method: Optional[str] = None,
        osd_order: Optional[int] = None,
    ):
        self._dem = dem
        dem_matrix = detector_error_model_to_check_matrices(dem)
        self._observable_matrix = dem_matrix.observables_matrix

        decoder_kwargs: dict = {}
        if bp_method is not None:
            decoder_kwargs["bp_method"] = bp_method
        if max_iter is not None:
            decoder_kwargs["max_iter"] = max_iter
        if ms_scaling_factor is not None:
            decoder_kwargs["ms_scaling_factor"] = ms_scaling_factor
        if schedule is not None:
            decoder_kwargs["schedule"] = schedule
        if omp_thread_count is not None:
            decoder_kwargs["omp_thread_count"] = omp_thread_count
        if random_serial_schedule is not None:
            decoder_kwargs["random_serial_schedule"] = random_serial_schedule
        if serial_schedule_order is not None:
            decoder_kwargs["serial_schedule_order"] = serial_schedule_order
        if osd_method is not None:
            decoder_kwargs["osd_method"] = osd_method
        if osd_order is not None:
            decoder_kwargs["osd_order"] = osd_order

        self._decoder = LdpcBpOsdDecoder(
            dem_matrix.check_matrix,
            error_channel=list(dem_matrix.priors),
            **decoder_kwargs,
        )

    def _decode(self, detector_bits: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
        estimated_error = self._decoder.decode(detector_bits)
        return estimated_error @ self._observable_matrix.T % 2
