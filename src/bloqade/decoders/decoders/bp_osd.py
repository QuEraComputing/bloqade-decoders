import stim
import numpy as np
from typing import Optional

from beliefmatching import detector_error_model_to_check_matrices
from ldpc.bposd_decoder import BpOsdDecoder as LdpcBpOsdDecoder

from .base import BaseDecoder


class BpOsdDecoder(BaseDecoder):
    """Belief propagation + ordered statistics decoder.

    Args:
        dem: The detector error model describing the error structure.
        max_iter: Maximum number of BP iterations.
        bp_method: BP method to use ('product_sum' or 'minimum_sum').
        ms_scaling_factor: Scaling factor used in the minimum sum method.
        schedule: Update schedule ('serial' or 'parallel').
        omp_thread_count: Number of OpenMP threads for parallel decoding.
        random_serial_schedule: Whether to use a random serial schedule order.
        serial_schedule_order: List specifying the serial schedule order.
        osd_method: OSD method ('osd_0', 'osd_e', 'osd_cs').
        osd_order: Order of the OSD post-processing.
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

    def _decode(self, detector_bits: np.ndarray) -> np.ndarray:
        return self._decoder.decode(detector_bits)
