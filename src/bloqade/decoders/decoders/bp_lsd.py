import stim
import numpy as np
from typing import Optional

from beliefmatching import detector_error_model_to_check_matrices
from ldpc.bplsd_decoder import BpLsdDecoder as LdpcBpLsdDecoder

from .base import BaseDecoder


class BpLsdDecoder(BaseDecoder):
    """Belief propagation + localized statistics decoder.

    Arguments match ldpc.BpLsdDecoder; defaults are used if not specified.
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
        if lsd_method is not None:
            decoder_kwargs["lsd_method"] = lsd_method
        if lsd_order is not None:
            decoder_kwargs["lsd_order"] = lsd_order

        self._decoder = LdpcBpLsdDecoder(
            dem_matrix.check_matrix,
            error_channel=list(dem_matrix.priors),
            **decoder_kwargs,
        )

    def _decode(self, detector_bits: np.ndarray) -> np.ndarray:
        return self._decoder.decode(detector_bits)
