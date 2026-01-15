import stim
import numpy as np
from typing import Optional

from beliefmatching import detector_error_model_to_check_matrices
from ldpc.belief_find_decoder import BeliefFindDecoder as LdpcBeliefFindDecoder

from .base import BaseDecoder


class BeliefFindDecoder(BaseDecoder):
    """Belief propagation + union-find decoder.

    Belief-find first attempts to solve the decoding using belief propagation.
    If this fails, a union-find decoder is run as a post-processor using the
    output of BP to guide cluster growth.

    Args:
        dem: The detector error model describing the error structure.
        max_iter: Maximum number of BP iterations.
        bp_method: BP method to use ('product_sum' or 'minimum_sum').
        ms_scaling_factor: Scaling factor used in the minimum sum method.
        schedule: Update schedule ('serial' or 'parallel').
        omp_thread_count: Number of OpenMP threads for parallel decoding.
        random_schedule_seed: Seed for random serial schedule order.
        serial_schedule_order: List specifying the serial schedule order.
        uf_method: Union-find method ('peeling' or 'inversion').
        bits_per_step: Number of bits added to cluster in each UFD step.
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

    def _decode(self, detector_bits: np.ndarray) -> np.ndarray:
        return self._decoder.decode(detector_bits)
