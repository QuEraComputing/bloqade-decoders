"""
Repetition-code like DEMs
"""

import stim
import numpy as np

space_error_dem = stim.DetectorErrorModel("""
    error(0.1) D0
    error(0.1) D0 D1
    error(0.1) D1 L0
""")

space_error_syndromes = np.array([[False, False], [True, False], [False, True]])
expected_space_error_decoded_obs = np.array([[False], [False], [True]])

time_error_dem = stim.DetectorErrorModel("""
    error(0.1) D0 D2
    error(0.1) D1 D3
    error(0.1) D2 D4
    error(0.1) D3 D5
    error(0.1) D5 L0
""")

time_error_syndromes = np.array(
    [[False] * 6, [True, False, True] + 3 * [False], [False] * 5 + [True]]
)
expected_time_error_decoded_obs = np.array([[False], [False], [True]])
