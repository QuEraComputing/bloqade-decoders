"""
Trivial two logical observable DEM
"""

import stim
import numpy as np

# For this super trivial case,
# if an error flips detectors D0 and D1, then the logical observable L0 is also flipped
# if an error flips detectors D1 and D2, then the logical observable L1 is also flipped
two_logical_dem_str = """
error(0.1) D0 D1 L0
error(0.2) D1 D2 L1
"""

two_logical_dem = stim.DetectorErrorModel(two_logical_dem_str)
# Two sets of syndromes, can pretend it was from two different rounds
two_logical_syndromes = np.array([[True, True, False], [False, True, True]])
two_logical_expected_decoded_obs = np.array([[True, False], [False, True]])
