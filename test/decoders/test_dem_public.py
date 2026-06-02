import numpy as np

from bloqade.decoders.dem import matrix_to_dem, make_layout_only_dem


def test_dem_helpers_create_layout_only_and_matrix_dems():
    layout_dem = make_layout_only_dem(num_detectors=2, num_observables=1)
    assert layout_dem.num_detectors == 2
    assert layout_dem.num_observables == 1

    dem = matrix_to_dem(
        check_matrix=np.array([[1, 0], [0, 1]], dtype=np.uint8),
        observables_matrix=np.array([[1, 0]], dtype=np.uint8),
        priors=np.array([0.25, 0.5], dtype=np.float64),
    )

    assert dem.num_detectors == 2
    assert dem.num_observables == 1
    assert "D0 L0" in str(dem)
