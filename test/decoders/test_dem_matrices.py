import stim
import numpy as np
import pytest

from bloqade.decoders._decoders._dem_matrices import dem_to_check_matrices


@pytest.mark.parametrize(
    "task", ["surface_code:rotated_memory_x", "surface_code:rotated_memory_z"]
)
def test_matrices_reproduce_sampled_syndromes(task):
    """H e = detectors and O e = observables (mod 2) for sampled error vectors."""
    circuit = stim.Circuit.generated(
        task,
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.01,
        before_measure_flip_probability=0.01,
    )
    dem = circuit.detector_error_model(decompose_errors=False)
    m = dem_to_check_matrices(dem)
    check = m.check_matrix.toarray()
    observables = m.observables_matrix.toarray()
    # No mechanisms were merged, so columns map 1:1 to error instructions.
    assert check.shape[1] == dem.num_errors

    dets, obs, errors = dem.compile_sampler(seed=0).sample(
        shots=500, return_errors=True
    )
    assert errors is not None
    e = errors.astype(np.uint8)
    np.testing.assert_array_equal((e @ check.T) % 2, dets)
    np.testing.assert_array_equal((e @ observables.T) % 2, obs)
    probs = [
        i.args_copy()[0]
        for i in dem.flattened()
        if isinstance(i, stim.DemInstruction) and i.type == "error"
    ]
    np.testing.assert_allclose(m.priors, probs)


def test_identical_mechanisms_merge_as_xor_of_independent_events():
    p1, p2 = 0.1, 0.25
    dem = stim.DetectorErrorModel(f"error({p1}) D0 D1 L0\nerror({p2}) D0 D1 L0")
    m = dem_to_check_matrices(dem)
    assert m.check_matrix.toarray().shape == (2, 1)
    np.testing.assert_allclose(m.priors, [p1 * (1 - p2) + p2 * (1 - p1)])


def test_separators_are_xored():
    decomposed = dem_to_check_matrices(
        stim.DetectorErrorModel("error(0.1) D0 D1 ^ D1 D2 L0")
    )
    direct = dem_to_check_matrices(stim.DetectorErrorModel("error(0.1) D0 D2 L0"))
    np.testing.assert_array_equal(
        decomposed.check_matrix.toarray(), direct.check_matrix.toarray()
    )
    np.testing.assert_array_equal(
        decomposed.observables_matrix.toarray(), direct.observables_matrix.toarray()
    )


def test_shape_covers_detectors_and_observables_without_errors():
    dem = stim.DetectorErrorModel(
        "error(0.1) D0\nrepeat 2 {\n error(0.1) D0\n shift_detectors 1\n}\n"
        "detector D4\nlogical_observable L2"
    )
    m = dem_to_check_matrices(dem)
    assert m.check_matrix.toarray().shape[0] == dem.num_detectors == 7
    assert m.observables_matrix.toarray().shape[0] == dem.num_observables == 3
