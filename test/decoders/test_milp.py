"""Tests for the PuLP-based multi-solver MILP decoder."""

import math

import stim
import numpy as np
import pytest
import sinter

from bloqade.decoders import MILPDecoder, GurobiDecoder
from bloqade.decoders.sinter_interface import SinterMILPDecoder

from .conftest import pack_dets, simple_dem, unpack_obs, repetition_circuit
from .test_mle import regular_dem, regular_samples

# Solver backends MILPDecoder claims to support. Tests only run for solvers
# actually available in the environment (e.g. CPLEX/COPT need a license).
SUPPORTED_SOLVERS = [
    "HiGHS",
    "CPLEX_PY",
    "CPLEX_CMD",
    "COPT",
    "COPT_CMD",
    "GUROBI",
    "GUROBI_CMD",
]


def available_solvers() -> list[str]:
    import pulp

    available = set(pulp.listSolvers(onlyAvailable=True))
    return [name for name in SUPPORTED_SOLVERS if name in available]


SOLVERS = available_solvers()


@pytest.mark.parametrize("solver", SOLVERS)
def test_regular(solver):
    dem = regular_dem()
    det_shots, obs_shots = regular_samples()
    decoder = MILPDecoder(dem, solver=solver)
    result = decoder.decode(det_shots)
    assert (obs_shots == result).all()


@pytest.mark.parametrize("solver", SOLVERS)
def test_hyper(solver):
    dem = stim.DetectorErrorModel("""
        error(0.1) D9 D0 D1 L0
        error(0.1) D0 D1
        error(0.1) D1 D2
        error(0.1) D2 D3
        error(0.1) D3 D4
        error(0.1) D4 D5
        error(0.1) D5 D6
        error(0.1) D6 D7
        error(0.1) D7 D8
        error(0.1) D8 D9
        """)
    det_shots = np.array(
        [[1, 1, 0, 0, 0, 0, 0, 0, 0, 1], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]],
        bool,
    )
    obs_shots = np.array([[1], [0]], bool)
    decoder = MILPDecoder(dem, solver=solver)
    result = decoder.decode(det_shots)
    assert (obs_shots == result).all()


@pytest.mark.parametrize("solver", SOLVERS)
def test_no_error_syndrome(solver):
    dem = regular_dem()
    decoder = MILPDecoder(dem, solver=solver)
    det_shots = np.zeros((1, 10), dtype=bool)
    result = decoder.decode(det_shots)
    assert np.array_equal(result, np.array([[False]]))


@pytest.mark.parametrize("solver", SOLVERS)
def test_single_shot_decode(solver):
    dem = regular_dem()
    det_shots = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=bool)
    decoder = MILPDecoder(dem, solver=solver)
    result = decoder.decode(det_shots)
    assert result.ndim == 1
    assert np.array_equal(result, np.array([True]))


def test_default_solver_is_available():
    dem = regular_dem()
    det_shots, obs_shots = regular_samples()
    decoder = MILPDecoder(dem)
    result = decoder.decode(det_shots)
    assert (obs_shots == result).all()


def test_unknown_solver_rejected():
    dem = regular_dem()
    with pytest.raises(Exception):  # noqa: B017 - pulp raises its own error type
        MILPDecoder(dem, solver="NO_SUCH_SOLVER")


def test_non_copyable_solver_instance_rejected():
    import threading

    import pulp

    class UnpicklableSolver(pulp.HiGHS):
        def __init__(self):
            super().__init__(msg=False)
            self._lock = threading.Lock()  # cannot be pickled, like COPT handles

    with pytest.raises(ValueError, match="deep-copied"):
        MILPDecoder(regular_dem(), solver=UnpicklableSolver())


def test_solver_factory_builds_a_fresh_solver_per_shot():
    import pulp

    dem = regular_dem()
    det_shots, obs_shots = regular_samples()
    created: list[pulp.LpSolver] = []

    def factory() -> pulp.LpSolver:
        solver = pulp.getSolver("HiGHS", msg=False)
        created.append(solver)
        return solver

    decoder = MILPDecoder(dem, solver=factory)
    result = decoder.decode(det_shots)

    assert (obs_shots == result).all()
    assert len(created) == len(det_shots)
    assert len({id(solver) for solver in created}) == len(created)


def test_decode_raises_on_non_optimal_status():
    import pulp

    dem = stim.DetectorErrorModel("""
        detector D0
        error(0) L0
        """)
    decoder = MILPDecoder(dem)

    with pytest.raises(pulp.PulpSolverError, match="no optimal solution"):
        decoder.decode(np.array([True], dtype=bool))


def test_separator_targets_rejected():
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 ^ D1 L0
        """)
    with pytest.raises(ValueError, match="separator"):
        MILPDecoder(dem)


def test_separator_targets_rejected_inside_repeat_block():
    dem = stim.DetectorErrorModel("""
        repeat 2 {
            error(0.1) D0 ^ D1 L0
        }
        """)
    with pytest.raises(ValueError, match="separator"):
        MILPDecoder(dem)


def test_short_syndrome_rejected():
    decoder = MILPDecoder(regular_dem())
    with pytest.raises(ValueError, match="detector"):
        decoder.decode(np.array([True], dtype=bool))


def test_msg_option_accepted_for_named_solver():
    dem = regular_dem()
    det_shots, obs_shots = regular_samples()
    decoder = MILPDecoder(dem, solver="HiGHS", msg=False)
    result = decoder.decode(det_shots)
    assert (obs_shots == result).all()


def test_decode_confidence_does_not_accumulate_diff_variables():
    dem = regular_dem()
    det_shots, _ = regular_samples()
    decoder = MILPDecoder(dem)

    decoder.decode_confidence(det_shots)
    num_variables = len(decoder._prob._variables)
    num_modified_constraints = len(decoder._prob.modifiedConstraints)
    decoder.decode_confidence(det_shots)

    assert len(decoder._prob._variables) == num_variables
    assert len(decoder._prob.modifiedConstraints) == num_modified_constraints


def test_milp_decoder_can_instantiate_without_training():
    dem = regular_dem()
    det_shots, obs_shots = regular_samples()

    decoder = MILPDecoder.instantiate(dem)
    result = decoder.decode(det_shots)

    np.testing.assert_array_equal(result, obs_shots)


def test_prob_zero_error_skipped():
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 D1 L0
        error(0.0) D1 D2 L0
        error(0.05) D0 D2
        """)
    decoder = MILPDecoder(dem)
    result = decoder.decode(np.array([[0, 1, 1]], dtype=bool))
    assert result.shape == (1, 1)


def test_prob_one_error_pre_applied():
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 D1 L0
        error(1.0) D1 D2 L0
        error(0.05) D0 D2
        """)
    decoder = MILPDecoder(dem)
    result = decoder.decode(np.array([[0, 1, 1]], dtype=bool))
    assert result.shape == (1, 1)
    assert result[0, 0]


# --- decode_confidence tests ---


@pytest.mark.parametrize("solver", SOLVERS)
def test_decode_confidence(solver):
    dem = regular_dem()
    det_shots, obs_shots = regular_samples()
    decoder = MILPDecoder(dem, solver=solver)

    result, confidence = decoder.decode_confidence(det_shots)

    np.testing.assert_array_equal(result, obs_shots)
    expected = np.tanh(4 * np.log(9))
    np.testing.assert_allclose(confidence, np.full(2, expected))


@pytest.mark.parametrize("solver", SOLVERS)
def test_single_shot_decode_confidence(solver):
    dem = regular_dem()
    det_shots = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=bool)
    decoder = MILPDecoder(dem, solver=solver)

    result, confidence = decoder.decode_confidence(det_shots)

    assert np.isclose(confidence, np.tanh(4 * np.log(9)))
    np.testing.assert_array_equal(result, np.array([True]))


@pytest.mark.parametrize("solver", SOLVERS)
def test_equal_likelihood_alternatives_have_zero_confidence(solver):
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 L0
        error(0.1) D0
        """)
    decoder = MILPDecoder(dem, solver=solver)

    result, confidence = decoder.decode_confidence(np.array([True], dtype=bool))

    assert result.shape == (1,)
    assert confidence == pytest.approx(0.0)


@pytest.mark.parametrize("solver", SOLVERS)
def test_finite_logical_gap_has_normalized_confidence(solver):
    dem = stim.DetectorErrorModel("""
        error(0.75) D0 L0
        error(0.25) D0
        """)
    decoder = MILPDecoder(dem, solver=solver)

    result, confidence = decoder.decode_confidence(np.array([True], dtype=bool))

    np.testing.assert_array_equal(result, np.array([True]))
    assert confidence == pytest.approx(0.8)


@pytest.mark.parametrize("solver", SOLVERS)
def test_nonoptimal_status_has_zero_confidence(solver):
    dem = stim.DetectorErrorModel("""
        detector D0
        error(0) L0
        """)
    decoder = MILPDecoder(dem, solver=solver)

    result, confidence = decoder.decode_confidence(
        np.array([[True], [False]], dtype=bool)
    )

    np.testing.assert_array_equal(result, np.array([[False], [False]]))
    assert isinstance(confidence, np.ndarray)
    assert confidence[0] == 0.0
    assert confidence[1] == 1.0

    result, confidence = decoder.decode_confidence(np.array([True], dtype=bool))

    np.testing.assert_array_equal(result, np.array([False]))
    assert confidence == 0.0


def test_nonoptimal_alternative_solve_returns_best_with_zero_confidence(monkeypatch):
    decoder = MILPDecoder(stim.DetectorErrorModel("error(0.1) D0 L0"))
    best = decoder._ConfidenceSolveResult(
        error=np.array([True]),
        logical=np.array([True]),
        objective=1.0,
    )

    def solve(detector_shot, *, verbose=False, forbidden_logical=None):
        if forbidden_logical is None:
            return best, True
        return None, False

    monkeypatch.setattr(decoder, "_solve_single_shot_for_confidence", solve)

    result, confidence = decoder.decode_confidence(np.array([[True]], dtype=bool))

    np.testing.assert_array_equal(result, np.array([[True]]))
    np.testing.assert_array_equal(confidence, np.array([0.0]))

    result, confidence = decoder.decode_confidence(np.array([True], dtype=bool))

    np.testing.assert_array_equal(result, np.array([True]))
    assert confidence == 0.0


@pytest.mark.parametrize("solver", SOLVERS)
def test_decode_confidence_includes_prob_one_error_contributions(solver):
    dem = stim.DetectorErrorModel("""
        error(1.0) D0 L0
        error(0.1) D1 L0
        """)
    decoder = MILPDecoder(dem, solver=solver)

    result, confidence = decoder.decode_confidence(
        np.array([[1, 0], [1, 1]], dtype=bool)
    )

    np.testing.assert_array_equal(result, np.array([[True], [False]]))
    assert isinstance(confidence, np.ndarray)
    np.testing.assert_array_equal(confidence, np.ones(2))


# --- SinterMILPDecoder tests ---


def test_sinter_milp_is_sinter_decoder():
    decoder = SinterMILPDecoder()
    assert isinstance(decoder, sinter.Decoder)


@pytest.mark.parametrize("solver", SOLVERS)
def test_sinter_milp_compile_returns_compiled_decoder(solver):
    dem = simple_dem()
    decoder = SinterMILPDecoder(solver=solver)
    compiled = decoder.compile_decoder_for_dem(dem=dem)
    assert isinstance(compiled, sinter.CompiledDecoder)


@pytest.mark.parametrize("solver", SOLVERS)
def test_sinter_milp_decode_shape_and_dtype(solver):
    dem = simple_dem()
    decoder = SinterMILPDecoder(solver=solver)
    compiled = decoder.compile_decoder_for_dem(dem=dem)

    det_shots = np.array([[1, 0], [0, 1], [0, 0]], dtype=bool)
    packed_dets = pack_dets(det_shots)

    result = compiled.decode_shots_bit_packed(
        bit_packed_detection_event_data=packed_dets
    )

    num_obs_bytes = math.ceil(dem.num_observables / 8)
    assert result.dtype == np.uint8
    assert result.shape == (3, num_obs_bytes)


@pytest.mark.parametrize("solver", SOLVERS)
def test_sinter_milp_decode_correctness(solver):
    dem = simple_dem()
    decoder = SinterMILPDecoder(solver=solver)
    compiled = decoder.compile_decoder_for_dem(dem=dem)

    det_shots = np.array([[1, 0], [0, 1], [0, 0]], dtype=bool)
    packed_dets = pack_dets(det_shots)

    result = compiled.decode_shots_bit_packed(
        bit_packed_detection_event_data=packed_dets
    )
    obs_predictions = unpack_obs(result, dem.num_observables)

    expected = np.array([[True], [True], [False]])
    assert np.array_equal(obs_predictions, expected)


@pytest.mark.parametrize("solver", SOLVERS)
def test_sinter_milp_no_error_syndrome(solver):
    dem = simple_dem()
    decoder = SinterMILPDecoder(solver=solver)
    compiled = decoder.compile_decoder_for_dem(dem=dem)

    det_shots = np.zeros((1, 2), dtype=bool)
    packed_dets = pack_dets(det_shots)

    result = compiled.decode_shots_bit_packed(
        bit_packed_detection_event_data=packed_dets
    )
    obs_predictions = unpack_obs(result, dem.num_observables)

    assert np.array_equal(obs_predictions, np.array([[False]]))


@pytest.mark.slow
def test_sinter_collect_milp():
    circuit = repetition_circuit()
    tasks = [
        sinter.Task(
            circuit=circuit,
            decoder="milp",
            json_metadata={"d": 3},
        ),
    ]
    stats = sinter.collect(
        num_workers=1,
        tasks=tasks,
        custom_decoders={"milp": SinterMILPDecoder()},
        max_shots=100,
    )
    assert len(stats) == 1
    assert stats[0].shots == 100
    assert stats[0].errors <= stats[0].shots
    assert stats[0].errors < 50


# --- Cross-solver consistency with GurobiDecoder ---


def consistency_dems() -> dict[str, stim.DetectorErrorModel]:
    return {
        "regular": regular_dem(),
        "hyper": stim.DetectorErrorModel("""
            error(0.1) D9 D0 D1 L0
            error(0.1) D0 D1
            error(0.1) D1 D2
            error(0.1) D2 D3
            error(0.1) D3 D4
            error(0.1) D4 D5
            error(0.1) D5 D6
            error(0.1) D6 D7
            error(0.1) D7 D8
            error(0.1) D8 D9
            """),
        "multi_observable": stim.DetectorErrorModel("""
            error(0.1) D0 L0
            error(0.05) D0 D1
            error(0.2) D1 L1
            error(0.1) D1 L0 L1
            """),
        "prob_one": stim.DetectorErrorModel("""
            error(0.1) D0 D1 L0
            error(1.0) D1 D2 L0
            error(0.05) D0 D2
            """),
    }


def sample_syndromes(dem: stim.DetectorErrorModel, num_shots: int = 20) -> np.ndarray:
    sampler = dem.compile_sampler()
    det, _, _ = sampler.sample(num_shots, bit_packed=False)
    return det


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("dem_name", list(consistency_dems()))
def test_decode_consistent_with_gurobi(solver, dem_name):
    dem = consistency_dems()[dem_name]
    det_shots = sample_syndromes(dem)

    expected = GurobiDecoder(dem).decode(det_shots)
    result = MILPDecoder(dem, solver=solver).decode(det_shots)

    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("dem_name", list(consistency_dems()))
def test_decode_confidence_consistent_with_gurobi(solver, dem_name):
    dem = consistency_dems()[dem_name]
    det_shots = sample_syndromes(dem)

    expected_obs, expected_conf = GurobiDecoder(dem).decode_confidence(det_shots)
    result_obs, result_conf = MILPDecoder(dem, solver=solver).decode_confidence(
        det_shots
    )

    np.testing.assert_array_equal(result_obs, expected_obs)
    np.testing.assert_allclose(result_conf, expected_conf, rtol=1e-6, atol=1e-9)
