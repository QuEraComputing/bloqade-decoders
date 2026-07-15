from bloqade.decoders.dialects.annotate.types import (
    MeasurementResult,
    MeasurementResultValue,
)


def test_measurement_result_init():
    result = MeasurementResult(value=MeasurementResultValue.Zero)
    assert result.value == MeasurementResultValue.Zero

    result = MeasurementResult(value=MeasurementResultValue.One)
    assert result.value == MeasurementResultValue.One

    result = MeasurementResult(value=MeasurementResultValue.Lost)
    assert result.value == MeasurementResultValue.Lost

def test_measurement_result_value_repr():
    assert repr(MeasurementResultValue.Zero) == "0"
    assert repr(MeasurementResultValue.One) == "1"
    assert repr(MeasurementResultValue.Lost) == "L"
    assert str(MeasurementResultValue.Lost) == "L"


def test_measurement_result_value_symbol_covers_all_members():
    assert {member.symbol for member in MeasurementResultValue} == {"0", "1", "L"}


def test_measurement_result_repr():
    assert repr(MeasurementResult(value=MeasurementResultValue.Zero)) == "0"
    assert repr(MeasurementResult(value=MeasurementResultValue.One)) == "1"
    assert repr(MeasurementResult(value=MeasurementResultValue.Lost)) == "L"


def test_measurement_result_repr_in_containers():
    results = [
        MeasurementResult(value=MeasurementResultValue.Zero),
        MeasurementResult(value=MeasurementResultValue.One),
        MeasurementResult(value=MeasurementResultValue.Lost),
    ]
    assert repr(results) == "[0, 1, L]"


def test_measurement_result_repr_bulk_shots():
    shot_values = [
        [0, 1, 1, 0, 1, 0, 0, 1],
        [1, 1, 0, 0, 2, 0, 1, 0],
        [0, 0, 1, 1, 0, 2, 2, 1],
    ]
    shots = [
        [MeasurementResult(value=MeasurementResultValue(v)) for v in shot]
        for shot in shot_values
    ]
    assert repr(shots) == (
        "[[0, 1, 1, 0, 1, 0, 0, 1],"
        " [1, 1, 0, 0, L, 0, 1, 0],"
        " [0, 0, 1, 1, 0, L, L, 1]]"
    )
