import pytest

from bloqade.decoders.dialects.annotate.types import (
    MeasurementResult,
    MeasurementResultValue,
)

Zero = MeasurementResultValue.Zero
One = MeasurementResultValue.One
Lost = MeasurementResultValue.Lost


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


@pytest.mark.parametrize(
    ("lhs", "rhs", "expected"),
    [
        (Zero, Zero, Zero),
        (Zero, One, One),
        (One, Zero, One),
        (One, One, Zero),
        (Lost, Zero, Lost),
        (Lost, One, Lost),
        (Zero, Lost, Lost),
        (One, Lost, Lost),
        (Lost, Lost, Lost),
    ],
)
def test_measurement_value_xor(lhs, rhs, expected):
    assert lhs ^ rhs is expected


@pytest.mark.parametrize(
    ("lhs", "rhs", "expected"),
    [
        (Zero, Zero, Zero),
        (Zero, One, One),
        (One, One, Zero),
        (Lost, One, Lost),
        (One, Lost, Lost),
    ],
)
def test_measurement_result_xor(lhs, rhs, expected):
    result = MeasurementResult(lhs) ^ MeasurementResult(rhs)
    assert isinstance(result, MeasurementResult)
    assert result.value is expected


def test_mixed_measurement_result_and_value_xor():
    assert MeasurementResult(One) ^ Zero == MeasurementResult(One)
    assert One ^ MeasurementResult(One) is Zero
    assert MeasurementResult(Lost) ^ One == MeasurementResult(Lost)
    assert Lost ^ MeasurementResult(Zero) is Lost


def test_measurement_result_xor_rejects_unrelated_types():
    with pytest.raises(TypeError):
        MeasurementResult(Zero) ^ 1
    with pytest.raises(TypeError):
        MeasurementResult(Zero) ^ "0"


def test_measurement_value_xor_rejects_int_fallback():
    with pytest.raises(TypeError):
        One ^ 1
    with pytest.raises(TypeError):
        1 ^ One
    with pytest.raises(TypeError):
        Lost ^ 0
    with pytest.raises(TypeError):
        One ^ True


def test_measurement_result_rxor_notimplemented_path():
    with pytest.raises(TypeError):
        object() ^ MeasurementResult(Zero)


def test_measurement_result_rxor_happy_path():
    result = MeasurementResult(Zero).__rxor__(One)
    assert result == MeasurementResult(One)


def test_measurement_value_rxor_happy_path():
    assert Zero.__rxor__(One) is One
