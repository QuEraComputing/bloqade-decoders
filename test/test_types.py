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
