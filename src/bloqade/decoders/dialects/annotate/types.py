from enum import IntEnum
from typing import Any
from dataclasses import dataclass

from kirin import types


def _as_measurement_value(other: Any) -> "MeasurementResultValue | None":
    if isinstance(other, MeasurementResult):
        return other.value
    if isinstance(other, MeasurementResultValue):
        return other
    return None


def _xor_measurement_values(
    lhs: "MeasurementResultValue", rhs: "MeasurementResultValue"
) -> "MeasurementResultValue":
    """XOR two measurement values with Lost absorbing.

    If either operand is `MeasurementResultValue.Lost`, the result is
    ``Lost``. Otherwise the result is the bitwise XOR of ``Zero``/``One``.
    """
    if lhs is MeasurementResultValue.Lost or rhs is MeasurementResultValue.Lost:
        return MeasurementResultValue.Lost
    return MeasurementResultValue(int(lhs) ^ int(rhs))


class MeasurementResultValue(IntEnum):
    """Classify a physical measurement as zero, one, or atom loss.

    Supports ``^`` between measurement values (and
    `MeasurementResult`). Semantics: if either operand is ``Lost``,
    the result is ``Lost``; otherwise the result is the bitwise XOR of
    ``Zero``/``One``. XORing with a non-measurement type raises ``TypeError``.
    """

    Zero = 0
    One = 1
    Lost = 2

    def __xor__(self, other: Any) -> "MeasurementResultValue":
        """Return Lost-aware XOR of this value with ``other``."""
        rhs = _as_measurement_value(other)
        if rhs is None:
            # Do not return NotImplemented: IntEnum would fall back to int XOR.
            raise TypeError(
                f"unsupported operand type(s) for ^: "
                f"'MeasurementResultValue' and '{type(other).__name__}'"
            )
        return _xor_measurement_values(self, rhs)

    def __rxor__(self, other: Any) -> "MeasurementResultValue":
        """Return Lost-aware XOR with this value on the right-hand side."""
        lhs = _as_measurement_value(other)
        if lhs is None:
            raise TypeError(
                f"unsupported operand type(s) for ^: "
                f"'{type(other).__name__}' and 'MeasurementResultValue'"
            )
        return _xor_measurement_values(lhs, self)


@dataclass
class MeasurementResult:
    """Represent a measurement outcome consumed by parity annotations.

    Supports ``^`` with another `MeasurementResult` or
    `MeasurementResultValue`. If either operand is ``Lost``, the
    result is ``Lost``; otherwise the result is the bitwise XOR of the
    underlying ``Zero``/``One`` values.
    """

    value: MeasurementResultValue

    def __xor__(self, other: Any) -> "MeasurementResult":
        """Return Lost-aware XOR of this result with ``other``."""
        rhs = _as_measurement_value(other)
        if rhs is None:
            return NotImplemented
        return MeasurementResult(_xor_measurement_values(self.value, rhs))

    def __rxor__(self, other: Any) -> "MeasurementResult":
        """Return Lost-aware XOR with this result on the right-hand side."""
        lhs = _as_measurement_value(other)
        if lhs is None:
            return NotImplemented
        return MeasurementResult(_xor_measurement_values(lhs, self.value))


class Detector:
    """Type marker returned when a detector parity is declared."""


class Observable:
    """Type marker returned when an observable parity is declared."""


# Kirin IR types
MeasurementResultType = types.PyClass(MeasurementResult)
DetectorType = types.PyClass(Detector)
ObservableType = types.PyClass(Observable)
