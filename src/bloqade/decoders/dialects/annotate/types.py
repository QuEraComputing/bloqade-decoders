from enum import IntEnum
from dataclasses import dataclass

from kirin import types


class MeasurementResultValue(IntEnum):
    """Classify a physical measurement as zero, one, or atom loss."""

    Zero = 0
    One = 1
    Lost = 2


@dataclass
class MeasurementResult:
    """Represent a measurement outcome consumed by parity annotations."""

    value: MeasurementResultValue


class Detector:
    """Type marker returned when a detector parity is declared."""

    pass


class Observable:
    """Type marker returned when an observable parity is declared."""

    pass


# Kirin IR types
MeasurementResultType = types.PyClass(MeasurementResult)
DetectorType = types.PyClass(Detector)
ObservableType = types.PyClass(Observable)
