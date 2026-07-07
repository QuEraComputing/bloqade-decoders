from enum import IntEnum
from dataclasses import dataclass

from kirin import types


class MeasurementResultValue(IntEnum):
    """Possible measurement outcomes represented in decoder annotations."""

    Zero = 0
    One = 1
    Lost = 2


@dataclass
class MeasurementResult:
    """A typed measurement result value for annotation statements."""

    value: MeasurementResultValue


class Detector:
    """Marker type for detector annotations."""

    pass


class Observable:
    """Marker type for observable annotations."""

    pass


# Kirin IR types
MeasurementResultType = types.PyClass(MeasurementResult)
DetectorType = types.PyClass(Detector)
ObservableType = types.PyClass(Observable)
