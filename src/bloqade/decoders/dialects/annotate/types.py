from enum import IntEnum
from dataclasses import dataclass

from kirin import types


class MeasurementResultValue(IntEnum):
    Zero = 0
    One = 1
    Lost = 2


@dataclass
class MeasurementResult:
    value: MeasurementResultValue


class Detector:
    pass


class Observable:
    pass


# Kirin IR types
MeasurementResultType = types.PyClass(MeasurementResult)
DetectorType = types.PyClass(Detector)
ObservableType = types.PyClass(Observable)
