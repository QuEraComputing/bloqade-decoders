from enum import IntEnum

from kirin import types


class MeasurementResultValue(IntEnum):
    Zero = 0
    One = 1
    Lost = 2


class MeasurementResult:
    pass


class Detector:
    pass


class Observable:
    pass


# Kirin IR types
MeasurementResultType = types.PyClass(MeasurementResult)
DetectorType = types.PyClass(Detector)
ObservableType = types.PyClass(Observable)
