from enum import IntEnum
from dataclasses import dataclass

from kirin import types


class MeasurementResultValue(IntEnum):
    """Classify a physical measurement as zero, one, or atom loss."""

    Zero = 0
    One = 1
    Lost = 2

    @property
    def symbol(self) -> str:
        """Single-character symbol: '0', '1', or 'L'(Lost)"""

        return {
            MeasurementResultValue.Zero: "0",
            MeasurementResultValue.One:  "1",
            MeasurementResultValue.Lost: "L",
        }[self]
    
    def __repr__(self) -> str:
        return self.symbol
    
    __str__ = __repr__

@dataclass
class MeasurementResult:
    """Represent a measurement outcome consumed by parity annotations."""

    value: MeasurementResultValue

    def __repr__(self) -> str:
        return self.value.symbol


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
