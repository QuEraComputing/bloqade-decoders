from typing import Any, TypeVar, Callable

from kirin.dialects import ilist
from kirin.lowering import wraps

from .stmts import Repeat, SetDetector, SetObservable
from .types import Detector, Observable, MeasurementResult


@wraps(SetDetector)
def set_detector(
    measurements: ilist.IList[MeasurementResult, Any] | list[MeasurementResult],
    coordinates: ilist.IList[float | int, Any] | list[float | int],
) -> Detector: ...


@wraps(SetObservable)
def set_observable(
    measurements: ilist.IList[MeasurementResult, Any] | list[MeasurementResult],
) -> Observable: ...


T = TypeVar("T")


@wraps(Repeat)
def repeat(
    num_iterations: int,
    method: Callable[[], T],
) -> ilist.IList[T, int | Any]: ...
