from typing import Any, TypeVar, Callable

from kirin.dialects import ilist
from kirin.lowering import wraps

from .stmts import Repeat

T = TypeVar("T")


@wraps(Repeat)
def repeat(
    num_iterations: int,
    method: Callable[[], T],
) -> ilist.IList[T, int | Any]: ...
