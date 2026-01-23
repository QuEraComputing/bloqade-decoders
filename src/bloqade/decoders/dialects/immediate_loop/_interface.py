from typing import Any, Generator
from contextlib import contextmanager

from kirin.lowering import wraps

from .stmts import Repeat


@wraps(Repeat)
@contextmanager
def repeat(count: int) -> Generator[Any, None, None]: ...
