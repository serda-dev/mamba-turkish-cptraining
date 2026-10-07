"""Close owned iterators before Python shuts down native streaming readers."""
from contextlib import contextmanager


@contextmanager
def closing_iterator(items):
    stream = iter(items)
    try:
        yield stream
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            close()
