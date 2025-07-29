import asyncio
import signal
import atexit
from typing import Set
import contextlib

# Registry of active background tasks
_tasks: Set[asyncio.Task] = set()
_shutdown_started = False
_DEFAULT_TIMEOUT = 10


def register_task(task: asyncio.Task) -> None:
    """Add ``task`` to the internal registry for graceful shutdown."""
    _tasks.add(task)
    task.add_done_callback(lambda t: _tasks.discard(t))


async def graceful_shutdown(timeout_s: int = _DEFAULT_TIMEOUT) -> None:
    """Wait for all registered tasks to finish or cancel them on timeout."""
    global _shutdown_started
    if _shutdown_started:
        return
    _shutdown_started = True

    pending = [t for t in list(_tasks) if not t.done()]
    if not pending:
        return

    try:
        await asyncio.wait_for(
            asyncio.gather(*pending, return_exceptions=True),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        for t in pending:
            if not t.done():
                t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)


def _run_shutdown(timeout_s: int = _DEFAULT_TIMEOUT) -> None:
    """Synchronously trigger ``graceful_shutdown`` respecting running loop."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None

    if loop and not loop.is_closed():
        if loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(graceful_shutdown(timeout_s), loop)
            with contextlib.suppress(Exception):
                fut.result(timeout_s + 1)
        else:
            try:
                loop.run_until_complete(graceful_shutdown(timeout_s))
            except Exception:
                pass
    else:
        try:
            asyncio.run(graceful_shutdown(timeout_s))
        except Exception:
            pass


def _signal_handler(signum, frame) -> None:  # pragma: no cover - manual testing
    _run_shutdown(_DEFAULT_TIMEOUT)


for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, _signal_handler)

atexit.register(_run_shutdown)
