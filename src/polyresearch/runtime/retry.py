"""Small bounded retry helpers for transient external-operation failures."""

import asyncio
import logging
from collections.abc import Awaitable, Callable

from polyresearch.security import redacted_exception_info

logger = logging.getLogger(__name__)


async def retry_async(
    operation: Callable[[], Awaitable[object]], *, attempts: int, delay_seconds: float = 0.1
) -> object:
    """Retry an async operation a bounded number of times without swallowing its error."""
    if attempts < 1:
        raise ValueError("attempts must be positive")
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return await operation()
        except Exception as error:
            last_error = error
            logger.warning(
                "Retryable operation failed",
                extra={
                    "operation": "retry_async",
                    "attempt": attempt + 1,
                    "attempts": attempts,
                    "will_retry": attempt + 1 < attempts,
                },
                exc_info=redacted_exception_info(error),
            )
            if attempt + 1 < attempts:
                await asyncio.sleep(delay_seconds * (attempt + 1))
    assert last_error is not None
    raise last_error
