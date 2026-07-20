import unittest

from polyresearch.runtime.retry import retry_async


class RetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_retries_transient_failure_with_bounded_attempts(self) -> None:
        calls = 0

        async def operation():
            nonlocal calls
            calls += 1
            if calls < 3:
                raise RuntimeError("temporary")
            return "ok"

        self.assertEqual(await retry_async(operation, attempts=3, delay_seconds=0), "ok")
        self.assertEqual(calls, 3)
