# utils/decorators.py

import time
from functools import wraps


def retry_on_error(
    max_retries: int = 3, delay: float = 1.0, exceptions: tuple = (Exception,)
):
    """Decorator for retrying operations on specific errors"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        # Log retry attempt
                        if hasattr(args[0], "logger"):
                            args[0].logger.warning(
                                f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying..."
                            )
                        time.sleep(delay * (2**attempt))  # Exponential backoff
                        continue
                    break
                except Exception as e:
                    # Don't retry on other types of errors, but log them
                    if hasattr(args[0], "logger"):
                        args[0].logger.error(
                            f"Non-retryable error in {func.__name__}: {e}"
                        )
                    raise e

            # Log final failure
            if hasattr(args[0], "logger"):
                args[0].logger.error(
                    f"All {max_retries} attempts failed for {func.__name__}"
                )
            raise last_exception

        return wrapper

    return decorator
