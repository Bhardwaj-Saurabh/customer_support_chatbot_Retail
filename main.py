import asyncio
from functools import wraps

import click

from assistant.application.generate_response import (
    get_streaming_response,
)


def async_command(f):
    """Decorator to run an async click command."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


@async_command
async def main(user_id: str, query: str) -> None:
    """CLI command to query a philosopher.

    Args:
        philosopher_id: ID of the philosopher to call.
        query: Query to call the agent with.
    """
    print("\033[32mResponse:\033[0m")
    print("\033[32m--------------------------------\033[0m")
    async for chunk in get_streaming_response(
        messages=query,
        user_id=user_id,
    ):
        print(f"\033[32m{chunk}\033[0m", end="", flush=True)
    print("\033[32m--------------------------------\033[0m")


if __name__ == "__main__":
    main(user_id = "saurabh", query = "How can I view my past orders and their billing details?")