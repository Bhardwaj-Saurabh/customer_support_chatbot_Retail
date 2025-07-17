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

@click.command()
@click.option(
    "--user-id",
    type=str,
    required=True,
    help="ID of the User.",
)
@click.option(
    "--query",
    type=str,
    required=True,
    help="Query to call the agent with.",
)


@async_command
async def main(user_id: str, query: str) -> None:
    """CLI command to query a philosopher.

    Args:
        user_id: ID of the User.
        query: Query to call the agent with.
    """
    async for chunk in get_streaming_response(
        messages=query,
        user_id=user_id,
    ):
        print(f"\033[32m{chunk}\033[0m", end="", flush=True)


if __name__ == "__main__":
    main()