"""Module to create directories along a path"""

import typer

from wasabi import msg


def mkdirs(
    path: str = typer.Argument(...),
    exist_ok: bool = typer.Option(True),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Creates directories along a path"""
    import os

    os.makedirs(path, exist_ok=exist_ok)
    if verbose:
        msg.good(f"Created directory {path}")


if __name__ == "__main__":
    typer.run(mkdirs)
