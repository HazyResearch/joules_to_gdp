"""Command line entry point for GDPVal energy profiling."""

from __future__ import annotations

import click

from .run import run_cmd


@click.group(help="Joules to GDP profiling commands")
def cli() -> None:
    """Top level CLI group."""


cli.add_command(run_cmd, "run")


def main() -> None:
    """Console script entry point."""
    cli()


__all__ = ["cli", "main"]
