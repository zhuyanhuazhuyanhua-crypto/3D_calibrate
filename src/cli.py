"""Typer-based CLI for the reconstruction pipeline.

Provides modular commands and keeps compatibility with `python -m src.main demo`.
"""
import typer
from pathlib import Path
from . import main as legacy_main

app = typer.Typer(help='Cultural Heritage 3D Reconstruction CLI')


@app.command()
def demo(root: str = '.'):
    """Run demo pipeline (same as legacy demo)."""
    legacy_main.run_demo(root)


@app.command()
def info():
    """Print basic project info."""
    p = Path('.').resolve()
    typer.echo(f'Project root: {p}')


if __name__ == '__main__':
    app()
