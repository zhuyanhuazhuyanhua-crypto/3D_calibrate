"""Simple wrapper for common COLMAP CLI steps.

This wrapper detects COLMAP in PATH and runs a minimal SfM pipeline:
- feature_extractor
- exhaustive_matcher
- mapper
- model_converter (export PLY)

It is written defensively: if COLMAP is not found, callers should fallback.
"""
import shutil
import os
from pathlib import Path
from typing import Dict, Optional

from ..utils.subprocess_utils import run_cmd


def is_colmap_installed(colmap_path: str = None) -> bool:
    """Check whether COLMAP is available. If colmap_path provided, check that path first."""
    if colmap_path:
        # If user provided a path, check it exists and is executable
        if os.path.isabs(colmap_path) and os.path.exists(colmap_path):
            return True
    exe = shutil.which('colmap') or shutil.which('colmap.exe')
    return exe is not None


def run_colmap_sfm(image_dir: str, workspace_dir: str, colmap_path: str = None) -> Dict:
    """Run a minimal COLMAP SfM pipeline. Returns dict with paths and status."""
    if not is_colmap_installed():
        return {'status': 'colmap-not-found'}

    image_dir = os.path.abspath(image_dir)
    workspace_dir = os.path.abspath(workspace_dir)
    db_path = os.path.join(workspace_dir, 'database.db')
    sparse_dir = os.path.join(workspace_dir, 'sparse')
    os.makedirs(sparse_dir, exist_ok=True)

    try:
        exe = colmap_path if (colmap_path and os.path.exists(colmap_path)) else 'colmap'

        # Feature extraction
        run_cmd([exe, 'feature_extractor', '--database_path', db_path, '--image_path', image_dir])

        # Exhaustive matcher (good default for small sets)
        run_cmd([exe, 'exhaustive_matcher', '--database_path', db_path])

        # Sparse reconstruction (mapper)
        run_cmd([exe, 'mapper', '--database_path', db_path, '--image_path', image_dir, '--output_path', sparse_dir])

        # Convert first model to PLY if present
        model_files = sorted(Path(sparse_dir).glob('*'))
        ply_out = None
        # COLMAP creates folders like 0/, 1/ with model files
        for mf in model_files:
            if mf.is_dir():
                model_path = str(mf)
                ply_out = os.path.join(workspace_dir, 'sparse_model.ply')
                run_cmd([exe, 'model_converter', '--input_path', model_path, '--output_path', ply_out, '--output_type', 'PLY'])
                break

        result = {'status': 'ok', 'database': db_path, 'sparse_dir': sparse_dir, 'sparse_ply': ply_out}
        return result
    except Exception as e:
        return {'status': 'error', 'error': str(e)}
