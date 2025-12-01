"""Wrapper for COLMAP dense reconstruction steps.

Implements:
- image_undistorter
- patch_match_stereo
- stereo_fusion -> fused.ply

If COLMAP is not installed, functions return status indicating so.
"""
import os
import shutil
from pathlib import Path
from typing import Dict

from .colmap_wrapper import is_colmap_installed
from ..utils.subprocess_utils import run_cmd


def run_dense_colmap(sparse_dir: str, workspace_dir: str, max_image_size: int = 2000, colmap_path: str = None) -> Dict:
    """Run dense reconstruction using COLMAP commands.

    sparse_dir: path to COLMAP sparse output folder (e.g., workspace/sparse/0)
    workspace_dir: base workspace where dense/ will be created
    returns dict with fused_ply path when successful
    """
    if not is_colmap_installed(colmap_path):
        return {'status': 'colmap-not-found'}

    sparse_dir = os.path.abspath(sparse_dir)
    workspace_dir = os.path.abspath(workspace_dir)
    dense_dir = os.path.join(workspace_dir, 'dense')
    os.makedirs(dense_dir, exist_ok=True)

    try:
        exe = colmap_path if (colmap_path and os.path.exists(colmap_path)) else 'colmap'

        # image_undistorter expects the sparse model folder (containing cameras.txt, images.txt, points3D.txt)
        run_cmd([
            exe, 'image_undistorter',
            '--image_path', os.path.dirname(sparse_dir) if os.path.isdir(sparse_dir) else sparse_dir,
            '--input_path', sparse_dir,
            '--output_path', dense_dir,
            '--output_type', 'COLMAP',
            '--max_image_size', str(max_image_size)
        ])

        # patch_match_stereo
        run_cmd([
            exe, 'patch_match_stereo',
            '--workspace_path', dense_dir,
            '--PatchMatchStereo.geom_consistency', 'true'
        ])

        # stereo fusion — produce fused point cloud
        fused = os.path.join(dense_dir, 'fused.ply')
        run_cmd([
            exe, 'stereo_fusion',
            '--workspace_path', dense_dir,
            '--output_path', fused
        ])

        return {'status': 'ok', 'dense_dir': dense_dir, 'fused_ply': fused}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}
