"""Minimal OpenMVS wrapper (optional).
This provides helpers to detect OpenMVS tools and run common steps.
"""
import shutil
import os
from typing import Dict
from ..utils.subprocess_utils import run_cmd


def is_openmvs_installed() -> bool:
    # Typical OpenMVS binaries: DensifyPointCloud, ReconstructMesh, RefineMesh, TextureMesh
    return any(shutil.which(bin) for bin in ['DensifyPointCloud', 'ReconstructMesh', 'RefineMesh', 'TextureMesh'])


def run_openmvs_densify(scene_mvs_file: str, workspace_dir: str) -> Dict:
    if not is_openmvs_installed():
        return {'status': 'openmvs-not-found'}
    try:
        run_cmd(['DensifyPointCloud', scene_mvs_file, '--output-file', os.path.join(workspace_dir, 'dense.mvs')])
        return {'status': 'ok'}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}
