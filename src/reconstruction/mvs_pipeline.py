"""MVS pipeline orchestration: run COLMAP dense (if available) and optionally create mesh via Open3D.

This module will:
- call `colmap_dense.run_dense_colmap` to produce `dense/fused.ply`
- if Open3D is available, load `fused.ply`, estimate normals, run Poisson reconstruction and save a mesh
"""
import os
import numpy as np
from typing import Dict

from . import colmap_dense

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except Exception:
    OPEN3D_AVAILABLE = False


def run_mvs_and_mesh(sparse_dir: str, workspace_dir: str, output_dir: str) -> Dict:
    """Run dense reconstruction then optional meshing. Returns status dict."""
    os.makedirs(output_dir, exist_ok=True)

    dense_res = colmap_dense.run_dense_colmap(sparse_dir, workspace_dir)
    if dense_res.get('status') != 'ok':
        return {'status': dense_res.get('status'), 'detail': dense_res}

    fused = dense_res.get('fused_ply')
    result = {'status': 'dense_ok', 'fused_ply': fused}

    if OPEN3D_AVAILABLE and fused and os.path.exists(fused):
        try:
            pcd = o3d.io.read_point_cloud(fused)
            if pcd.is_empty():
                return {'status': 'error', 'error': 'fused point cloud empty'}

            # estimate normals
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

            # Poisson reconstruction
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

            # remove low-density vertices
            densities = np.asarray(densities)
            density_threshold = np.quantile(densities, 0.01)
            vertices_to_keep = densities > density_threshold

            # optionally crop mesh by bounding box of original pcd
            mesh_path = os.path.join(output_dir, 'mesh_poisson.ply')
            o3d.io.write_triangle_mesh(mesh_path, mesh)
            result['mesh'] = mesh_path
            result['status'] = 'dense_and_meshed'
        except Exception as e:
            result['mesh_error'] = str(e)

    else:
        result['note'] = 'Open3D not available or fused.ply missing — mesh not generated'

    # copy fused.ply to outputs
    try:
        import shutil
        if fused and os.path.exists(fused):
            shutil.copy2(fused, os.path.join(output_dir, os.path.basename(fused)))
            result['fused_copy'] = os.path.join(output_dir, os.path.basename(fused))
    except Exception:
        pass

    return result
