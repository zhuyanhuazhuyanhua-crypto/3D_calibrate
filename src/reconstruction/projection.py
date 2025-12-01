"""2D -> 3D projection helpers: project mesh vertices into images and test visibility.

This module implements a lightweight 'vertex projection + z-buffer-like visibility' method:
- Project 3D vertices into an image using given intrinsics/extrinsics
- Build a per-pixel min-depth map from projected vertices (approximate depth buffer)
- A vertex is considered visible in an image if the vertex depth is within a small tolerance of the per-pixel min depth

The approach is approximate but fast and doesn't require a full rasterizer or raytracer.
"""
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np


def project_vertices(vertices: np.ndarray, intrinsics: Dict, extrinsics: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Project Nx3 vertices to image pixels.

    Args:
      vertices: (N,3) numpy array in world coords
      intrinsics: dict with fx, fy, cx, cy
      extrinsics: (4,4) world-to-camera matrix

    Returns:
      (uvs, depths) where uvs is (N,2) pixel coords and depths is (N,) z in camera coords
    """
    assert vertices.ndim == 2 and vertices.shape[1] == 3
    fx = intrinsics['fx']; fy = intrinsics['fy']; cx = intrinsics['cx']; cy = intrinsics['cy']
    K = np.array([[fx, 0, cx],[0, fy, cy],[0,0,1.0]])

    hom = np.concatenate([vertices, np.ones((vertices.shape[0],1))], axis=1)
    cam = (extrinsics @ hom.T).T
    depths = cam[:,2].copy()
    # avoid division by zero
    mask_positive = depths > 1e-6
    proj = np.zeros((vertices.shape[0],2), dtype=float)
    if mask_positive.any():
        proj_pts = (K @ cam[mask_positive,:3].T).T
        proj_pts[:,0] /= proj_pts[:,2]; proj_pts[:,1] /= proj_pts[:,2]
        proj[mask_positive] = proj_pts[:,:2]
    return proj, depths


def build_min_depth_map(uvs: np.ndarray, depths: np.ndarray, image_size: Tuple[int,int]) -> np.ndarray:
    """Build an image-shaped min-depth map from projected uvs and depths.

    Args:
      uvs: (N,2) pixel coords
      depths: (N,) depths
      image_size: (height, width)

    Returns:
      depth_map with shape (height, width) filled with np.inf where no vertex projected
    """
    h, w = image_size
    depth_map = np.full((h,w), np.inf, dtype=float)
    u = np.round(uvs[:,0]).astype(int)
    v = np.round(uvs[:,1]).astype(int)
    valid = (u >= 0) & (u < w) & (v >=0) & (v < h) & (depths > 0)
    for ui, vi, d in zip(u[valid], v[valid], depths[valid]):
        # keep minimum depth (closest to camera)
        if d < depth_map[vi, ui]:
            depth_map[vi, ui] = d
    return depth_map


def vertex_visibility(vertices: np.ndarray, image_size: Tuple[int,int], intrinsics: Dict, extrinsics: np.ndarray, tolerance: float = 0.02) -> List[bool]:
    """Return per-vertex boolean list indicating visibility in this camera view.

    tolerance: relative tolerance (fraction of depth) allowed between vertex depth and min depth at pixel.
    """
    uvs, depths = project_vertices(vertices, intrinsics, extrinsics)
    depth_map = build_min_depth_map(uvs, depths, image_size)
    u = np.round(uvs[:,0]).astype(int)
    v = np.round(uvs[:,1]).astype(int)
    h, w = image_size
    visible = [False] * len(vertices)
    for i in range(len(vertices)):
        if depths[i] <= 0:
            visible[i] = False
            continue
        ui, vi = int(u[i]), int(v[i])
        if ui <0 or ui >= w or vi <0 or vi >= h:
            visible[i] = False
            continue
        min_d = depth_map[vi, ui]
        if not np.isfinite(min_d):
            visible[i] = False
            continue
        # consider visible if within tolerance
        if depths[i] <= min_d * (1.0 + tolerance):
            visible[i] = True
        else:
            visible[i] = False
    return visible
