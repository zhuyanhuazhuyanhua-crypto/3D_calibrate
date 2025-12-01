"""Simple texturing utilities.

This module provides a lightweight fallback for texturing:
- `bake_vertex_colors_from_normals`: when camera poses/images are not available, color vertices by normals
- `bake_texture_from_images`: placeholder that documents expected inputs; minimal implementation attempts to color vertices
  by projecting them into images when camera intrinsics and extrinsics are supplied.

Note: Full photometric texturing (seam-aware UV atlas, blending) is out of scope for this demo and should
be implemented using specialised tools (MVS/texturing tools or Blender APIs).
"""
from pathlib import Path
from typing import Dict, Optional


def bake_vertex_colors_from_normals(mesh_path: str, out_path: Optional[str] = None) -> Dict:
    """Assign vertex colors based on normals (simple visual cue)."""
    try:
        import open3d as o3d
        import numpy as np
    except Exception:
        return {'status': 'open3d-not-installed'}

    p = Path(mesh_path)
    if not p.exists():
        return {'status': 'error', 'error': 'mesh not found'}

    try:
        mesh = o3d.io.read_triangle_mesh(str(p))
        if mesh.is_empty():
            return {'status': 'error', 'error': 'mesh empty'}
        mesh.compute_vertex_normals()
        normals = np.asarray(mesh.vertex_normals)
        colors = (normals * 0.5) + 0.5
        colors = colors.clip(0.0, 1.0)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

        out = out_path or str(p.with_name(p.stem + '_vc.ply'))
        o3d.io.write_triangle_mesh(out, mesh)
        return {'status': 'ok', 'textured_mesh': out}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def bake_texture_from_images(mesh_path: str, images_dir: str, camera_params: Optional[dict] = None, out_path: Optional[str] = None) -> Dict:
    """Try to bake texture by projecting mesh vertices into provided images.

    camera_params should be a dictionary mapping image filenames to a dict with 'intrinsics' and 'extrinsics'.
    intrinsics: dict with fx, fy, cx, cy
    extrinsics: 4x4 camera-to-world matrix or world-to-camera as appropriate.

    This function implements a minimal per-vertex color sampling approach. For robust textured UV maps,
    use dedicated tools (e.g., Blender, OpenMVS texture tools, ortexmapper).
    """
    try:
        import open3d as o3d
        import numpy as np
        import cv2
    except Exception:
        return {'status': 'missing-deps'}

    pmesh = Path(mesh_path)
    if not pmesh.exists():
        return {'status': 'error', 'error': 'mesh not found'}

    img_dir = Path(images_dir)
    if not img_dir.exists():
        return {'status': 'error', 'error': 'images_dir not found'}

    if camera_params is None:
        return {'status': 'not_implemented', 'error': 'camera_params required for image-based baking'}

    try:
        mesh = o3d.io.read_triangle_mesh(str(pmesh))
        if mesh.is_empty():
            return {'status': 'error', 'error': 'mesh empty'}
        verts = np.asarray(mesh.vertices)
        colors = np.zeros((len(verts), 3), dtype=float)
        counts = np.zeros((len(verts),), dtype=int)

        # For each image, project vertices and sample color
        for img_name, cam in camera_params.items():
            img_path = img_dir / img_name
            if not img_path.exists():
                continue
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]

            intr = cam.get('intrinsics')
            extr = cam.get('extrinsics')
            if intr is None or extr is None:
                continue
            fx = intr.get('fx'); fy = intr.get('fy'); cx = intr.get('cx'); cy = intr.get('cy')
            K = np.array([[fx, 0, cx],[0, fy, cy],[0,0,1.0]])

            # extr is 4x4 world-to-camera
            ext = np.array(extr)

            # transform vertices to camera coords
            hom = np.hstack([verts, np.ones((len(verts),1))])
            cam_pts = (ext @ hom.T).T
            in_front = cam_pts[:,2] > 0
            proj = (K @ cam_pts[in_front,:3].T).T
            proj[:,0] /= proj[:,2]; proj[:,1] /= proj[:,2]

            u = np.round(proj[:,0]).astype(int)
            v = np.round(proj[:,1]).astype(int)

            valid = (u >=0) & (u < w) & (v >=0) & (v < h)
            idxs = np.where(in_front)[0][valid]
            us = u[valid]; vs = v[valid]
            sampled = img[vs, us, ::-1].astype(float)/255.0  # BGR->RGB

            colors[idxs] += sampled
            counts[idxs] += 1

        nonzero = counts > 0
        colors[nonzero] = colors[nonzero] / counts[nonzero][:,None]
        # for vertices without samples, fallback to normal-based color
        mesh.compute_vertex_normals()
        normals = np.asarray(mesh.vertex_normals)
        fallback = (normals * 0.5) + 0.5
        colors[~nonzero] = fallback[~nonzero]

        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        out = out_path or str(pmesh.with_name(pmesh.stem + '_tex.ply'))
        o3d.io.write_triangle_mesh(out, mesh)
        return {'status': 'ok', 'textured_mesh': out}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}
