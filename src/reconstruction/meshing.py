"""简单的网格生成工具（演示用）
生成一个示例立方体并保存为 mesh 文件。
"""
from pathlib import Path


def _write_simple_cube_ply(path: str):
    # Simple ASCII PLY for a unit cube centered at origin
    verts = [
        (-0.5, -0.5, -0.5),
        (0.5, -0.5, -0.5),
        (0.5, 0.5, -0.5),
        (-0.5, 0.5, -0.5),
        (-0.5, -0.5, 0.5),
        (0.5, -0.5, 0.5),
        (0.5, 0.5, 0.5),
        (-0.5, 0.5, 0.5),
    ]
    faces = [
        (0,1,2),(0,2,3),
        (4,7,6),(4,6,5),
        (0,4,5),(0,5,1),
        (1,5,6),(1,6,2),
        (2,6,7),(2,7,3),
        (3,7,4),(3,4,0),
    ]
    with open(path, 'w', encoding='utf-8') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {len(verts)}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write(f'element face {len(faces)}\n')
        f.write('property list uchar int vertex_indices\nend_header\n')
        for v in verts:
            f.write(f'{v[0]} {v[1]} {v[2]}\n')
        for face in faces:
            f.write(f'3 {face[0]} {face[1]} {face[2]}\n')


def create_demo_mesh(save_path: str):
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        import open3d as o3d
        mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(p), mesh)
        return str(p)
    except Exception:
        # fallback: write simple ASCII PLY
        _write_simple_cube_ply(str(p))
        return str(p)


def poisson_reconstruction(pcd_path: str, mesh_out_path: str, depth: int = 9, density_crop_quantile: float = 0.01):
    """Perform Poisson surface reconstruction from a point cloud.

    Returns dict with status and mesh path on success.
    """
    try:
        import open3d as o3d
        import numpy as np
    except Exception:
        return {'status': 'open3d-not-installed'}

    if not Path(pcd_path).exists():
        return {'status': 'error', 'error': f'pcd not found: {pcd_path}'}

    try:
        pcd = o3d.io.read_point_cloud(pcd_path)
        if pcd.is_empty():
            return {'status': 'error', 'error': 'input point cloud empty'}

        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
        densities = np.asarray(densities)

        # crop low density
        try:
            threshold = np.quantile(densities, density_crop_quantile)
            vertices_to_keep = densities > threshold
            mesh = mesh.select_by_index(np.where(vertices_to_keep)[0])
        except Exception:
            pass

        Path(mesh_out_path).parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(mesh_out_path, mesh)
        return {'status': 'ok', 'mesh': mesh_out_path}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def ball_pivoting_reconstruction(pcd_path: str, mesh_out_path: str, radii: list = None):
    """Perform Ball Pivoting reconstruction. radii should be a list of radii values.
    Returns dict with status and mesh path.
    """
    try:
        import open3d as o3d
        import numpy as np
    except Exception:
        return {'status': 'open3d-not-installed'}

    if not Path(pcd_path).exists():
        return {'status': 'error', 'error': f'pcd not found: {pcd_path}'}

    try:
        pcd = o3d.io.read_point_cloud(pcd_path)
        if pcd.is_empty():
            return {'status': 'error', 'error': 'input point cloud empty'}

        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

        if radii is None:
            # heuristic: use multiple scales based on average nearest neighbor distance
            import numpy as _np
            dists = pcd.compute_nearest_neighbor_distance()
            avg_dist = float(_np.mean(_np.asarray(dists)))
            radii = [avg_dist * f for f in (1.5, 3.0, 6.0)]

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
        Path(mesh_out_path).parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(mesh_out_path, mesh)
        return {'status': 'ok', 'mesh': mesh_out_path}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

