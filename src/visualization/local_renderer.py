"""本地可视化器，使用 Open3D 展示网格/点云。若 Open3D 不可用，回退为打印路径。"""

def show_mesh(mesh_path: str):
    try:
        import open3d as o3d
    except Exception:
        print('Open3D not installed; cannot display mesh. Mesh at:', mesh_path)
        return

    try:
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if mesh.is_empty():
            print(f"failed to load mesh: {mesh_path}")
            return
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh], window_name='3D Reconstruction Viewer')
    except Exception as e:
        print('Visualization failed:', e)
