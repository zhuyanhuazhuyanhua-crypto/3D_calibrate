"""占位的 SfM 管道实现。
在真实工程中，这里会封装对 COLMAP/OpenMVG 的调用并处理相机位姿与稀疏点云导出。
目前实现提供流程占位并返回一个状态结构。
"""
from typing import List, Dict
import os
from ..acquisition import image_loader
from . import colmap_wrapper
from .adapters.colmap_adapter import ColmapAdapter


def run_sfm(image_dir: str, workspace_dir: str = None, colmap_path: str = None) -> Dict:
    images = image_loader.list_images(image_dir)
    result = {
        'n_images': len(images),
        'status': 'no-op',
        'notes': ''
    }
    if len(images) == 0:
        result['notes'] = 'no images found — skipping SfM'
        return result

    # If user provided workspace_dir use it, otherwise create a workspace under image_dir
    if workspace_dir is None:
        workspace_dir = os.path.join(os.path.dirname(image_dir), '..', 'sfm_workspace')
    workspace_dir = os.path.abspath(workspace_dir)
    os.makedirs(workspace_dir, exist_ok=True)

    # Prefer COLMAP via adapter if available
    adapter = ColmapAdapter(colmap_path)
    if adapter.is_available():
        try:
            colmap_res = adapter.run_sfm(image_dir, workspace_dir)
            res = {'n_images': len(images), 'status': 'colmap', 'colmap': colmap_res}
            return res
        except Exception as e:
            return {'n_images': len(images), 'status': 'error', 'error': str(e)}

    # Fallback placeholder
    result['status'] = 'placeholder-completed'
    result['notes'] = f'found {len(images)} images; SfM engine not integrated in this demo.'
    return result
