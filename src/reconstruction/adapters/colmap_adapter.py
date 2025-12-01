import os
from typing import Dict, Optional
from .. import colmap_wrapper, colmap_dense
from .base import AdapterBase


class ColmapAdapter(AdapterBase):
    def __init__(self, colmap_path: Optional[str] = None):
        self._path = colmap_path

    def name(self) -> str:
        return 'colmap'

    def is_available(self) -> bool:
        return colmap_wrapper.is_colmap_installed(self._path)

    def run_sfm(self, image_dir: str, workspace_dir: str) -> Dict:
        return colmap_wrapper.run_colmap_sfm(image_dir, workspace_dir, colmap_path=self._path)

    def run_dense(self, sparse_dir: str, workspace_dir: str, max_image_size: int = 2000) -> Dict:
        return colmap_dense.run_dense_colmap(sparse_dir, workspace_dir, max_image_size=max_image_size, colmap_path=self._path)
