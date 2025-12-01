import shutil
from typing import Dict, Optional
from .base import AdapterBase


class PotreeAdapter(AdapterBase):
    def __init__(self, potree_path: Optional[str] = None):
        self._path = potree_path

    def name(self) -> str:
        return 'potree'

    def is_available(self) -> bool:
        if self._path:
            return shutil.which(self._path) is not None or False
        return shutil.which('PotreeConverter') is not None or shutil.which('PotreeConverter.exe') is not None

    def convert(self, input_path: str, out_dir: str) -> Dict:
        exe = self._path or shutil.which('PotreeConverter') or shutil.which('PotreeConverter.exe')
        if not exe:
            return {'status': 'not-found'}
        try:
            import subprocess
            cmd = [exe, input_path, '-o', out_dir, '--generate-page', 'index']
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            return {'returncode': res.returncode, 'output': res.stdout}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
