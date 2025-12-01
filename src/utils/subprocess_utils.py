import subprocess
from typing import List, Optional

def run_cmd(cmd: List[str], cwd: Optional[str] = None, check: bool = True):
    """Run a command list, return CompletedProcess. Raises CalledProcessError if check and exit!=0."""
    print(f"Running: {' '.join(cmd)} (cwd={cwd})")
    res = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(res.stdout)
    if check and res.returncode != 0:
        raise subprocess.CalledProcessError(res.returncode, cmd, output=res.stdout)
    return res
