import unittest
import tempfile
import os
from pathlib import Path

import src.main as mainmod


class TestDemoRun(unittest.TestCase):
    def test_demo_runs_with_minimal_configs(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            # create minimal config files
            cfg_dir = tdp / 'config'
            cfg_dir.mkdir()
            (cfg_dir / 'dataset.yaml').write_text('raw_images: "data/raw/images"\nraw_pointclouds: "data/raw/pointclouds"\nprocessed: "data/processed"\noutputs: "data/outputs"\n')
            (cfg_dir / 'reconstruction.yaml').write_text('sfm:\n  engine: "placeholder"\n  use_colmap: false\n')
            (cfg_dir / 'visualization.yaml').write_text('viewer: "open3d"\n')
            (tdp / 'data' / 'raw' / 'images').mkdir(parents=True)
            # run demo should not raise
            mainmod.run_demo(str(tdp))
            # check outputs
            out = tdp / 'data' / 'outputs' / 'demo_cube.ply'
            self.assertTrue(out.exists())


if __name__ == '__main__':
    unittest.main()
