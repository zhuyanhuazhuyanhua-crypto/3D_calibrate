import unittest
import numpy as np
import tempfile
from pathlib import Path

from src.reconstruction.semantic_enrichment import project_masks_to_mesh_vertices


class TestSemanticEnrichment(unittest.TestCase):
    def test_project_masks_to_mesh_vertices_simple(self):
        # create temporary mesh PLY with 3 vertices
        with tempfile.TemporaryDirectory() as td:
            tdpath = Path(td)
            mesh_p = tdpath / 'in_mesh.ply'
            # write simple PLY with 3 vertices
            verts = [
                (0.0, 0.0, 2.0),  # projects to center
                (0.2, 0.0, 2.0),  # projects to right
                (0.0, 0.2, 2.0),  # projects to down
            ]
            with open(mesh_p, 'w', encoding='utf-8') as f:
                f.write('ply\nformat ascii 1.0\n')
                f.write('element vertex 3\n')
                f.write('property float x\nproperty float y\nproperty float z\n')
                f.write('end_header\n')
                for v in verts:
                    f.write(f'{v[0]} {v[1]} {v[2]}\n')

            # create mask image where center pixel is class 1, others 0
            h, w = 101, 101
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[50, 50] = 1

            # camera intrinsics/extrinsics
            intr = {'fx': 100.0, 'fy': 100.0, 'cx': 50.0, 'cy': 50.0}
            ext = np.eye(4)

            image_masks = {'img1': mask}
            camera_params = {'img1': {'intrinsics': intr, 'extrinsics': ext.tolist()}}

            out = tdpath / 'out_mesh_semantic.ply'
            res = project_masks_to_mesh_vertices(str(mesh_p), image_masks, camera_params, out_path=str(out), tolerance=0.05)
            self.assertEqual(res.get('status'), 'ok')
            self.assertTrue(out.exists())
            # read output ply and check first vertex color is class 1 color
            data = out.read_text(encoding='utf-8')
            # find first color line
            lines = data.splitlines()
            # header ends at line with 'end_header'
            idx = lines.index('end_header')
            vertex_line = lines[idx+1].split()
            # last three entries are rgb
            r, g, b = int(vertex_line[-3]), int(vertex_line[-2]), int(vertex_line[-1])
            # class 1 maps to deterministic color via _class_to_color mapping; check r>0
            self.assertTrue(r > 0 or g > 0 or b > 0)


if __name__ == '__main__':
    unittest.main()
