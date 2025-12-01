import unittest
import numpy as np

from src.reconstruction.projection import project_vertices, build_min_depth_map, vertex_visibility


class TestProjection(unittest.TestCase):
    def test_projection_and_visibility_simple(self):
        # simple 4 vertices in front of camera
        verts = np.array([
            [0.0, 0.0, 2.0],
            [0.2, 0.0, 2.0],
            [0.0, 0.2, 2.0],
            [0.0, 0.0, 4.0],
        ])
        intr = {'fx': 100.0, 'fy': 100.0, 'cx': 50.0, 'cy': 50.0}
        # camera at origin, looking down -Z (world-to-camera is identity)
        ext = np.eye(4)
        uvs, depths = project_vertices(verts, intr, ext)
        self.assertEqual(uvs.shape[0], 4)
        # build depth map on small image
        depth_map = build_min_depth_map(uvs, depths, (100,100))
        # check that vertex 0 has minimal depth at its pixel
        u0 = int(round(uvs[0,0])); v0 = int(round(uvs[0,1]))
        self.assertAlmostEqual(depth_map[v0,u0], depths[0])

        vis = vertex_visibility(verts, (100,100), intr, ext, tolerance=0.05)
        # first three should be visible, the far one likely visible but deeper
        self.assertTrue(all(isinstance(x, bool) for x in vis))


if __name__ == '__main__':
    unittest.main()
