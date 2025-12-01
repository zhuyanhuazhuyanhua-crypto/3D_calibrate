import unittest
from unittest import mock

from src.reconstruction.adapters.colmap_adapter import ColmapAdapter


class TestColmapAdapter(unittest.TestCase):
    def test_is_available_false_when_not_installed(self):
        a = ColmapAdapter(colmap_path=None)
        # patch underlying is_colmap_installed
        with mock.patch('src.reconstruction.colmap_wrapper.is_colmap_installed', return_value=False):
            self.assertFalse(a.is_available())

    def test_run_sfm_calls_wrapper(self):
        a = ColmapAdapter(colmap_path='C:/colmap/colmap.exe')
        with mock.patch('src.reconstruction.colmap_wrapper.run_colmap_sfm') as mocked:
            mocked.return_value = {'status': 'ok'}
            res = a.run_sfm('images', 'workspace')
            mocked.assert_called_once()
            self.assertEqual(res['status'], 'ok')


if __name__ == '__main__':
    unittest.main()
