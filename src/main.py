"""项目主入口：协调各模块流程并提供一个 demo 模式。"""
import argparse
import os
from pathlib import Path

from .acquisition import image_loader
from .reconstruction import sfm_pipeline, meshing
from .reconstruction import mvs_pipeline
from .visualization import local_renderer
from .utils import file_io


def run_demo(root_dir: str):
    print('Running demo pipeline...')
    cfg = file_io.read_yaml(os.path.join(root_dir, 'config', 'dataset.yaml'))
    recon_cfg = file_io.read_yaml(os.path.join(root_dir, 'config', 'reconstruction.yaml'))
    colmap_path = None
    try:
        colmap_cfg = recon_cfg.get('sfm', {})
        colmap_path = colmap_cfg.get('colmap_path')
    except Exception:
        colmap_path = None
    image_dir = os.path.join(root_dir, cfg.get('raw_images', 'data/raw/images'))
    outputs = os.path.join(root_dir, 'data', 'outputs')
    Path(outputs).mkdir(parents=True, exist_ok=True)

    images = image_loader.list_images(image_dir)
    print(f'Found {len(images)} images in {image_dir}')

    sfm_res = sfm_pipeline.run_sfm(image_dir, colmap_path=colmap_path)
    print('SfM result:', sfm_res)

    mesh_path = os.path.join(outputs, 'demo_cube.ply')
    if len(images) == 0:
        print('No images available — generating demo mesh')
        mesh_path = meshing.create_demo_mesh(mesh_path)
        print('Demo mesh saved to', mesh_path)
    else:
        # If SfM produced a COLMAP workspace, try MVS pipeline
        if isinstance(sfm_res, dict) and sfm_res.get('status') == 'colmap':
            col_res = sfm_res.get('colmap', {})
            sparse_dir = col_res.get('sparse_dir')
            if sparse_dir:
                print('Running MVS pipeline (dense reconstruction + optional meshing)')
                mvs_out = mvs_pipeline.run_mvs_and_mesh(sparse_dir, os.path.dirname(sparse_dir), outputs)
                print('MVS result:', mvs_out)
                if mvs_out.get('mesh'):
                    mesh_path = mvs_out.get('mesh')
        else:
            print('Images present — please integrate SfM/MVS to generate mesh.')

    # Try to show mesh if it exists
    if os.path.exists(mesh_path):
        try:
            local_renderer.show_mesh(mesh_path)
        except Exception as e:
            print('Visualization failed:', e)


def cli():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')
    sub.required = True

    parser_demo = sub.add_parser('demo')
    parser_demo.add_argument('--root', default='.', help='project root')

    args = parser.parse_args()
    if args.cmd == 'demo':
        run_demo(args.root)


if __name__ == '__main__':
    # backward compatibility: support legacy invocation
    try:
        from .cli import app
        app()
    except Exception:
        cli()
