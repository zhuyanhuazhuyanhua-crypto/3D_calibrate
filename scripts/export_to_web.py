"""Export outputs to web-friendly formats.

Features:
- Copy existing `.glb`/`.obj`/`.ply` into `data/web_export/` by default
- If PotreeConverter is available (or provided via config), convert pointclouds/meshes to Potree format
- For Three.js, try to convert meshes to binary `glb` using `trimesh` if available, otherwise try Open3D

This script is defensive: it will not fail if external converters are missing — it will copy files and report what it could do.
"""
import os
import subprocess
import shutil
from pathlib import Path
import json

def find_outputs(root: str):
    out = Path(root) / 'data' / 'outputs'
    return out


def load_visualization_config(root: str):
    cfg = Path(root) / 'config' / 'visualization.yaml'
    if not cfg.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(cfg.read_text(encoding='utf-8')) or {}
    except Exception:
        return {}


def is_potree_installed(potree_path: str = None) -> str:
    # return executable path or empty
    if potree_path:
        if Path(potree_path).exists():
            return str(Path(potree_path))
    # common names
    exe = shutil.which('PotreeConverter') or shutil.which('PotreeConverter.exe')
    return exe or ''


def run_potree_converter(exe: str, input_path: str, out_dir: str) -> dict:
    # Typical usage: PotreeConverter.exe input.ply -o out_dir --generate-page pageName
    try:
        cmd = [exe, str(input_path), '-o', str(out_dir), '--generate-page', 'index']
        print('Running:', ' '.join(cmd))
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return {'returncode': res.returncode, 'output': res.stdout}
    except Exception as e:
        return {'error': str(e)}


def convert_to_glb(input_path: Path, out_path: Path) -> dict:
    # If already glb, copy
    if input_path.suffix.lower() == '.glb':
        shutil.copy2(input_path, out_path)
        return {'status': 'copied', 'path': str(out_path)}

    # Try trimesh first
    try:
        import trimesh
        mesh = trimesh.load(str(input_path), force='mesh')
        if mesh.is_empty:
            return {'status': 'empty', 'path': str(input_path)}
        mesh.export(str(out_path))
        return {'status': 'exported_trimesh', 'path': str(out_path)}
    except Exception as e:
        print('trimesh conversion failed:', e)

    # Try Open3D fallback
    try:
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(str(input_path))
        if mesh.is_empty():
            return {'status': 'empty', 'path': str(input_path)}
        o3d.io.write_triangle_mesh(str(out_path), mesh)
        return {'status': 'exported_open3d', 'path': str(out_path)}
    except Exception as e:
        print('Open3D conversion failed:', e)

    # final fallback: copy
    try:
        shutil.copy2(input_path, out_path)
        return {'status': 'copied_fallback', 'path': str(out_path)}
    except Exception as e:
        return {'error': str(e)}


def export_web(root: str):
    out = find_outputs(root)
    web = Path(root) / 'data' / 'web_export'
    web.mkdir(parents=True, exist_ok=True)

    vis_cfg = load_visualization_config(root)
    potree_path_cfg = vis_cfg.get('potree_converter_path') if isinstance(vis_cfg, dict) else None
    desired_formats = vis_cfg.get('export_formats', ['potree', 'glb']) if isinstance(vis_cfg, dict) else ['potree','glb']

    potree_exe = is_potree_installed(potree_path_cfg)

    summary = {'copied': [], 'converted_glb': [], 'potree': []}

    # handle pointclouds and meshes
    for ext in ('*.ply', '*.obj', '*.glb', '*.pcd'):
        for p in out.glob(ext):
            try:
                if 'potree' in desired_formats and potree_exe and p.suffix.lower() in ('.ply', '.pcd'):
                    potree_out = web / (p.stem + '_potree')
                    potree_out.mkdir(parents=True, exist_ok=True)
                    res = run_potree_converter(potree_exe, p, potree_out)
                    summary['potree'].append({'input': str(p), 'out': str(potree_out), 'res': res})
                if 'glb' in desired_formats:
                    target = web / (p.stem + '.glb')
                    res = convert_to_glb(p, target)
                    summary['converted_glb'].append({'input': str(p), 'out': str(target), 'res': res})
                # always copy original for safe-keeping
                dst = web / p.name
                shutil.copy2(p, dst)
                summary['copied'].append(str(dst))
            except Exception as e:
                print('export error for', p, e)

    return summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Export meshes/pointclouds to web-friendly formats (Potree / glb)')
    parser.add_argument('--root', default='.')
    args = parser.parse_args()
    summary = export_web(args.root)
    print('Export summary:')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
