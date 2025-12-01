"""Semantic enrichment helpers for reconstruction pipeline.

Provides optional wrappers for two popular segmentation/detection toolkits:
- Detectron2 (object detection / instance segmentation)
- Segment-Anything (SAM) for promptable masks

All heavy imports happen inside functions to keep the module safe to import when dependencies
are not installed. Functions return dicts with `status` keys and useful messages/paths.

Typical usage examples are in docstrings below each function.
"""
from pathlib import Path
from typing import List, Dict, Optional


def detectron2_available() -> bool:
    try:
        import detectron2  # type: ignore
        return True
    except Exception:
        return False


def run_detectron2_inference(image_paths: List[str], config: Dict = None) -> Dict:
    """Run Detectron2 instance segmentation/detection on a list of images.

    Args:
      image_paths: list of image file paths
      config: optional dict with keys to configure model, example:
        { 'model_weights': '/path/to/model_final.pth', 'config_file': 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml', 'score_thresh': 0.5 }

    Returns:
      dict with status and `predictions` mapping image -> predictor outputs (as dicts)

    Notes: Detectron2 must be installed and its configuration registry available. This wrapper
    builds a `DefaultPredictor` and returns per-image raw outputs. Post-processing (projection to mesh)
    is left to caller.
    """
    if not detectron2_available():
        return {'status': 'detectron2-not-installed'}

    try:
        import cv2
        from detectron2.engine import DefaultPredictor  # type: ignore
        from detectron2.config import get_cfg  # type: ignore
        from detectron2 import model_zoo  # type: ignore
    except Exception as e:
        return {'status': 'import-error', 'error': str(e)}

    cfg = get_cfg()
    cfg.MODEL.DEVICE = 'cuda' if cfg.get('MODEL', {}).get('DEVICE', 'cpu') == 'cuda' else 'cpu'

    # configure from provided config dict or use a common COCO model
    try:
        if config and config.get('config_file'):
            cfg.merge_from_file(model_zoo.get_config_file(config['config_file']))
        else:
            # default
            cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))

        if config and config.get('model_weights'):
            cfg.MODEL.WEIGHTS = config['model_weights']
        else:
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')

        score_thresh = 0.5
        if config and 'score_thresh' in config:
            score_thresh = float(config['score_thresh'])
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh

        predictor = DefaultPredictor(cfg)
    except Exception as e:
        return {'status': 'config-error', 'error': str(e)}

    results = {}
    for im_path in image_paths:
        try:
            im = cv2.imread(im_path)
            outputs = predictor(im)
            # convert tensors to CPU numpy where possible; keep structure generic
            results[im_path] = {'instances': outputs.get('instances').to('cpu') if outputs.get('instances') is not None else None}
        except Exception as e:
            results[im_path] = {'error': str(e)}

    return {'status': 'ok', 'predictions': results}


def sam_available() -> bool:
    try:
        # segment_anything package or legacy sam implementations may be present
        import segment_anything  # type: ignore
        return True
    except Exception:
        try:
            # some distributions expose sam via other names
            from segment_anything import SamPredictor  # type: ignore
            return True
        except Exception:
            return False


def run_sam_on_image(image_path: str, checkpoint: Optional[str] = None, prompts: Dict = None) -> Dict:
    """Run Segment-Anything on an image.

    Args:
      image_path: path to a single image
      checkpoint: optional path to SAM checkpoint
      prompts: dict containing prompt types, e.g. {'points': [(x,y,1), ...], 'boxes': [(x1,y1,x2,y2), ...]}

    Returns:
      dict with status and masks (if available). Each mask is returned as a numpy array in memory (may be large).
    """
    if not sam_available():
        return {'status': 'sam-not-installed'}

    try:
        import cv2
        import numpy as np
        from segment_anything import sam_model_registry, SamPredictor  # type: ignore
    except Exception as e:
        return {'status': 'import-error', 'error': str(e)}

    try:
        image = cv2.imread(image_path)
        if image is None:
            return {'status': 'error', 'error': 'failed to read image'}

        # load default model if checkpoint provided
        if checkpoint and Path(checkpoint).exists():
            model_type = 'vit_b'
            sam = sam_model_registry.get(model_type)(checkpoint=checkpoint)
        else:
            # try to use a default lightweight model if provided by package
            try:
                # assume registry has a default key
                sam = sam_model_registry.get('vit_b')(checkpoint=None)
            except Exception:
                return {'status': 'error', 'error': 'no SAM checkpoint and default model unavailable'}

        predictor = SamPredictor(sam)
        predictor.set_image(image[:,:,::-1])  # SAM expects RGB

        # build prompts
        boxes = prompts.get('boxes') if prompts else None
        points = prompts.get('points') if prompts else None

        masks = []
        if boxes:
            for box in boxes:
                # box: [x1,y1,x2,y2]
                transformed_box = predictor.transform.apply_boxes_torch([box], image.shape[:2])[0].cpu().numpy()
                result = predictor.predict_boxes(torch.tensor([transformed_box]))  # type: ignore
                masks.append(result[0])

        if points:
            # points: list of (x,y,label) where label 1 = foreground
            coords = np.array([[p[0], p[1]] for p in points])
            labels = np.array([p[2] for p in points])
            masks_pts, scores, logits = predictor.predict(point_coords=coords, point_labels=labels, multimask_output=False)
            masks.append(masks_pts[0])

        return {'status': 'ok', 'masks': masks}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def project_masks_to_mesh(mesh_path: str, masks: List, image_path: str, camera_params: Dict) -> Dict:
    """Placeholder: project image masks to mesh vertices/faces using camera calibration.

    This is a non-trivial step requiring correct camera intrinsics/extrinsics and a method to map 2D pixels
    to mesh geometry (ray casting or projecting vertices). Here we provide a placeholder that documents
    expected inputs and returns `not_implemented` unless user supplies a custom projector function.
    """
    # Backward compatibility alias to new implementation
    return project_masks_to_mesh_vertices(mesh_path, masks, image_path, camera_params)


def _read_ply_vertices(ply_path: str):
    """Minimal ASCII PLY reader to extract vertices when Open3D is not available.

    Returns numpy array (N,3)
    """
    import re
    verts = []
    with open(ply_path, 'r', encoding='utf-8') as f:
        header = True
        vertex_count = 0
        while header:
            line = f.readline()
            if not line:
                break
            if line.startswith('element vertex'):
                parts = line.strip().split()
                vertex_count = int(parts[-1])
            if line.strip() == 'end_header':
                header = False
                break
        for i in range(vertex_count):
            line = f.readline()
            if not line:
                break
            vals = re.split(r'\s+', line.strip())
            if len(vals) >= 3:
                verts.append([float(vals[0]), float(vals[1]), float(vals[2])])
    import numpy as np
    return np.array(verts, dtype=float)


def _write_ply_with_vertex_colors(vertices, colors, out_path: str):
    # vertices: (N,3), colors: (N,3) floats [0,1]
    from pathlib import Path
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {len(vertices)}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('end_header\n')
        for v, c in zip(vertices, colors):
            r = int(max(0, min(255, round(c[0]*255))))
            g = int(max(0, min(255, round(c[1]*255))))
            b = int(max(0, min(255, round(c[2]*255))))
            f.write(f'{v[0]} {v[1]} {v[2]} {r} {g} {b}\n')


def _class_to_color(class_id: int):
    # deterministic pseudo-color mapping for small integers
    r = (class_id * 37) % 255
    g = (class_id * 73) % 255
    b = (class_id * 151) % 255
    return (r/255.0, g/255.0, b/255.0)


def _build_vertex_adjacency(triangles, n_vertices):
    """Build adjacency list from triangle array (M,3)."""
    adjacency = [[] for _ in range(n_vertices)]
    for tri in triangles:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        if b not in adjacency[a]:
            adjacency[a].append(b)
        if c not in adjacency[a]:
            adjacency[a].append(c)
        if a not in adjacency[b]:
            adjacency[b].append(a)
        if c not in adjacency[b]:
            adjacency[b].append(c)
        if a not in adjacency[c]:
            adjacency[c].append(a)
        if b not in adjacency[c]:
            adjacency[c].append(b)
    return adjacency


def _smooth_vertex_labels(classes, adjacency, iterations: int = 1):
    """Smooth labels by majority vote within vertex neighborhood."""
    import collections
    labels = list(classes)
    for _ in range(iterations):
        new_labels = labels.copy()
        for i, neigh in enumerate(adjacency):
            counts = collections.Counter()
            counts[labels[i]] += 1
            for j in neigh:
                counts[labels[j]] += 1
            # choose majority, tie-breaker smallest id
            best = max(sorted(counts.items(), key=lambda x: x[0]), key=lambda x: x[1])
            new_labels[i] = best[0]
        labels = new_labels
    return labels


def convert_detectron2_outputs_to_image_masks(detectron_results: Dict, image_shape: Tuple[int,int]) -> Dict:
    """Convert a Detectron2-like result dict to per-pixel class-confidence maps.

    Input format supported (for compatibility/testing):
      detectron_results: {
         'masks': [HxW bool arrays],
         'scores': [float],
         'classes': [int]
      }

    Returns:
      {'conf': HxW x C numpy array} where C = max(class)+1
    """
    import numpy as np
    masks = detectron_results.get('masks', [])
    scores = detectron_results.get('scores', [])
    classes = detectron_results.get('classes', [])
    if len(masks) == 0:
        return {}
    max_cls = max(classes) if len(classes) > 0 else 0
    C = int(max_cls) + 1
    h, w = image_shape
    conf = np.zeros((h, w, C), dtype=float)
    for m, s, c in zip(masks, scores, classes):
        # assume m is boolean mask array shape h,w
        conf[..., int(c)] += (m.astype(float) * float(s))
    return {'conf': conf}


def convert_sam_output_to_image_masks(sam_masks: List, image_shape: Tuple[int,int]) -> Dict:
    """Convert SAM masks (list of HxW boolean arrays) to class-confidence map.

    For SAM we don't have class ids; we treat each mask as a single class id 1 with score 1.
    This returns a conf map with two classes: 0=background,1=mask
    """
    import numpy as np
    h, w = image_shape
    conf = np.zeros((h, w, 2), dtype=float)
    for m in sam_masks:
        conf[...,1] += m.astype(float)
    return {'conf': conf}


def project_masks_to_mesh_vertices(mesh_path: str, image_masks: Dict[str, object], camera_params: Dict[str, Dict], out_path: Optional[str] = None, tolerance: float = 0.02) -> Dict:
    """Project per-image masks to mesh vertices and produce a vertex-colored PLY.

    Args:
      mesh_path: path to input mesh (PLY). If Open3D is available it will be used; otherwise a simple PLY parser is used.
      image_masks: mapping image_filename -> mask (numpy array HxW with integer class ids) or path to mask image
      camera_params: mapping image_filename -> {'intrinsics': {fx,fy,cx,cy}, 'extrinsics': 4x4 world-to-camera matrix}
      out_path: optional output path; defaults to `data/outputs/mesh_semantic_vc.ply`
      tolerance: visibility tolerance passed to projection.vertex_visibility

    Returns:
      dict with status and output path
    """
    try:
        import numpy as np
    except Exception as e:
        return {'status': 'error', 'error': 'numpy required'}

    # load vertices
    verts = None
    try:
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        verts = np.asarray(mesh.vertices)
    except Exception:
        try:
            verts = _read_ply_vertices(mesh_path)
        except Exception as e:
            return {'status': 'error', 'error': f'failed to read mesh: {e}'}

    if verts is None or len(verts) == 0:
        return {'status': 'error', 'error': 'no vertices found in mesh'}

    # prepare vote storage: a list of dicts
    from collections import defaultdict
    votes = [defaultdict(int) for _ in range(len(verts))]

    # iterate images
    for img_name, mask_obj in image_masks.items():
        cam = camera_params.get(img_name)
        if cam is None:
            continue
        intr = cam.get('intrinsics')
        extr = cam.get('extrinsics')
        if intr is None or extr is None:
            continue

        # load mask array
        mask_arr = None
        if isinstance(mask_obj, str):
            # path
            try:
                import cv2
                ma = cv2.imread(mask_obj, cv2.IMREAD_UNCHANGED)
                if ma is None:
                    continue
                if ma.ndim == 3:
                    # assume single-channel class encoded in one channel or rgb palette; take first channel
                    mask_arr = ma[:,:,0]
                else:
                    mask_arr = ma
            except Exception:
                continue
        else:
            mask_arr = mask_obj

        if mask_arr is None:
            continue

        # image size
        h, w = mask_arr.shape[:2]

        # compute visibility
        from . import projection
        vis = projection.vertex_visibility(verts, (h, w), intr, np.array(extr), tolerance=tolerance)

        # compute projections to sample mask values
        uvs, depths = projection.project_vertices(verts, intr, np.array(extr))

        for vi, visible in enumerate(vis):
            if not visible:
                continue
            u, v = int(round(uvs[vi,0])), int(round(uvs[vi,1]))
            if u < 0 or u >= w or v < 0 or v >= h:
                continue
            class_id = int(mask_arr[v, u])
            votes[vi][class_id] += 1

    # determine per-vertex class by (weighted) vote
    classes = []
    for d in votes:
        if len(d) == 0:
            classes.append(0)
        else:
            # pick class with max accumulated weight; tie-breaker is smallest class id
            best = max(sorted(d.items(), key=lambda x: x[0]), key=lambda x: x[1])
            classes.append(int(best[0]))

    # optional spatial smoothing: if mesh triangles available, smooth labels by neighbor majority
    try:
        # attempt to read triangles for adjacency
        import open3d as o3d
        mesh_in = o3d.io.read_triangle_mesh(mesh_path)
        tris = None
        try:
            tris = np.asarray(mesh_in.triangles)
        except Exception:
            tris = None
    except Exception:
        tris = None

    # smoothing param: can be provided via camera_params['_smoothing_iters'] (global) or default 1
    smoothing_iters = 0
    if isinstance(camera_params, dict) and '_smoothing_iters' in camera_params:
        try:
            smoothing_iters = int(camera_params['_smoothing_iters'])
        except Exception:
            smoothing_iters = 0

    if smoothing_iters > 0 and tris is not None:
        # build adjacency
        adjacency = _build_vertex_adjacency(tris, len(verts))
        classes = _smooth_vertex_labels(classes, adjacency, iterations=smoothing_iters)

    # map classes to colors
    colors = [_class_to_color(c) for c in classes]

    # write output PLY (try to use open3d if available and preserve faces if possible)
    out = out_path or 'data/outputs/mesh_semantic_vc.ply'
    try:
        import open3d as o3d
        mesh_out = o3d.geometry.TriangleMesh()
        mesh_out.vertices = o3d.utility.Vector3dVector(verts)
        # faces are not preserved when we used fallback read; attempt to read faces
        try:
            mesh_in = o3d.io.read_triangle_mesh(mesh_path)
            mesh_out.triangles = mesh_in.triangles
        except Exception:
            pass
        mesh_out.vertex_colors = o3d.utility.Vector3dVector(colors)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(out, mesh_out)
        return {'status': 'ok', 'out': out}
    except Exception:
        # fallback: write simple PLY with vertex colors
        try:
            _write_ply_with_vertex_colors(verts, colors, out)
            return {'status': 'ok', 'out': out}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}


def enrich_mesh_with_semantics(mesh_path: str, images_dir: str, camera_params: Dict = None, method: str = 'detectron2') -> Dict:
    """High-level convenience function: run chosen segmentation and (optionally) project results to mesh.

    Args:
      mesh_path: path to mesh file to annotate
      images_dir: directory containing reference images
      camera_params: mapping image->intrinsics/extrinsics required for projection
      method: 'detectron2' or 'sam'

    Returns:
      dict summarizing steps and outputs. For projection step the function currently returns `not_implemented`.
    """
    p = Path(images_dir)
    if not p.exists():
        return {'status': 'error', 'error': 'images_dir not found'}

    images = [str(x) for x in p.glob('*') if x.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tif', '.tiff')]
    if len(images) == 0:
        return {'status': 'error', 'error': 'no images found in images_dir'}

    summary = {'method': method, 'images': len(images)}

    if method == 'detectron2':
        det_res = run_detectron2_inference(images)
        summary['detectron2'] = det_res
        # projection is left as not implemented
        summary['projection'] = 'not_implemented'
        return {'status': 'ok', 'summary': summary}

    if method == 'sam':
        sam_results = {}
        for im in images:
            sam_results[im] = run_sam_on_image(im)
        summary['sam'] = sam_results
        summary['projection'] = 'not_implemented'
        return {'status': 'ok', 'summary': summary}

    return {'status': 'error', 'error': f'unknown method {method}'}
