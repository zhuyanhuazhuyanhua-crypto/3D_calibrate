import os
import cv2
from pathlib import Path

def list_images(image_dir):
    p = Path(image_dir)
    if not p.exists():
        return []
    exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
    files = [str(x) for x in p.glob('**/*') if x.suffix.lower() in exts]
    return sorted(files)

def load_image(path, grayscale=False):
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(path, flag)
    return img
