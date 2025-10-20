import os
import re
import glob
from collections import defaultdict, Counter
import logging
from PIL import Image, ImageOps
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_img(p: Path) -> bool:
    """
    Check if the input path has an image file extension.

    Args:
        p (Path): The input path to check.

    Returns:
        bool: True if the path has an image file extension, False otherwise.
    """
    return p.suffix.lower() in IMG_EXT


def load_img(p: Path, max_hw=512):
    """
    Load an image from the specified path, convert it to RGB format,
    and resize it if its dimensions exceed max_hw.

    Args:
        p (Path): The path to the image file.
        max_hw (int, optional): The maximum height or width of the image. Defaults to 512.

    Returns:
        Image: The loaded and possibly resized image in RGB format.
    """
    img = Image.open(p).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_hw / max(h, w))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))
    return img


def idx_img_from_dir(dir: Path) -> list[Path]:
    """
    Get a list of image file paths from the specified directory and its subdirectories.

    Args:
        dir (Path): The directory path to search for image files.

    Returns:
        list[Path]: A list of paths to image files in the directory.
    """
    if not dir.exists():
        raise ValueError(f"{dir} is not a valid directory.")

    # recursively find all img files in the directory and its subdirectories
    files = [Path(p) for p in glob.glob(str(dir / "**" / "*"), recursive=True)]
    return [p for p in files if is_img(p) and p.is_file()]


def parse_inpaint(fake_root: Path, quiet: bool = True) -> dict:
    """
    Parse the inpainting images from the fake image directory.

    Args:
        fake_root (Path): The root directory containing fake images.
    """
    masks = {}

    for p in glob.glob(str(fake_root / "*" / "mask" / "*" / "*")):
        pth = Path(p)
        if not pth.is_file() or not is_img(pth):
            continue
        parts = pth.parts
        parts_l = [s.lower() for s in parts]
        try:
            i = parts_l.index("mask")
            subset = parts[i - 1]  # 0000
            id_ = parts[i + 1]  # <id>
            label = pth.stem  # <label> : the object mask label
        except Exception as e:
            if not quiet:
                logging.warning(f"Skip mask {pth}: {e}")
            continue

        key = (subset, id_)
        if key not in masks:
            masks[key] = {"path": pth, "labels": {}}
        masks[key]["labels"][label] = pth

    return masks


def parse_fake_path(fake_root: Path, quiet: bool = True) -> dict:
    """
    Parse the fake image directory to organize the information.

    The expected directory structure is as follows:
    fake_images/
    ├─ 0000/
    │   ├─ inpainting/<id>/(SD|SDXL|FLUX1)_inpainting_<id>_<label>.png
    │   ├─ text2image/<id>/(SD|SDXL|FLUX1)_text2image_<id>.png
    │   ├─ mask/<id>/<label>.png
    │   └─ image_info/<id>.txt
    └─ 0010/...

    Args:
        fake_root (Path): The root directory containing fake images.

    Returns:
        dict: A dictionary containing the parsed information.
        - inpaint[(subset, idx)] = { 'paths': [Path...], 'models': [SD/SDXL/FLUX1...], 'mask_label': set(...), 'mask_path': Path|None }
        - t2i[(subset, idx)]     = { 'paths': [Path...], 'models': [SD/SDXL/FLUX1...] }
        - masks[(subset, idx)]   = { 'path': Path, 'labels': ... }
        - infos[(subset, idx)]   = Path(txt)
    """
    if not fake_root.exists():
        raise ValueError(f"{fake_root} is not a valid directory.")

    inpaint = defaultdict(
        lambda: {"paths": [], "models": [], "mask_label": set(), "mask_path": None}
    )
    t2i = defaultdict(lambda: {"paths": [], "models": []})
    masks = dict()
    infos = dict()

    for p in glob.glob(str(fake_root / "*" / "inpainting" / "*" / "*")):
        pth = Path(p)
        if not pth.is_file():
            if not quiet:
                print(f"Skip {pth}, not a file.")
            continue
        parts = pth.parts
        parts_l = [s.lower() for s in parts]
        try:
            i = parts_l.index("inpainting")
            subset = parts[i - 1]  # 0000
            id_ = parts[i + 1]  # <id>
        except ValueError:
            if not quiet:
                print(f"Skip {pth}, invalid path structure.")
            continue
