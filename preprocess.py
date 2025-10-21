import os
import re
import glob
import json
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


def parse_mask(fake_root: Path, quiet: bool = True) -> dict:
    """
    Parse the mask images from the fake image directory.
    The expected directory structure is as follows:
    fake_root/
    └─ 0000/
        └─ mask/
            └─ <id>/
                └─ mask_<id>_<label>.png
                └─ <label>.png  (alternative simpler naming)
                └─ mask_<label>.png  (alternative simpler naming)

    Args:
        fake_root (Path): The root directory containing fake images.

    Returns:
        dict: A dictionary mapping (subset, id) to mask information.
        - masks[(subset, id)] = { 'path': Path, 'labels': { <label>: Path, ... } }
    """
    masks = {}
    re_name = re.compile(
        r"(?i)^mask_(?P<fid>[^_]+)_(?P<label>[^.]+)\.(?:png|jpg|jpeg|bmp|webp)$"
    )
    re_simple = re.compile(r"(?i)^(?P<label>[^.]+)\.(?:png|jpg|jpeg|bmp|webp)$")

    for p in glob.glob(str(fake_root / "*" / "mask" / "*" / "*")):
        pth = Path(p)
        if not pth.is_file() or not is_img(pth):
            continue
        parts = pth.parts
        parts_l = [s.lower() for s in parts]
        try:
            i = parts_l.index("mask")
            subset = parts[i - 1]  # 0000
            id_dir = parts[i + 1]  # <id>
        except Exception as e:
            if not quiet:
                logging.warning(f"Skip mask {pth}: {e}")
            continue

        stem = pth.stem  # remove extension
        name_lower = pth.name.lower()

        if name_lower.startswith("mask_"):
            rest = stem[5:]  # remove 'mask_'
            id_prefix = f"{id_dir}_"
            if rest.startswith(id_prefix):
                # mask_<id>_<label>.png
                label = rest[len(id_prefix) :]
            else:
                # mask_<label>.png
                label = rest
            id_from_name = id_dir
        else:
            # simpler naming: <label>.png
            label = stem
            id_from_name = id_dir

        if id_from_name != id_dir and not quiet:
            logging.warning(
                f"ID mismatch in mask file: dir={id_dir}, name={id_from_name} in {pth}"
            )

        key = (subset, id_dir)
        if key not in masks:  # new hosting entry
            masks[key] = {"path": pth, "labels": {}}

        if label in masks[key]["labels"]:
            continue  # already have this label

        masks[key]["labels"][label] = pth  # one entry can have multiple labels & masks

    return masks


def _parse_txt_json(text: str) -> dict:
    """
    Parse the .txt info file content into a dictionary.
    .txt file should follow the format:
    ["description", ["object1", "object2", ...]]

    Args:
        text (str): The content of the text file, including a short description and an object list.

    Returns:
        dict: A dictionary containing parsed key-value pairs.
    """
    text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, list) and len(data) >= 2:
            desc = str(data[0])
            objs = [
                str(x) for x in (data[1 if isinstance(data[1], (list, tuple)) else []])
            ]
            return {"description": desc, "objects": objs}
    except Exception:
        logging.warning(f"Failed to parse {text} as JSON.")
        pass


def parse_info(fake_root: Path, quiet: bool = True) -> dict:
    """
    Parse the image info text files from the fake image directory.
    The expected directory structure is as follows:
    fake_root/
    └─ 0000/
        └─ image_info/
            └─ <id>.txt

    Args:
        fake_root (Path): The root directory containing fake images.

    Returns:
        dict: A dictionary mapping (subset, id) to the parsed info dictionary.
        - infos[(subset, id)] = { 'description': str, 'objects': [str, ...] }
    """
    infos = {}
    for p in glob.glob(str(fake_root / "*" / "image_info" / "*")):
        pth = Path(p)
        if not pth.is_file() or pth.suffix.lower() != ".txt":
            continue
        parts = pth.parts
        parts_l = [s.lower() for s in parts]
        try:
            i = parts_l.index("image_info")
            subset = parts[i - 1]  # 0000
            id_ = pth.stem  # <id>
        except Exception as e:
            if not quiet:
                logging.warning(f"Skip info {pth}: {e}")
            continue

        try:
            text = pth.read_text(encoding="utf-8", errors="ignore")
            payload = _parse_txt_json(text)

            if not payload or "description" not in payload or "objects" not in payload:
                raise ValueError("Missing required fields in info file.")

            infos[(subset, id_)] = payload
        except Exception as e:
            if not quiet:
                logging.warning(f"Failed to read/parse info file {pth}: {e}")
            continue
            infos[(subset, id_)] = {"description": "", "objects": []}
    return infos


def parse_inpaint(
    fake_root: Path,
    quiet: bool = True,
    masks: dict = None,
    infos: dict = None,
    parse_record=True,
) -> dict:
    """
    Parse the inpainting images from the fake image directory.
    The expected directory structure is as follows:
    fake_root/
    └─ 0000/
        └─ inpainting/
            └─ <id>/
                └─ (SD|SDXL|FLUX1)_inpainting_<id>_<label>.png

    Args:
        fake_root (Path): The root directory containing fake images.
        quiet (bool, optional): Whether to suppress warnings. Defaults to True.
        masks (dict, optional): A dictionary of masks parsed from parse_mask(). Defaults to None
        infos (dict, optional): A dictionary of infos parsed from parse_info(). Defaults to None
        parse_record (bool, optional): Whether to return the record list summarizing all files for a sample. Defaults to True.

    Returns:
        dict: A dictionary mapping (subset, id) to inpainting image information.
        - inpaint[(subset, idx)] = { 'paths': [Path...], 'models': [SD/SDXL/FLUX1...], 'mask_label': set(...), 'mask_path': Path|None }

        if parse_record is True, also returns:
        - records: A list of dictionaries summarizing each inpainting image file.

    """
    inpaint = defaultdict(
        lambda: {"paths": [], "models": [], "mask_label": set(), "mask_path": None}
    )
    records = []

    re_inp = re.compile(
        r"(?i)^(SD|SDXL|FLUX1)_inpainting_(?P<id>[^_]+)_(?P<label>[^.]+)\.(?:png|jpg|jpeg|bmp|webp)$"
    )

    for p in glob.glob(str(fake_root / "*" / "inpainting" / "*" / "*")):
        pth = Path(p)
        if not pth.is_file() or not is_img(pth):
            continue

        parts = pth.parts
        parts_l = [s.lower() for s in parts]
        try:
            i = parts_l.index("inpainting")
            subset = parts[i - 1]  # 0000
            id_dir = parts[i + 1]  # <id>
        except Exception as e:
            if not quiet:
                logging.warning(f"Skip inpaint {pth}: {e}")
            continue

        m = re_inp.match(pth.name)
        if not m:
            if not quiet:
                logging.warning(f"Filename not matched (inpainting): {pth.name}")
            continue

        model = m.group(1).upper()
        id_from_name = m.group("id")
        label = m.group("label")

        if id_from_name != id_dir and not quiet:
            logging.warning(
                f"ID mismatch in inpainting file: dir={id_dir}, name={id_from_name} in {pth}"
            )

        key = (subset, id_dir)
        inpaint[key]["paths"].append(pth)
        inpaint[key]["models"].append(model)
        inpaint[key]["mask_label"].add(label)

        mask_path = None

        # append the label-specific mask path if available
        if masks and key in masks and label in masks[key]["labels"]:
            mask_path = masks[key]["labels"][label]
            inpaint[key]["mask_path"] = mask_path

        if infos:
            info_path = infos.get(key, None)

        if parse_record:
            records.append(
                {
                    "subset": subset,
                    "id": id_dir,
                    "task": "inpainting",
                    "model": model,
                    "inpaint_path": pth,
                    "mask_label": label,
                    "mask_path": mask_path,
                    "info": info_path,
                }
            )

    if parse_record:
        return inpaint, records

    return inpaint


def parse_t2i(
    fake_root: Path, quiet: bool = True, infos=None, parse_record=True
) -> dict:
    """
    Parse the text-to-image images from the fake image directory.

    The expected directory structure is as follows:
    fake_root/
    └─ 0000/
        └─ text2image/
            └─ <id>/
                └─ (SD|SDXL|FLUX1)_text2image_<id>.png

    Args:
        fake_root (Path): The root directory containing fake images.
        quiet (bool, optional): Whether to suppress warnings. Defaults to True.
        infos (dict, optional): A dictionary of infos parsed from parse_info(). Defaults to None
        parse_record (bool, optional): Whether to return the record list summarizing all files for
        a sample. Defaults to True.

    Returns:
        dict: A dictionary mapping (subset, id) to text-to-image information.
        - t2i[(subset, idx)]     = { 'paths': [Path...], 'models': [SD/SDXL/FLUX1...] }
        if parse_record is True, also returns:
        - records: A list of dictionaries summarizing each text-to-image file.
    """
    t2i = defaultdict(lambda: {"paths": [], "models": []})
    records = []

    re_t2i = re.compile(
        r"(?i)^(SD|SDXL|FLUX1)_text2image_(?P<id>[^_.]+)\.(?:png|jpg|jpeg|bmp|webp)$"
    )

    for p in glob.glob(str(fake_root / "*" / "text2image" / "*" / "*")):
        pth = Path(p)
        if not pth.is_file() or not is_img(pth):
            continue

        parts = pth.parts
        parts_l = [s.lower() for s in parts]
        try:
            i = parts_l.index("text2image")
            subset = parts[i - 1]  # 0000
            id_dir = parts[i + 1]  # <id>
        except Exception as e:
            if not quiet:
                logging.warning(f"Skip t2i {pth}: {e}")
            continue

        m = re_t2i.match(pth.name)
        if not m:
            if not quiet:
                logging.warning(f"Filename not matched (text2image): {pth.name}")
            continue

        model = m.group(1).upper()
        id_from_name = m.group("id")

        if id_from_name != id_dir and not quiet:
            logging.warning(
                f"ID mismatch in text2image file: dir={id_dir}, name={id_from_name} in {pth}"
            )

        key = (subset, id_dir)
        t2i[key]["paths"].append(pth)
        t2i[key]["models"].append(model)

        info_path = infos.get(key, None) if infos else None
        records.append(
            {
                "subset": subset,
                "id": id_dir,
                "task": "text2image",
                "model": model,
                "img_path": pth,
                "mask_label": None,
                "mask_path": None,
                "info": info_path,
            }
        )

    if parse_record:
        return t2i, records

    return t2i


def parse_fake_path(fake_root: Path, quiet: bool = True) -> dict:
    """
    Parse the fake image directory to organize the information.

    The expected directory structure is as follows:
    fake_images/
    ├─ 0000/
    │   ├─ inpainting/<id>/(SD|SDXL|FLUX1)_inpainting_<id>_<label>.png
    │   ├─ text2image/<id>/(SD|SDXL|FLUX1)_text2image_<id>.png
    │   ├─ mask/<id>/mask_<id>_<label>.png
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

    masks = parse_mask(fake_root, quiet=quiet)
    infos = parse_info(fake_root, quiet=quiet)
    inpaint, inpaint_records = parse_inpaint(
        fake_root, quiet=quiet, masks=masks, infos=infos, parse_record=True
    )
    t2i, t2i_records = parse_t2i(fake_root, quiet=quiet, infos=infos, parse_record=True)
    records = inpaint_records + t2i_records
    return {
        "inpaint": inpaint,
        "t2i": t2i,
        "masks": masks,
        "infos": infos,
        "records": records,
    }
