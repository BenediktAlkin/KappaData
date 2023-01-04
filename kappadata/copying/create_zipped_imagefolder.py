import os
import shutil
from pathlib import Path

def create_zipped_imagefolder_classwise(src, dst):
    src_path = Path(src).expanduser()
    assert src_path.exists(), f"src_path '{src_path}' doesn't exist"
    dst_path = Path(dst).expanduser()
    dst_path.mkdir(exist_ok=True, parents=True)

    for item in os.listdir(src_path):
        src_uri = src_path / item
        if not src_uri.is_dir():
            continue
        shutil.make_archive(
            base_name=dst_path / item,
            format="zip",
            root_dir=src_uri,
        )
