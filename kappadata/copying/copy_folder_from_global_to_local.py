import shutil
import zipfile
from collections import namedtuple
from pathlib import Path

CopyResult = namedtuple("CopyFolderResult", "was_copied was_deleted was_zip")


def _log(log, msg):
    if log is not None:
        log(msg)


def copy_folder_from_global_to_local(global_path, local_path, relative_path=None, log=None):
    if not isinstance(global_path, Path):
        global_path = Path(global_path).expanduser()
    if not isinstance(local_path, Path):
        local_path = Path(local_path).expanduser()
    if relative_path is not None and not isinstance(relative_path, Path):
        # relative path can be .zip -> relative path is always without .zip as dst_path is never .zip
        relative_path = Path(relative_path).with_suffix("")

    # check src_path exists (src_path can be folder or .zip)
    src_path = global_path / relative_path if relative_path is not None else global_path
    assert (src_path.exists() and src_path.is_dir()) or src_path.with_suffix(".zip").exists(), \
        f"src_path doesn't exist (can be folder or .zip) '{src_path}'"

    # if dst_path exists:
    # - autocopy start/end file exists -> already copied -> do nothing
    # - autocopy start file exists && autocopy end file doesn't exist -> incomplete copy -> delete and copy again
    # - autocopy start file doesn't exists -> manually copied dataset -> do nothing
    dst_path = local_path / relative_path if relative_path is not None else local_path
    start_copy_file = dst_path / "autocopy_start.txt"
    end_copy_file = dst_path / "autocopy_end.txt"
    was_deleted = False
    if dst_path.exists():
        if start_copy_file.exists():
            if end_copy_file.exists():
                # already automatically copied -> do nothing
                _log(log, f"dataset was already automatically copied '{dst_path}'")
                return CopyResult(was_copied=False, was_deleted=False, was_zip=False)
            else:
                # incomplete copy -> delete and copy again
                _log(log, f"found incomplete automatic copy in '{dst_path}' -> deleting folder")
                shutil.rmtree(dst_path)
                was_deleted = True
                dst_path.mkdir()
        else:
            _log(log, f"using manually copied dataset '{dst_path}'")
            return CopyResult(was_copied=False, was_deleted=False, was_zip=False)
    else:
        dst_path.mkdir(parents=True)

    # create start_copy_file
    with open(start_copy_file, "w") as f:
        f.write("this file indicates that an attempt to copy the dataset automatically was started")

    # copy
    was_zip = False
    if src_path.exists() and src_path.is_dir():
        _log(log, f"copying '{src_path}' to '{dst_path}'")
        # copy folder (dirs_exist_ok=True because dst_path is created for start_copy_file)
        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
    elif src_path.with_suffix(".zip").exists():
        _log(log, f"extracting '{src_path.with_suffix('.zip')}' to '{dst_path}'")
        # extract zip
        was_zip = True
        with zipfile.ZipFile(src_path.with_suffix(".zip")) as f:
            f.extractall(dst_path)
    else:
        raise NotImplementedError

    # create end_copy_file
    with open(end_copy_file, "w") as f:
        f.write("this file indicates that the copying the dataset automatically was successful")

    _log(log, "finished copying data from global to local")
    return CopyResult(was_copied=True, was_deleted=was_deleted, was_zip=was_zip)
