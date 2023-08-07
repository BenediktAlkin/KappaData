import shutil
from argparse import ArgumentParser
from pathlib import Path
from kappadata.copying.create_zips import create_zips_imagefolder, create_zips_folder


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--dst", type=str, required=True)
    zip_group = parser.add_mutually_exclusive_group(required=True)
    zip_group.add_argument("--zip", action="store_const", dest="zip_format", const="zip")
    zip_group.add_argument("--zips", action="store_const", dest="zip_format", const="zips")
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument("--folder", action="store_const", dest="dataset_format", const="folder")
    dataset_group.add_argument("--image_folder", action="store_const", dest="dataset_format", const="image_folder")
    return vars(parser.parse_args())


def main(src, dst, dataset_format, zip_format):
    print(f"src={src}")
    print(f"dst={dst}")
    print(f"zip_format={zip_format}")
    src = Path(src).expanduser()
    dst = Path(dst).expanduser()
    if zip_format == "zip":
        assert not str(dst).endswith(".zip"), "pass --dst without the .zip ending (appended automatically)"
        assert dst.parent.exists(), dst.as_posix()
        assert not dst.with_suffix(".zip").exists(), dst.as_posix()
        shutil.make_archive(src, "zip", dst)
    elif zip_format == "zips":
        if dataset_format == "image_folder":
            create_zips_imagefolder(src=src, dst=dst)
        elif dataset_format == "folder":
            create_zips_folder(src=src, dst=dst)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main(**parse_args())
