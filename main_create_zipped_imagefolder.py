import shutil
from argparse import ArgumentParser

from kappadata.copying.image_folder import create_zipped_imagefolder_classwise


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--dst", type=str, required=True)
    classwise_group = parser.add_mutually_exclusive_group()
    classwise_group.add_argument("--splitwise", action="store_false", dest="classwise")
    classwise_group.add_argument("--classwise", action="store_true")
    classwise_group.set_defaults(classwise=True)
    return vars(parser.parse_args())


def main(src, dst, classwise):
    if classwise:
        create_zipped_imagefolder_classwise(src=src, dst=dst)
    else:
        shutil.make_archive(src, "zip", dst)


if __name__ == "__main__":
    main(**parse_args())
