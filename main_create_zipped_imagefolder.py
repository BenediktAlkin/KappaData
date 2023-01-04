from argparse import ArgumentParser
from kappadata.copying.image_folder import create_zipped_imagefolder_classwise

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--dst", type=str, required=True)
    return vars(parser.parse_args())

def main(src, dst):
    create_zipped_imagefolder_classwise(src=src, dst=dst)


if __name__ == "__main__":
    main(**parse_args())