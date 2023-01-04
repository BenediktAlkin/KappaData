from argparse import ArgumentParser
from kappadata.copying.zipped_imagefolder import unzip_imagefolder_classwise

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--dst", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=0)
    return vars(parser.parse_args())

def main(src, dst, num_workers):
    unzip_imagefolder_classwise(src=src, dst=dst, num_workers=num_workers)


if __name__ == "__main__":
    main(**parse_args())