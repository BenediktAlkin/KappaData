from argparse import ArgumentParser
from kappadata.copying.image_folder import unzip_imagefolder_classwise
from time import time

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--dst", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=0)
    return vars(parser.parse_args())

def main(src, dst, num_workers):
    start_time = time()
    unzip_imagefolder_classwise(src=src, dst=dst, num_workers=num_workers)
    end_time = time()
    print(f"unzipping took {end_time - start_time}s")


if __name__ == "__main__":
    main(**parse_args())