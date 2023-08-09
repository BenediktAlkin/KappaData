from kappadata.mock.datasets.imagenet import generate_mock_imagenet
from kappadata.mock.datasets.audioset import generate_mock_audioset
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dst", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    return vars(parser.parse_args())

def main(dst, dataset):
    kwargs = dict(dst=dst, log_fn=print)
    if dataset == "imagenet":
        generate_mock_imagenet(**kwargs)
    elif dataset == "audioset":
        generate_mock_audioset(**kwargs)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main(**parse_args())