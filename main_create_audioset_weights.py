from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "src",
        type=str,
        help="path to the csv file (e.g. /data/audioset/unbalanced_train_samples.csv)",
    )
    return vars(parser.parse_args())


def main(src):
    src = Path(src).expanduser()
    assert src.exists() and src.name.endswith(".csv")

    # load labels
    print(f"loading labels from {src.as_posix()}")
    with open(src) as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    # load and remove metadata
    num_ytids, _, num_unique_labels, num_positive_labels = map(int, lines[1][2:].replace("=", ", ").split(", ")[1::2])
    lines = lines[3:]
    assert num_ytids == len(lines)

    # count class frequencies
    print(f"counting class frequencies of {len(lines)} samples")
    class_counts = defaultdict(int)
    all_class_ids = []
    for line in tqdm(lines):
        class_ids = line.split(", ")[3][1:-1].split(",")
        all_class_ids.append(class_ids)
        for class_id in class_ids:
            class_counts[class_id] += 1
    assert len(class_counts) == num_unique_labels
    assert sum(class_counts.values()) == num_positive_labels

    # calculate weights for classes (as in AudioMAE Appendix B https://arxiv.org/abs/2207.06405)
    class_to_weight = {k: 1000 / (v + 0.01) for k, v in class_counts.items()}

    # generate weights
    print(f"generating sample weights")
    weights = []
    for class_ids in all_class_ids:
        weight = 0
        for class_id in class_ids:
            weight += class_to_weight[class_id]
        weights.append(weight)

    # save as tensor
    dst = src.parent / f"{src.with_suffix('').name}_weights.pth"
    print(f"save sample weights to {dst}")
    torch.save(torch.tensor(weights), dst)


if __name__ == "__main__":
    main(**parse_args())
