from torch.utils.data import DataLoader, RandomSampler, Dataset
from kappadata.batch_samplers.infinite_batch_sampler import InfiniteBatchSampler
from time import sleep

# noinspection PyPackageRequirements
import kappaprofiler as kp


class SleepDataset(Dataset):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def __getitem__(self, idx):
        sleep(0.1)
        return idx

    def __len__(self):
        return self.size

def main():
    dataset = SleepDataset(size=1000)
    batch_size = 256
    epochs = 10
    num_workers = 10

    # torch
    # loader = DataLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     drop_last=True,
    #     num_workers=num_workers,
    #     persistent_workers=True,
    # )
    # with kp.Stopwatch() as sw1:
    #     for i in range(epochs):
    #         print(f"epoch {i+1}")
    #         iterator = iter(loader)
    #         batch = 0
    #         while True:
    #             try:
    #                 with kp.Stopwatch() as sw2:
    #                     next(iterator)
    #                 batch += 1
    #                 print(f"batch {batch} {sw2.elapsed_seconds}")
    #             except StopIteration:
    #                 break
    # print(sw1.elapsed_seconds)

    # kd
    sampler = RandomSampler(data_source=dataset)
    batch_sampler = InfiniteBatchSampler(sampler=sampler, batch_size=batch_size, drop_last=True)
    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        persistent_workers=True,
    )
    with kp.Stopwatch() as sw1:
        iterator = iter(loader)
        batches_per_epoch = len(dataset) // batch_size
        for i in range(epochs):
            print(f"epoch {i+1}")
            for j in range(batches_per_epoch):
                with kp.Stopwatch() as sw2:
                    x = next(iterator)
                print(f"batch {j+1} {sw2.elapsed_seconds} {x.tolist()}")
    print(sw1.elapsed_seconds)




if __name__ == "__main__":
    main()