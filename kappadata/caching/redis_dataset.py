# adapted from https://github.com/ptrblck/pytorch_misc/blob/master/pytorch_redis.py
import redis
from .cached_dataset import CachedDataset
import torch
import io

class RedisDataset(CachedDataset):
    CUSTOM_ENCODE_TRANSFORMS = []
    CUSTOM_DECODE_TRANSFORMS = []

    @staticmethod
    def _tensor_to_bytes(tensor):
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        buffer.seek(0)
        return buffer.read()

    @staticmethod
    def _tensor_from_bytes(raw_item):
        buffer = io.BytesIO()
        buffer.write(raw_item)
        buffer.seek(0)
        return torch.load(buffer)

    @staticmethod
    def get_encode_transform(sample_item):
        for custom_encode_transform in RedisDataset.CUSTOM_ENCODE_TRANSFORMS:
            data_type_matched, encode_transform = custom_encode_transform(sample_item)
            if data_type_matched:
                return encode_transform
        if isinstance(sample_item, bool):
            return int
        if torch.is_tensor(sample_item):
            return RedisDataset._tensor_to_bytes
        return None

    @staticmethod
    def get_decode_transform(sample_item):
        for custom_decode_transform in RedisDataset.CUSTOM_DECODE_TRANSFORMS:
            data_type_matched, decode_transform = custom_decode_transform(sample_item)
            if data_type_matched:
                return decode_transform
        if isinstance(sample_item, int):
            return int
        if isinstance(sample_item, float):
            return float
        if isinstance(sample_item, str):
            return lambda raw_item: raw_item.decode("utf-8")
        if isinstance(sample_item, bool):
            return lambda b: bool(int(b))
        if torch.is_tensor(sample_item):
            return RedisDataset._tensor_from_bytes
        return None

    def __init__(self, dataset, host, port, transform=None, encode_transforms=None, decode_transforms=None):
        self.db = redis.Redis(host=host, port=port)
        self.dataset = dataset
        self.transform = transform
        # store how many items are returned from the dataset
        # e.g. ImageNet returns 2 items (image and class label)
        self.items_per_sample = None
        # redis encoder doesn't support all datatypes
        self.encode_transforms = encode_transforms
        # redis returns bytes -> apply a decoding transform to every retrieved sample
        self.decode_transforms = decode_transforms

    def __getitem__(self, index):
        if not self.db.exists(index):
            sample = self.dataset[index]
            # initialize items_per_sample and encode_transforms/decode_transforms
            if self.items_per_sample is None:
                self.items_per_sample = len(sample)
                if self.encode_transforms is None:
                    # automatically initialize encode_transforms from known datatypes
                    self.encode_transforms = [
                        self.get_encode_transform(item)
                        for item in sample
                    ]
                if self.decode_transforms is None:
                    # automatically initialize decode_transforms from known datatypes
                    self.decode_transforms = [
                        self.get_decode_transform(item)
                        for item in sample
                    ]
                else:
                    assert len(self.decode_transforms) == self.items_per_sample

            # store in db
            db_idx = index * self.items_per_sample
            for i, (encode_transform, item) in enumerate(zip(self.encode_transforms, sample)):
                encoded_item = encode_transform(item) if encode_transform is not None else item
                self.db.set(db_idx + i, encoded_item)
        else:
            db_idx = index * self.items_per_sample
            raw_sample = [self.db.get(db_idx + i) for i in range(self.items_per_sample)]
            sample = [
                decode_transform(raw_item) if decode_transform is not None else raw_item
                for decode_transform, raw_item in zip(self.decode_transforms, raw_sample)
            ]

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.dataset)

    def dispose(self):
        self.db.close()