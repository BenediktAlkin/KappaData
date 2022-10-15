# adapted from https://github.com/ptrblck/pytorch_misc/blob/master/pytorch_redis.py
import redis
from .cached_dataset import CachedDataset
import torch
import io
import subprocess
import shutil

class RedisDataset(CachedDataset):
    CONNECTION_TRIES = 10
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
        buffer = io.BytesIO(raw_item)
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

    def __init__(
            self,
            *args,
            host="localhost",
            port=6379,
            start_if_not_running=True,
            encode_transforms=None,
            decode_transforms=None,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.db = redis.Redis(host=host, port=port)
        self.db_process = None
        for i in range(self.CONNECTION_TRIES):
            try:
                self.db.ping()
                self.db.flushall()
                break
            except redis.exceptions.ConnectionError:
                msg = f"couldn't find redis server on '{host}:{port}'"
                if start_if_not_running:
                    assert shutil.which("redis-server") is not None, "can't find command 'redis-server'"
                    try_start_local_str = " -> trying to start local server"
                    self.logger.info(f"{msg}{try_start_local_str}")
                    try:
                        self.db_process = subprocess.Popen(["redis-server", "--port", f"{port}"])
                        self.logger.info(f"started redis-server on port {port}")
                    except:
                        raise
                else:
                    try_string = "" if i == 0 else f" {i + 1}/{self.CONNECTION_TRIES}"
                    self.logger.info(f"{msg}{try_string}")
                if i == self.CONNECTION_TRIES - 1:
                    raise

        # store how many items are returned from the dataset
        # e.g. ImageNet returns 2 items (image and class label)
        sample = self.dataset[0]
        self.items_per_sample = len(sample)
        # redis encoder doesn't support all datatypes
        self.encode_transforms = encode_transforms
        # redis returns bytes -> apply a decoding transform to every retrieved sample
        self.decode_transforms = decode_transforms

        # initialize encode_transforms
        if self.encode_transforms is None:
            # automatically initialize encode_transforms from known datatypes
            self.encode_transforms = [self.get_encode_transform(item) for item in sample]
        else:
            assert len(self.encode_transforms) == self.items_per_sample
        # initialize decode_transforms
        if self.decode_transforms is None:
            # automatically initialize decode_transforms from known datatypes
            self.decode_transforms = [self.get_decode_transform(item) for item in sample]
        else:
            assert len(self.decode_transforms) == self.items_per_sample

    def _getitem_impl(self, idx):
        db_idx = idx * self.items_per_sample
        if not self.db.exists(db_idx):
            sample = self.dataset[idx]
            for i, (encode_transform, item) in enumerate(zip(self.encode_transforms, sample)):
                encoded_item = encode_transform(item) if encode_transform is not None else item
                self.db.set(db_idx + i, encoded_item)
        else:
            raw_sample = [self.db.get(db_idx + i) for i in range(self.items_per_sample)]
            sample = tuple(
                decode_transform(raw_item) if decode_transform is not None else raw_item
                for decode_transform, raw_item in zip(self.decode_transforms, raw_sample)
            )
        return sample

    def dispose(self):
        self.db.close()
        if self.db_process is not None:
            self.db_process.close()