import unittest
from dataclasses import dataclass
from unittest.mock import patch

from kappadata.transforms.base.kd_scheduled_transform import KDScheduledTransform
from tests_util.transforms.strength_transform import StrengthTransform


@dataclass
class WorkerInfoMock:
    num_workers: int


class TestKDScheduledTransform(unittest.TestCase):
    def test_mocked(self):
        batch_size = 2
        updates = 5
        worker0 = KDScheduledTransform(transform=StrengthTransform(strength=10.))
        worker1 = KDScheduledTransform(transform=StrengthTransform(strength=10.))
        with patch(
                target="kappadata.transforms.base.kd_transform.get_worker_info",
                new=lambda: WorkerInfoMock(num_workers=2),
        ):
            worker0.worker_init_fn(rank=0, batch_size=batch_size, updates=updates)
            worker1.worker_init_fn(rank=1, batch_size=batch_size, updates=updates)

        strengths = []
        workers = [worker0, worker1]
        for i in range(updates):
            worker_idx = i % len(workers)
            batch = [workers[worker_idx](None, ctx=None) for _ in range(batch_size)]
            strengths += batch

        self.assertEqual([0, 0, 2.5, 2.5, 5, 5, 7.5, 7.5, 10, 10], strengths)
