import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from kappadata.collators import KDComposeCollator
from kappadata.common.wrappers import ByolMultiViewWrapper
from kappadata.wrappers import ModeWrapper, XTransformWrapper
from tests_util.collators import AddRandomSequenceCollator
from tests_util.datasets.x_dataset import XDataset
from tests_util.transforms import ReplaceWithRandomTransform, AddRandomTransform


class TestStochasticTransformSeed(unittest.TestCase):
    def test_single_worker_noseed(self):
        dataset = XTransformWrapper(dataset=XDataset(torch.arange(10)), transform=ReplaceWithRandomTransform())
        dataset = ModeWrapper(dataset, mode="x")
        values = torch.concat([x for x in DataLoader(dataset, batch_size=2)])
        self.assertEqual(values.unique().numel(), values.numel())

    def test_single_worker_globalseed(self):
        for _ in range(2):
            np.random.seed(0)
            dataset = XTransformWrapper(dataset=XDataset(torch.arange(10)), transform=ReplaceWithRandomTransform())
            dataset = ModeWrapper(dataset, mode="x")
            values = torch.concat([x for x in DataLoader(dataset, batch_size=2)])
            expected = [
                0.38255389159742137,
                0.34776831374136297,
                0.7534087540864453,
                0.5815352814832978,
                0.6920447974218022,
                0.3324638559200749,
                0.7439437217563455,
                0.6524144695739501,
                0.4358178202087156,
                0.3025380253474623,
            ]
            self.assertEqual(expected, values.tolist())

    def test_two_worker_noseed(self):
        dataset = XTransformWrapper(dataset=XDataset(torch.arange(10)), transform=ReplaceWithRandomTransform())
        dataset = ModeWrapper(dataset, mode="x")
        dataloader = DataLoader(dataset, batch_size=2, num_workers=2, worker_init_fn=dataset.worker_init_fn)
        values = torch.concat([x for x in dataloader])
        self.assertEqual(values.unique().numel(), values.numel())

    def test_two_worker_globalseed(self):
        for _ in range(2):
            torch.manual_seed(0)
            dataset = XTransformWrapper(dataset=XDataset(torch.arange(10)), transform=ReplaceWithRandomTransform())
            dataset = ModeWrapper(dataset, mode="x")
            dataloader = DataLoader(dataset, batch_size=2, num_workers=2, worker_init_fn=dataset.worker_init_fn)
            values = torch.concat([x for x in dataloader])
            expected = [
                0.05265565657170268,
                0.5856545413116812,
                0.5245641455144165,
                0.9032732387757539,
                0.7345538141574777,
                0.9076975596537473,
                0.2264209460224459,
                0.9893990449288441,
                0.7417763462601573,
                0.7998301927398731,
            ]
            self.assertEqual(expected, values.tolist())

    def test_is_num_worker_independent_with_seed(self):
        dataset = XTransformWrapper(dataset=XDataset(torch.arange(10)), transform=ReplaceWithRandomTransform(), seed=0)
        dataset = ModeWrapper(dataset, mode="x")
        dataloader0 = DataLoader(dataset, batch_size=2)
        dataloader1 = DataLoader(dataset, batch_size=2, num_workers=2, worker_init_fn=dataset.worker_init_fn)
        values0 = torch.concat([x for x in dataloader0])
        values1 = torch.concat([x for x in dataloader1])
        self.assertEqual(values0.tolist(), values1.tolist())

    def test_nested_xtransformwrapper(self):
        dataset = XDataset(torch.arange(10))
        dataset = XTransformWrapper(dataset=dataset, transform=ReplaceWithRandomTransform())
        dataset = XTransformWrapper(dataset=dataset, transform=AddRandomTransform())
        dataset = ModeWrapper(dataset, mode="x")
        values = torch.concat([
            torch.concat(x)
            for x in DataLoader(dataset, batch_size=2, worker_init_fn=dataset.worker_init_fn)
        ])
        self.assertEqual(values.unique().numel(), values.numel())

    def test_singleworker_xtransformwrapper_collated(self):
        dataset = XDataset(torch.arange(10), collators=[AddRandomSequenceCollator(dataset_mode="x", return_ctx=False)])
        dataset = XTransformWrapper(dataset=dataset, transform=ReplaceWithRandomTransform())
        dataset = ModeWrapper(dataset, mode="x")
        values = torch.concat([
            x
            for x in DataLoader(
                dataset=dataset,
                batch_size=2,
                collate_fn=dataset.collators[0],
            )
        ])
        self.assertEqual(values.unique().numel(), values.numel())

    def test_twoworker_xtransformwrapper_collated(self):
        dataset = XDataset(torch.arange(10), collators=[AddRandomSequenceCollator(dataset_mode="x", return_ctx=False)])
        dataset = XTransformWrapper(dataset=dataset, transform=ReplaceWithRandomTransform())
        dataset = ModeWrapper(dataset, mode="x")
        values = torch.concat([
            x
            for x in DataLoader(
                dataset=dataset,
                batch_size=2,
                num_workers=2,
                worker_init_fn=dataset.worker_init_fn,
                collate_fn=dataset.collators[0],
            )
        ])
        self.assertEqual(values.unique().numel(), values.numel())

    def test_singleworker_xtransformwrapper_composecollated(self):
        for num_collators in [1, 2]:
            collators = [AddRandomSequenceCollator() for _ in range(num_collators)]
            dataset = XDataset(torch.arange(10), collators=collators)
            dataset = XTransformWrapper(dataset=dataset, transform=ReplaceWithRandomTransform())
            dataset = ModeWrapper(dataset, mode="x")
            values = torch.concat([
                x
                for x in DataLoader(
                    dataset=dataset,
                    batch_size=2,
                    collate_fn=KDComposeCollator(dataset.collators, dataset_mode="x", return_ctx=False),
                )
            ])
            self.assertEqual(values.unique().numel(), values.numel())

    def test_twoworker_xtransformwrapper_composecollated(self):
        for num_collators in [1, 2]:
            collators = [AddRandomSequenceCollator() for _ in range(num_collators)]
            dataset = XDataset(torch.arange(10), collators=collators)
            dataset = XTransformWrapper(dataset=dataset, transform=ReplaceWithRandomTransform())
            dataset = ModeWrapper(dataset, mode="x")
            values = torch.concat([
                x
                for x in DataLoader(
                    dataset=dataset,
                    batch_size=2,
                    num_workers=2,
                    worker_init_fn=dataset.worker_init_fn,
                    collate_fn=KDComposeCollator(dataset.collators, dataset_mode="x", return_ctx=False),
                )
            ])
            self.assertEqual(values.unique().numel(), values.numel())

    def test_deterministic_byolaug(self):
        x = [to_pil_image(xx) for xx in torch.randn(20, 3, 32, 32, generator=torch.Generator().manual_seed(39))]
        dataset = XDataset(x=x)
        dataset = ByolMultiViewWrapper(dataset=dataset, seed=0)
        dataset = ModeWrapper(dataset=dataset, mode="x")

        iter0_workers0 = torch.concat([torch.stack(x, dim=1) for x in DataLoader(dataset, batch_size=4)])
        iter1_workers0 = torch.concat([torch.stack(x, dim=1) for x in DataLoader(dataset, batch_size=2)])
        self.assertTrue(torch.all(iter0_workers0 == iter1_workers0))

        iter0_workers2 = torch.concat([torch.stack(x, dim=1) for x in DataLoader(dataset, batch_size=4, num_workers=2)])
        self.assertTrue(torch.all(iter0_workers0 == iter0_workers2))
        iter1_workers2 = torch.concat([torch.stack(x, dim=1) for x in DataLoader(dataset, batch_size=2, num_workers=2)])
        self.assertTrue(torch.all(iter0_workers0 == iter1_workers2))
