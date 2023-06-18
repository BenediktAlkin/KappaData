import unittest

import torch

from kappadata.transforms import KDAdditiveGaussianNoise, KDIdentityTransform, KDComposeTransform
from kappadata.wrappers.sample_wrappers.kd_multi_view_wrapper import KDMultiViewWrapper
from tests_util.datasets.x_dataset import XDataset


class TestKDMultiViewWrapper(unittest.TestCase):
    def test_ctor(self):
        ds = XDataset(x=torch.randn(10, generator=torch.Generator().manual_seed(5)))
        wrapper0 = KDMultiViewWrapper(dataset=ds, configs=[(2, KDIdentityTransform())])
        wrapper1 = KDMultiViewWrapper(dataset=ds, configs=[(2, None)])
        wrapper2 = KDMultiViewWrapper(dataset=ds, configs=[2])
        wrapper3 = KDMultiViewWrapper(dataset=ds, configs=[KDIdentityTransform()])
        wrapper4 = KDMultiViewWrapper(dataset=ds, configs=[(2, dict(kind="kd_identity_transform"))])
        wrapper5 = KDMultiViewWrapper(dataset=ds, configs=[dict(transform=dict(kind="kd_identity_transform"))])
        wrapper6 = KDMultiViewWrapper(dataset=ds, configs=[
            dict(n_views=2, transform=dict(kind="kd_identity_transform"))
        ])
        wrapper7 = KDMultiViewWrapper(dataset=ds, configs=[dict(n_views=2)])
        wrapper8 = KDMultiViewWrapper(dataset=ds, configs=[
            dict(transform=[dict(kind="kd_identity_transform"), dict(kind="kd_identity_transform")])
        ])
        wrapper9 = KDMultiViewWrapper(dataset=ds, configs=[
            [dict(kind="kd_identity_transform"), dict(kind="kd_identity_transform")],
        ])
        for n_views, wrapper in [
            (2, wrapper0),
            (2, wrapper1),
            (2, wrapper2),
            (1, wrapper3),
            (2, wrapper4),
            (1, wrapper5),
            (2, wrapper6),
            (2, wrapper7),
            (1, wrapper8),
            (1, wrapper9),
        ]:
            self.assertEqual(1, len(wrapper.transform_configs))
            self.assertEqual(n_views, wrapper.transform_configs[0].n_views)
            if isinstance(wrapper.transform_configs[0].transform, KDComposeTransform):
                for transform in wrapper.transform_configs[0].transform.transforms:
                    self.assertIsInstance(transform, KDIdentityTransform)
            else:
                self.assertIsInstance(wrapper.transform_configs[0].transform, KDIdentityTransform)

    def test_2views(self):
        data = torch.randn(10, generator=torch.Generator().manual_seed(5))
        ds = KDMultiViewWrapper(
            dataset=XDataset(x=data),
            configs=[1, (1, lambda x: -x)],
        )
        for i in range(len(ds)):
            sample = ds.getitem_x(i)
            self.assertIsInstance(sample, list)
            self.assertEqual(data[i].item(), sample[0].item())
            self.assertEqual((-data[i]).item(), sample[1].item())

    def test_2views_seed(self):
        data = torch.randn(10, generator=torch.Generator().manual_seed(5))
        ds = KDMultiViewWrapper(
            dataset=XDataset(x=data),
            configs=[KDAdditiveGaussianNoise(std=0.5), KDAdditiveGaussianNoise(std=0.2)],
            seed=3,
        )
        for i in range(len(ds)):
            sample0 = ds.getitem_x(i)
            sample1 = ds.getitem_x(i)
            self.assertIsInstance(sample0, list)
            self.assertIsInstance(sample1, list)
            self.assertEqual(len(sample0), len(sample1))
            for j in range(len(sample0)):
                self.assertEqual(sample0[j].tolist(), sample1[j].tolist())

    def test_2views_identity(self):
        data = torch.randn(10, generator=torch.Generator().manual_seed(5))
        ds = KDMultiViewWrapper(
            dataset=XDataset(x=data),
            configs=[2],
        )
        for i in range(len(ds)):
            sample0 = ds.getitem_x(i)
            sample1 = ds.getitem_x(i)
            self.assertIsInstance(sample0, list)
            self.assertIsInstance(sample1, list)
            self.assertEqual(2, len(sample0))
            self.assertEqual(2, len(sample1))
            for j in range(len(sample0)):
                self.assertEqual(sample0[j].tolist(), sample1[j].tolist())

    def test_2views_identity_dict(self):
        data = torch.randn(10, generator=torch.Generator().manual_seed(5))
        ds = KDMultiViewWrapper(
            dataset=XDataset(x=data),
            configs=[dict(n_views=2)],
        )
        for i in range(len(ds)):
            sample0 = ds.getitem_x(i)
            sample1 = ds.getitem_x(i)
            self.assertIsInstance(sample0, list)
            self.assertIsInstance(sample1, list)
            self.assertEqual(2, len(sample0))
            self.assertEqual(2, len(sample1))
            for j in range(len(sample0)):
                self.assertEqual(sample0[j].tolist(), sample1[j].tolist())
