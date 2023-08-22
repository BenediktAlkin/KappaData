import numpy as np
from pathlib import Path

import torch
from pyfakefs.fake_filesystem_unittest import TestCase

from kappadata.wrappers.dataset_wrappers.kd_pseudo_label_wrapper import KDPseudoLabelWrapper
from tests_util.datasets.class_dataset import ClassDataset


class TestPseudoLabelWrapper(TestCase):
    @staticmethod
    def _setup_pseudo_labels_file(labels, fname):
        labels = torch.tensor(labels)
        uri = Path("/temp")
        uri.mkdir(exist_ok=True)
        uri = uri / fname
        with open(uri, "wb") as f:
            torch.save(labels, f)
        return uri

    def test_hard(self):
        self.setUpPyfakefs()
        original = [0, 1, 2, 3, 4, 4, 3, 2, 1, 0]
        pseudo_labels = [4, 3, 2, 1, 0, 4, 3, 2, 1, 0]
        uri = self._setup_pseudo_labels_file(labels=pseudo_labels, fname="hard.th")
        ds = KDPseudoLabelWrapper(ClassDataset(classes=original), uri=uri)
        self.assertEqual(10, len(ds))
        self.assertEqual(pseudo_labels, [ds.getitem_class(i) for i in range(len(ds))])
        self.assertEqual(pseudo_labels, ds.getall_class())
        self.assertEqual(pseudo_labels, [ds.getitem_class(i) for i in range(len(ds))])
        self.assertEqual(pseudo_labels, ds.getall_class())

    def test_soft_topk_uniform_static(self):
        self.setUpPyfakefs()
        original = [0, 1, 2, 3, 4, 4, 3, 2, 1, 0]
        pseudo_labels = torch.randn(len(original), max(original) + 1, generator=torch.Generator().manual_seed(3657))
        uri = self._setup_pseudo_labels_file(labels=pseudo_labels, fname="soft_uniform_static.th")
        ds = KDPseudoLabelWrapper(ClassDataset(classes=original), uri=uri, topk=2, tau=float("inf"), seed=943)
        self.assertEqual(10, len(ds))
        expected = [0, 4, 0, 2, 4, 3, 2, 3, 2, 2]
        self.assertEqual(expected, [ds.getitem_class(i) for i in range(len(ds))])
        self.assertEqual(expected, [ds.getitem_class(i) for i in range(len(ds))])
        #self.assertEqual(expected, ds.getall_class())

    def test_soft_topk_weighted_static(self):
        self.setUpPyfakefs()
        original = [0, 1, 2, 3, 4, 4, 3, 2, 1, 0]
        pseudo_labels = torch.randn(len(original), max(original) + 1, generator=torch.Generator().manual_seed(3657))
        uri = self._setup_pseudo_labels_file(labels=pseudo_labels, fname="soft_weighted_static.th")
        ds = KDPseudoLabelWrapper(ClassDataset(classes=original), uri=uri, topk=2, tau=1., seed=943)
        self.assertEqual(10, len(ds))
        expected = [0, 4, 4, 1, 4, 3, 3, 0, 2, 4]
        self.assertEqual(expected, [ds.getitem_class(i) for i in range(len(ds))])
        self.assertEqual(expected, [ds.getitem_class(i) for i in range(len(ds))])
        #self.assertEqual(expected, ds.getall_class())

    def test_soft_topk_uniform_dynamic(self):
        rng = np.random.default_rng(seed=9843)
        class KDPseudoLabelWrapperMock(KDPseudoLabelWrapper):
            @property
            def _global_rng(self):
                return rng

        self.setUpPyfakefs()
        original = [0, 1, 2, 3, 4, 4, 3, 2, 1, 0]
        pseudo_labels = torch.randn(len(original), max(original) + 1, generator=torch.Generator().manual_seed(3657))
        uri = self._setup_pseudo_labels_file(labels=pseudo_labels, fname="soft_uniform_dynamic.th")
        ds = KDPseudoLabelWrapperMock(ClassDataset(classes=original), uri=uri, topk=2, tau=float("inf"))
        self.assertEqual(10, len(ds))
        self.assertEqual([0, 4, 4, 1, 4, 3, 2, 3, 0, 4], [ds.getitem_class(i) for i in range(len(ds))])
        self.assertEqual([1, 4, 4, 2, 0, 0, 2, 3, 0, 2], [ds.getitem_class(i) for i in range(len(ds))])
