# TODO since pytorch 2.1 setUpPyfakefs throws "AttributeError: torch._functorch.config.os does not exist"
# import numpy as np
# from pathlib import Path
#
# import torch
# from pyfakefs.fake_filesystem_unittest import TestCase
#
# from kappadata.wrappers.dataset_wrappers.kd_pseudo_label_wrapper import KDPseudoLabelWrapper
# from tests_util.datasets.class_dataset import ClassDataset
#
#
# class TestKDPseudoLabelWrapper(TestCase):
#     @staticmethod
#     def _setup_pseudo_labels_file(labels, fname):
#         labels = torch.tensor(labels)
#         uri = Path("/temp")
#         uri.mkdir(exist_ok=True)
#         uri = uri / fname
#         with open(uri, "wb") as f:
#             torch.save(labels, f)
#         return uri
#
#     def test_hard(self):
#         self.setUpPyfakefs()
#         original = [0, 1, 2, 3, 4, 4, 3, 2, 1, 0]
#         pseudo_labels = [4, 3, 2, 1, 0, 4, 3, 2, 1, 0]
#         uri = self._setup_pseudo_labels_file(labels=pseudo_labels, fname="hard.th")
#         ds = KDPseudoLabelWrapper(ClassDataset(classes=original), uri=uri)
#         self.assertEqual(10, len(ds))
#         self.assertEqual(5, ds.getshape_class()[0])
#         self.assertEqual(pseudo_labels, [ds.getitem_class(i) for i in range(len(ds))])
#         self.assertEqual(pseudo_labels, ds.getall_class())
#         self.assertEqual(pseudo_labels, [ds.getitem_class(i) for i in range(len(ds))])
#         self.assertEqual(pseudo_labels, ds.getall_class())
#
#     def test_soft_topk_uniform_static(self):
#         self.setUpPyfakefs()
#         original = [0, 1, 2, 3, 4, 4, 3, 2, 1, 0]
#         pseudo_labels = torch.randn(len(original), max(original) + 1, generator=torch.Generator().manual_seed(3657))
#         uri = self._setup_pseudo_labels_file(labels=pseudo_labels, fname="soft_uniform_static.th")
#         ds = KDPseudoLabelWrapper(ClassDataset(classes=original), uri=uri, topk=2, tau=float("inf"), seed=943)
#         self.assertEqual(10, len(ds))
#         expected = [0, 4, 0, 2, 4, 3, 2, 3, 2, 2]
#         self.assertEqual(expected, [ds.getitem_class(i) for i in range(len(ds))])
#         self.assertEqual(expected, [ds.getitem_class(i) for i in range(len(ds))])
#         #self.assertEqual(expected, ds.getall_class())
#
#     def test_soft_topk_weighted_static(self):
#         self.setUpPyfakefs()
#         original = [0, 1, 2, 3, 4, 4, 3, 2, 1, 0]
#         pseudo_labels = torch.randn(len(original), max(original) + 1, generator=torch.Generator().manual_seed(3657))
#         uri = self._setup_pseudo_labels_file(labels=pseudo_labels, fname="soft_weighted_static.th")
#         ds = KDPseudoLabelWrapper(ClassDataset(classes=original), uri=uri, topk=2, tau=1., seed=943)
#         self.assertEqual(10, len(ds))
#         expected = [0, 4, 4, 1, 4, 3, 3, 0, 2, 4]
#         self.assertEqual(expected, [ds.getitem_class(i) for i in range(len(ds))])
#         self.assertEqual(expected, [ds.getitem_class(i) for i in range(len(ds))])
#         #self.assertEqual(expected, ds.getall_class())
#
#     def test_soft_topk_uniform_dynamic(self):
#         rng = np.random.default_rng(seed=9843)
#         class KDPseudoLabelWrapperMock(KDPseudoLabelWrapper):
#             @property
#             def _global_rng(self):
#                 return rng
#
#         self.setUpPyfakefs()
#         original = [0, 1, 2, 3, 4, 4, 3, 2, 1, 0]
#         pseudo_labels = torch.randn(len(original), max(original) + 1, generator=torch.Generator().manual_seed(3657))
#         uri = self._setup_pseudo_labels_file(labels=pseudo_labels, fname="soft_uniform_dynamic.th")
#         ds = KDPseudoLabelWrapperMock(ClassDataset(classes=original), uri=uri, topk=2, tau=float("inf"))
#         self.assertEqual(10, len(ds))
#         self.assertEqual([0, 4, 4, 1, 4, 3, 2, 3, 0, 4], [ds.getitem_class(i) for i in range(len(ds))])
#         self.assertEqual([1, 4, 4, 2, 0, 0, 2, 3, 0, 2], [ds.getitem_class(i) for i in range(len(ds))])
#
#     def test_threshold(self):
#         self.setUpPyfakefs()
#         original = [0, 1, 2, 3, 4, 4, 3, 2, 1, 0]
#         pseudo_labels = torch.randn(len(original), 5, generator=torch.Generator().manual_seed(0))
#         uri = self._setup_pseudo_labels_file(labels=pseudo_labels, fname="threshold.th")
#         ds = KDPseudoLabelWrapper(ClassDataset(classes=original), uri=uri, threshold=0.5)
#         self.assertEqual(10, len(ds))
#         self.assertEqual([4, -1, -1, -1, 3, -1, -1, -1, 3, 1], [ds.getitem_class(i) for i in range(len(ds))])
#
#     def test_shuffle_samplewise(self):
#         self.setUpPyfakefs()
#         pseudo_labels = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
#         uri = self._setup_pseudo_labels_file(labels=pseudo_labels, fname="hard.th")
#         ds = KDPseudoLabelWrapper(
#             ClassDataset(classes=pseudo_labels),
#             uri=uri,
#             shuffle_world_size=4,
#             shuffle_preprocess_mode="shuffle_samplewise",
#             seed=0,
#         )
#         self.assertEqual([1, 1, 1, 0, 0, 0, 0, 1, 0, 0], [ds.getitem_class(i) for i in range(len(ds))])
#
#     def test_shuffle_classwise(self):
#         self.setUpPyfakefs()
#         pseudo_labels = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
#         uri = self._setup_pseudo_labels_file(labels=pseudo_labels, fname="hard.th")
#         ds = KDPseudoLabelWrapper(
#             ClassDataset(classes=pseudo_labels),
#             uri=uri,
#             shuffle_world_size=2,
#             shuffle_preprocess_mode="shuffle_classwise",
#             seed=0,
#         )
#         self.assertEqual(
#             [1, 1, 3, 3, 2, 2, 0, 0, 2, 2, 0, 0, 1, 1, 3, 3],
#             [ds.getitem_class(i) for i in range(len(ds))]
#         )
#
#     def test_shuffle_pseudoclass(self):
#         self.setUpPyfakefs()
#         pseudo_labels = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
#         uri = self._setup_pseudo_labels_file(labels=pseudo_labels, fname="hard.th")
#         ds = KDPseudoLabelWrapper(
#             ClassDataset(classes=list(reversed(pseudo_labels))),
#             uri=uri,
#             shuffle_world_size=2,
#             shuffle_preprocess_mode="shuffle_pseudoclass",
#             seed=0,
#         )
#         self.assertEqual(
#             [1, 1, 3, 3, 2, 2, 0, 0, 2, 2, 0, 0, 1, 1, 3, 3],
#             [ds.getitem_class(i) for i in range(len(ds))]
#         )
