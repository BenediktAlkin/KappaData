from pathlib import Path

import torch
from pyfakefs.fake_filesystem_unittest import TestCase

from kappadata.wrappers.dataset_wrappers.overwrite_classes_wrapper import OverwriteClassesWrapper
from tests_util.datasets.class_dataset import ClassDataset


class TestOverwriteClassesWrapper(TestCase):
    def test_object(self):
        original = [0, 1, 2, 3, 4, 4, 3, 2, 1, 0]
        overwritten = [4, 3, 2, 1, 0, 4, 3, 2, 1, 0]
        ds = OverwriteClassesWrapper(ClassDataset(classes=original), classes=overwritten)
        self.assertEqual(10, len(ds))
        self.assertEqual(overwritten, [ds.getitem_class(i) for i in range(len(ds))])

    def test_file(self):
        self.setUpPyfakefs(additional_skip_names=["torch._functorch.config.os"])
        original = [0, 1, 2, 3, 4, 4, 3, 2, 1, 0]
        overwritten = torch.tensor([4, 3, 2, 1, 0, 4, 3, 2, 1, 0])
        uri = Path("/temp")
        uri.mkdir()
        uri = uri / "overwritten.th"
        with open(uri, "wb") as f:
            torch.save(overwritten, f)
        ds = OverwriteClassesWrapper(ClassDataset(classes=original), uri=uri)
        self.assertEqual(10, len(ds))
        self.assertEqual(overwritten.tolist(), [ds.getitem_class(i) for i in range(len(ds))])
