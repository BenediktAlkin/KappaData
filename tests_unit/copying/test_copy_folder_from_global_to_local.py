# noinspection PyPackageRequirements
from pyfakefs.fake_filesystem_unittest import TestCase
from pathlib import Path
from kappadata.copying.copy_folder_from_global_to_local import copy_folder_from_global_to_local
import os
class TestCopyFolderFromGlobalToLocal(TestCase):
    def setUp(self):
        self.setUpPyfakefs()

    def _setup_imagenet(self):
        global_path = Path("/global/imagenet")
        local_path = Path("/local/data")
        self.fs.create_dir(global_path / "train")
        self.fs.create_dir(global_path / "val")
        for split in ["train", "val"]:
            for cls in ["n01560419", "n01630670"]:
                for i in range(10):
                    self.fs.create_file(global_path / split / cls / f"{cls}_{i+1}.JPEG")
        return global_path, local_path

    def _assert_imagenet_split_exists(self, split_path):
        for cls in ["n01560419", "n01630670"]:
            for i in range(10):
                self.assertTrue((split_path / cls / f"{cls}_{i + 1}.JPEG").exists())

    def test_imagenet_autocopy_folder(self):
        global_path, local_path = self._setup_imagenet()
        result = copy_folder_from_global_to_local(global_path, local_path, relative_path="train")
        self.assertTrue(result.was_copied)
        self.assertFalse(result.was_deleted)
        self.assertFalse(result.was_zip)
        self._assert_imagenet_split_exists(local_path / "train")
        self.assertTrue((local_path / "train" / "autocopy_start.txt").exists())
        self.assertTrue((local_path / "train" / "autocopy_end.txt").exists())

    def test_imagenet_already_exists_auto(self):
        global_path, local_path = self._setup_imagenet()
        copy_folder_from_global_to_local(global_path, local_path, relative_path="train")
        self._assert_imagenet_split_exists(local_path / "train")

        result = copy_folder_from_global_to_local(global_path, local_path, relative_path="train")
        self.assertFalse(result.was_copied)
        self.assertFalse(result.was_deleted)
        self.assertFalse(result.was_zip)
        self._assert_imagenet_split_exists(local_path / "train")

    def test_imagenet_incomplete_copy(self):
        global_path, local_path = self._setup_imagenet()
        copy_folder_from_global_to_local(global_path, local_path, relative_path="train")
        self._assert_imagenet_split_exists(local_path / "train")
        # remove end_copy_file
        os.remove(local_path / "train" / "autocopy_end.txt")

        result = copy_folder_from_global_to_local(global_path, local_path, relative_path="train")
        self.assertTrue(result.was_copied)
        self.assertTrue(result.was_deleted)
        self.assertFalse(result.was_zip)
        self._assert_imagenet_split_exists(local_path / "train")
