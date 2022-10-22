# noinspection PyPackageRequirements
from pyfakefs.fake_filesystem_unittest import TestCase
from pathlib import Path
from kappadata.copying.copy_folder_from_global_to_local import copy_folder_from_global_to_local
import os
import shutil


class TestCopyFolderFromGlobalToLocal(TestCase):
    def setUp(self):
        self.setUpPyfakefs()

    def _setup_imagenet(self):
        global_path = Path("/global/imagenet")
        local_path = Path("/local/data")
        (global_path / "train").mkdir(parents=True)
        (global_path / "val").mkdir()
        for split in ["train", "val"]:
            for cls in ["n01560419", "n01630670"]:
                for i in range(10):
                    self.fs.create_file(global_path / split / cls / f"{cls}_{i+1}.JPEG")
        return global_path, local_path

    def _setup_imagenet_zip(self):
        global_path, local_path = self._setup_imagenet()
        # create zip files
        zip_path = Path("/zip/imagenet")
        zip_path.mkdir(parents=True)
        shutil.make_archive(zip_path / "train", 'zip', global_path / "train")
        shutil.make_archive(zip_path / "val", 'zip', global_path / "val")
        return zip_path, local_path

    def _assert_imagenet_split_exists(self, split_path):
        for cls in ["n01560419", "n01630670"]:
            for i in range(10):
                self.assertTrue((split_path / cls / f"{cls}_{i + 1}.JPEG").exists())

    def _test_imagenet_autocopy(self, global_path, local_path, was_zip):
        result = copy_folder_from_global_to_local(global_path, local_path, relative_path="train")
        self.assertTrue(result.was_copied)
        self.assertFalse(result.was_deleted)
        self.assertEqual(was_zip, result.was_zip)
        self._assert_imagenet_split_exists(local_path / "train")
        self.assertTrue((local_path / "train" / "autocopy_start.txt").exists())
        self.assertTrue((local_path / "train" / "autocopy_end.txt").exists())

    def test_imagenet_autocopy_folder(self):
        global_path, local_path = self._setup_imagenet()
        self._test_imagenet_autocopy(global_path, local_path, was_zip=False)

    def test_imagenet_autocopy_zip(self):
        global_path, local_path = self._setup_imagenet_zip()
        self._test_imagenet_autocopy(global_path, local_path, was_zip=True)

    def _test_imagenet_already_exists_auto(self, global_path, local_path):
        copy_folder_from_global_to_local(global_path, local_path, relative_path="train")
        self._assert_imagenet_split_exists(local_path / "train")

        result = copy_folder_from_global_to_local(global_path, local_path, relative_path="train")
        self.assertFalse(result.was_copied)
        self.assertFalse(result.was_deleted)
        # was_zip is never set because nothing is extracted
        self.assertFalse(result.was_zip)
        self._assert_imagenet_split_exists(local_path / "train")

    def test_imagenet_already_exists_auto_folder(self):
        global_path, local_path = self._setup_imagenet()
        self._test_imagenet_already_exists_auto(global_path, local_path)

    def test_imagenet_already_exists_auto_zip(self):
        global_path, local_path = self._setup_imagenet()
        self._test_imagenet_already_exists_auto(global_path, local_path)

    def _test_imagenet_incomplete_copy(self, global_path, local_path, was_zip):
        copy_folder_from_global_to_local(global_path, local_path, relative_path="train")
        self._assert_imagenet_split_exists(local_path / "train")
        # remove end_copy_file
        os.remove(local_path / "train" / "autocopy_end.txt")

        result = copy_folder_from_global_to_local(global_path, local_path, relative_path="train")
        self.assertTrue(result.was_copied)
        self.assertTrue(result.was_deleted)
        self.assertEqual(was_zip, result.was_zip)
        self._assert_imagenet_split_exists(local_path / "train")

    def test_imagenet_incomplete_copy_folder(self):
        global_path, local_path = self._setup_imagenet()
        self._test_imagenet_incomplete_copy(global_path, local_path, was_zip=False)

    def test_imagenet_incomplete_copy_zip(self):
        global_path, local_path = self._setup_imagenet_zip()
        self._test_imagenet_incomplete_copy(global_path, local_path, was_zip=True)