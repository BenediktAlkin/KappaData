import os
import shutil
from pathlib import Path

from pyfakefs.fake_filesystem_unittest import TestCase

from kappadata.copying.image_folder import copy_imagefolder_from_global_to_local, create_zipped_imagefolder_classwise


class TestCopyFolderFromGlobalToLocal(TestCase):
    class MockLogger:
        def __init__(self):
            self.msgs = []

        def __call__(self, msg):
            self.msgs.append(msg)

        @staticmethod
        def path_msg_equals(expected, actual):
            return expected.replace("\\", "/") == actual.replace("\\", "/")

    def setUp(self):
        self.setUpPyfakefs()

    def _setup_imagenet(self):
        global_path = Path("/global/imagenet")
        local_path = Path("/local/data/imagenet")
        (global_path / "train").mkdir(parents=True)
        (global_path / "val").mkdir()
        for split in ["train", "val"]:
            for cls in ["n01560419", "n01630670"]:
                for i in range(10):
                    self.fs.create_file(global_path / split / cls / f"{cls}_{i + 1}.JPEG")
        return global_path, local_path

    def _setup_imagenet_zip(self):
        global_path, local_path = self._setup_imagenet()
        # create zip files
        zip_path = Path("/zip/imagenet")
        zip_path.mkdir(parents=True)
        shutil.make_archive(zip_path / "train", 'zip', global_path / "train")
        shutil.make_archive(zip_path / "val", 'zip', global_path / "val")
        return zip_path, local_path

    def _setup_imagenet_zip_classwise(self):
        global_path, local_path = self._setup_imagenet()
        # create zip files
        zip_path = Path("/zip/imagenet")
        create_zipped_imagefolder_classwise(src=global_path / "train", dst=zip_path / "train")
        create_zipped_imagefolder_classwise(src=global_path / "val", dst=zip_path / "val")
        return zip_path, local_path

    def _assert_imagenet_split_exists(self, split_path):
        for cls in ["n01560419", "n01630670"]:
            for i in range(10):
                self.assertTrue((split_path / cls / f"{cls}_{i + 1}.JPEG").exists())

    def _test_imagenet_autocopy(self, global_path, local_path, msg0, was_zip=False, was_zip_classwise=False):
        logger = self.MockLogger()
        result = copy_imagefolder_from_global_to_local(global_path, local_path, relative_path="train", log_fn=logger)
        self.assertTrue(result.was_copied)
        self.assertFalse(result.was_deleted)
        self.assertEqual(was_zip, result.was_zip)
        self.assertEqual(was_zip_classwise, result.was_zip_classwise)
        self._assert_imagenet_split_exists(local_path / "train")
        self.assertTrue((local_path / "train" / "autocopy_start.txt").exists())
        self.assertTrue((local_path / "train" / "autocopy_end.txt").exists())
        self.assertEqual(2, len(logger.msgs))
        self.assertTrue(logger.path_msg_equals(msg0, logger.msgs[0]), f"'{msg0}' != '{logger.msgs[0]}'")
        self.assertEqual("finished copying data from global to local", logger.msgs[1])

    def test_imagenet_autocopy_folder(self):
        global_path, local_path = self._setup_imagenet()
        msg0 = r"copying folders of '/global/imagenet/train' to '/local/data/imagenet/train'"
        self._test_imagenet_autocopy(global_path, local_path, was_zip=False, msg0=msg0)

    def test_imagenet_autocopy_zip(self):
        global_path, local_path = self._setup_imagenet_zip()
        msg0 = "extracting '/zip/imagenet/train.zip' to '/local/data/imagenet/train'"
        self._test_imagenet_autocopy(global_path, local_path, was_zip=True, msg0=msg0)

    def test_imagenet_autocopy_zip_classwise(self):
        global_path, local_path = self._setup_imagenet_zip_classwise()
        msg0 = "extracting 2 zips from '/zip/imagenet/train' to '/local/data/imagenet/train' using 0 workers"
        self._test_imagenet_autocopy(global_path, local_path, was_zip_classwise=True, msg0=msg0)

    def _test_imagenet_already_exists_auto(self, global_path, local_path):
        copy_imagefolder_from_global_to_local(global_path, local_path, relative_path="train")
        self._assert_imagenet_split_exists(local_path / "train")

        logger = self.MockLogger()
        result = copy_imagefolder_from_global_to_local(global_path, local_path, relative_path="train", log_fn=logger)
        self.assertFalse(result.was_copied)
        self.assertFalse(result.was_deleted)
        # was_zip is never set because nothing is extracted
        self.assertFalse(result.was_zip)
        self._assert_imagenet_split_exists(local_path / "train")

        self.assertEqual(1, len(logger.msgs))
        msg0 = "dataset was already automatically copied '/local/data/imagenet/train'"
        self.assertTrue(logger.path_msg_equals(msg0, logger.msgs[0]))

    def test_imagenet_already_exists_auto_folder(self):
        global_path, local_path = self._setup_imagenet()
        self._test_imagenet_already_exists_auto(global_path, local_path)

    def test_imagenet_already_exists_auto_zip(self):
        global_path, local_path = self._setup_imagenet()
        self._test_imagenet_already_exists_auto(global_path, local_path)

    def _test_imagenet_incomplete_copy(self, global_path, local_path, was_zip, msg1):
        copy_imagefolder_from_global_to_local(global_path, local_path, relative_path="train")
        self._assert_imagenet_split_exists(local_path / "train")
        # remove end_copy_file
        os.remove(local_path / "train" / "autocopy_end.txt")

        logger = self.MockLogger()
        result = copy_imagefolder_from_global_to_local(global_path, local_path, relative_path="train", log_fn=logger)
        self.assertTrue(result.was_copied)
        self.assertTrue(result.was_deleted)
        self.assertEqual(was_zip, result.was_zip)
        self._assert_imagenet_split_exists(local_path / "train")

        self.assertEqual(3, len(logger.msgs))
        msg0 = "found incomplete automatic copy in '/local/data/imagenet/train' -> deleting folder"
        self.assertTrue(logger.path_msg_equals(msg0, logger.msgs[0]), f"'{msg0}' != '{logger.msgs[0]}'")
        self.assertTrue(logger.path_msg_equals(msg1, logger.msgs[1]), f"'{msg1}' != '{logger.msgs[1]}'")
        self.assertTrue(logger.path_msg_equals("finished copying data from global to local", logger.msgs[2]))

    def test_imagenet_incomplete_copy_folder(self):
        global_path, local_path = self._setup_imagenet()
        msg1 = "copying folders of '/global/imagenet/train' to '/local/data/imagenet/train'"
        self._test_imagenet_incomplete_copy(global_path, local_path, was_zip=False, msg1=msg1)

    def test_imagenet_incomplete_copy_zip(self):
        global_path, local_path = self._setup_imagenet_zip()
        msg1 = "extracting '/zip/imagenet/train.zip' to '/local/data/imagenet/train'"
        self._test_imagenet_incomplete_copy(global_path, local_path, was_zip=True, msg1=msg1)
