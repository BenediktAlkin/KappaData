import os
import shutil
from pathlib import Path

from pyfakefs.fake_filesystem_unittest import TestCase

from kappadata.copying.folder import copy_folder_from_global_to_local, create_zips
from tests_util.mock_logger import MockLogger


class TestCopyFolderFromGlobalToLocal(TestCase):
    _YT_IDS = [
        "--0AzKXCHj8",
        "--0B3G_C3qc",
        "--0CNhurbZE",
        "--0F7kbzAMA",
        "--0FMNFsVeg",
        "--0MF9K5N30",
        "--0Oh0JxzjQ",
        "--0PQM4-hqg",
        "--0XKTm28ts",
        "--0_x3T5DQI",
        "--0aJtOMp2M",
        "--0bntG9i7E",
        "--0fYwELbpk",
        "--0fim4-6Ig",
        "--0pVlB7mQ8",
        "--0ukMG7yH4",
    ]

    def setUp(self):
        self.setUpPyfakefs()

    def _setup_data(self):
        global_path = Path("/global/dataset")
        local_path = Path("/local/data/dataset")
        (global_path / "train").mkdir(parents=True)
        for vid in self._YT_IDS:
            self.fs.create_file(global_path / "train" / f"{vid}_train.wav")
        return global_path, local_path

    def _setup_data_zip(self):
        global_path, local_path = self._setup_data()
        # create zip files
        zip_path = Path("/zip/dataset")
        zip_path.mkdir(parents=True)
        shutil.make_archive(zip_path / "train", "zip", global_path / "train")
        return zip_path, local_path

    def _setup_data_zips(self):
        global_path, local_path = self._setup_data()
        # create zip files
        zip_path = Path("/zips/dataset")
        create_zips(src=global_path / "train", dst=zip_path / "train", batch_size=5)
        return zip_path, local_path

    def _assert_consistent_copy(self, split_path):
        for ytid in self._YT_IDS:
            self.assertTrue((split_path / f"{ytid}_train.wav").exists())

    def _test_autocopy(self, global_path, local_path, source_format, msg0):
        logger = MockLogger()
        result = copy_folder_from_global_to_local(global_path, local_path, relative_path="train", log_fn=logger)
        self.assertTrue(result.was_copied)
        self.assertFalse(result.was_deleted)
        self.assertEqual(source_format, result.source_format)
        self._assert_consistent_copy(local_path / "train")
        self.assertTrue((local_path / "train" / "autocopy_start.txt").exists())
        self.assertTrue((local_path / "train" / "autocopy_end.txt").exists())
        self.assertEqual(2, len(logger.msgs))
        self.assertTrue(logger.path_msg_equals(msg0, logger.msgs[0]), f"'{msg0}' != '{logger.msgs[0]}'")
        self.assertEqual("finished copying data from global to local", logger.msgs[1])

    def test_autocopy_folder(self):
        global_path, local_path = self._setup_data()
        msg0 = r"copying folders of '/global/dataset/train' to '/local/data/dataset/train'"
        self._test_autocopy(global_path, local_path, source_format="raw", msg0=msg0)

    def test_autocopy_zip(self):
        global_path, local_path = self._setup_data_zip()
        msg0 = "extracting '/zip/dataset/train.zip' to '/local/data/dataset/train'"
        self._test_autocopy(global_path, local_path, source_format="zip", msg0=msg0)

    def test_autocopy_zips(self):
        global_path, local_path = self._setup_data_zips()
        msg0 = "extracting 4 zips from '/zips/dataset/train' to '/local/data/dataset/train' using 0 workers"
        self._test_autocopy(global_path, local_path, source_format="zips", msg0=msg0)

    def _test_already_exists_auto(self, global_path, local_path):
        copy_folder_from_global_to_local(global_path, local_path, relative_path="train")
        self._assert_consistent_copy(local_path / "train")

        logger = MockLogger()
        result = copy_folder_from_global_to_local(global_path, local_path, relative_path="train", log_fn=logger)
        self.assertFalse(result.was_copied)
        self.assertFalse(result.was_deleted)
        # was_zip is never set because nothing is extracted
        self.assertIsNone(result.source_format)
        self._assert_consistent_copy(local_path / "train")

        self.assertEqual(1, len(logger.msgs))
        msg0 = "dataset was already automatically copied '/local/data/dataset/train'"
        self.assertTrue(logger.path_msg_equals(msg0, logger.msgs[0]))

    def test_already_exists_auto_raw(self):
        global_path, local_path = self._setup_data()
        self._test_already_exists_auto(global_path, local_path)

    def test_already_exists_auto_zip(self):
        global_path, local_path = self._setup_data_zip()
        self._test_already_exists_auto(global_path, local_path)

    def test_already_exists_auto_zips(self):
        global_path, local_path = self._setup_data_zips()
        self._test_already_exists_auto(global_path, local_path)

    def _test_incomplete_copy(self, global_path, local_path, source_format, msg1):
        copy_folder_from_global_to_local(global_path, local_path, relative_path="train")
        self._assert_consistent_copy(local_path / "train")
        # remove end_copy_file
        os.remove(local_path / "train" / "autocopy_end.txt")

        logger = MockLogger()
        result = copy_folder_from_global_to_local(global_path, local_path, relative_path="train", log_fn=logger)
        self.assertTrue(result.was_copied)
        self.assertTrue(result.was_deleted)
        self.assertEqual(source_format, result.source_format)
        self._assert_consistent_copy(local_path / "train")

        self.assertEqual(3, len(logger.msgs))
        msg0 = "found incomplete automatic copy in '/local/data/dataset/train' -> deleting folder"
        self.assertTrue(logger.path_msg_equals(msg0, logger.msgs[0]), f"'{msg0}' != '{logger.msgs[0]}'")
        self.assertTrue(logger.path_msg_equals(msg1, logger.msgs[1]), f"'{msg1}' != '{logger.msgs[1]}'")
        self.assertTrue(logger.path_msg_equals("finished copying data from global to local", logger.msgs[2]))

    def test_incomplete_copy_raw(self):
        global_path, local_path = self._setup_data()
        msg1 = "copying folders of '/global/dataset/train' to '/local/data/dataset/train'"
        self._test_incomplete_copy(global_path, local_path, source_format="raw", msg1=msg1)

    def test_incomplete_copy_zip(self):
        global_path, local_path = self._setup_data_zip()
        msg1 = "extracting '/zip/dataset/train.zip' to '/local/data/dataset/train'"
        self._test_incomplete_copy(global_path, local_path, source_format="zip", msg1=msg1)

    def test_incomplete_copy_zips(self):
        global_path, local_path = self._setup_data_zips()
        msg1 = "extracting 4 zips from '/zips/dataset/train' to '/local/data/dataset/train' using 0 workers"
        self._test_incomplete_copy(global_path, local_path, source_format="zips", msg1=msg1)
