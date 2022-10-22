import unittest
from unittest.mock import patch, mock_open
from kappadata.loading.image_folder import raw_image_loader, raw_image_folder_sample_to_pil_sample
from torchvision.transforms.functional import to_tensor
import torch

class TestImageFolder(unittest.TestCase):
    # 2x2 image with a black/red/green/blue pixel
    IMG = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x02\x00\x00\x00\x02\x08\x02\x00\x00\x00\xfd\xd4\x9as\x00\x00\x01\x85iCCPICC profile\x00\x00(\x91}\x91=H\xc3P\x14\x85OS\xa5R+\x0e\x16Qq\xc8P\x9d,\x88\x8a8J\x15\x8b`\xa1\xb4\x15Zu0y\xe9\x8f\xd0\xa4!Iqq\x14\\\x0b\x0e\xfe,V\x1d\\\x9cuup\x15\x04\xc1\x1f\x10G\'\'E\x17)\xf1\xbe\xa4\xd0"\xc6\x07\x97\xf7q\xde;\x87\xfb\xee\x03\x84z\x99\xa9f\xc78\xa0j\x96\x91\x8a\xc7\xc4lnE\x0c\xbc"\x88~\xaaAtK\xcc\xd4\x13\xe9\x85\x0c<\xd7\xd7=||\xbf\x8b\xf2,\xef{\x7f\xae\x1e%o2\xc0\'\x12\xcf2\xdd\xb0\x88\xd7\x89\xa77-\x9d\xf3>q\x98\x95$\x85\xf8\x9cx\xcc\xa0\x06\x89\x1f\xb9.\xbb\xfc\xc6\xb9\xe8\xb0\xc03\xc3F&5G\x1c&\x16\x8bm,\xb71+\x19*\xf1\x14qDQ5\xca\x17\xb2.+\x9c\xb78\xab\xe5*k\xf6\xc9_\x18\xcak\xcbi\xaeS\r#\x8eE$\x90\x84\x08\x19Ul\xa0\x0c\x0bQ\xda5RL\xa4\xe8<\xe6\xe1\x1fr\xfcIr\xc9\xe4\xda\x00#\xc7<*P!9~\xf0?\xf8=[\xb309\xe1&\x85b@\xe7\x8bm\x7f\x8c\x00\x81]\xa0Q\xb3\xed\xefc\xdbn\x9c\x00\xfeg\xe0Jk\xf9+u`\xe6\x93\xf4ZK\x8b\x1c\x01\xbd\xdb\xc0\xc5uK\x93\xf7\x80\xcb\x1d`\xe0I\x97\x0c\xc9\x91\xfcTB\xa1\x00\xbc\x9f\xd17\xe5\x80\xbe[ \xb8\xea\xce\xady\x8e\xd3\x07 C\xb3Z\xba\x01\x0e\x0e\x81\xd1"e\xafy\xbc\xbb\xab}n\xff\xdei\xce\xef\x07R\x82r\x9aO\xbd\x1e\xdb\x00\x00\x00\tpHYs\x00\x00.#\x00\x00.#\x01x\xa5?v\x00\x00\x00\x07tIME\x07\xe6\n\x16\n\x1a\x01\xf9\xb6\rF\x00\x00\x00\x19tEXtComment\x00Created with GIMPW\x81\x0e\x17\x00\x00\x00\x11IDAT\x08\xd7c```\xf8\xcf\x00%\xfe\x03\x00\x0f\xfe\x02\xfe1;%E\x00\x00\x00\x00IEND\xaeB`\x82'


    def test_raw_image_loader(self):
        with patch("builtins.open", mock_open(read_data=self.IMG)) as mock_file:
            self.assertEqual(self.IMG, raw_image_loader("image.png"))
        mock_file.assert_called_with("image.png", "rb")

    def test_raw_image_folder_sample_to_pil_sample(self):
        sample = (self.IMG, 0)
        x, y = raw_image_folder_sample_to_pil_sample(sample)
        self.assertEqual(0, y)
        x_tensor = to_tensor(x)
        # black pixel
        self.assertTrue(torch.all(torch.tensor([0., 0., 0.]) == x_tensor[:, 0, 0]))
        # red pixel
        self.assertTrue(torch.all(torch.tensor([1., 0., 0.]) == x_tensor[:, 0, 1]))
        # green pixel
        self.assertTrue(torch.all(torch.tensor([0., 1., 0.]) == x_tensor[:, 1, 0]))
        # blue pixel
        self.assertTrue(torch.all(torch.tensor([0., 0., 1.]) == x_tensor[:, 1, 1]))