import torch
from torchvision.transforms.functional import to_pil_image, to_tensor
import unittest
from kappadata.transforms.kd_color_jitter import KDColorJitter
from kappadata.transforms.base.kd_compose_transform import KDComposeTransform


class TestIsDeterministic(unittest.TestCase):
    @staticmethod
    def _create_images():
        return torch.rand(8, 3, 32, 32, generator=torch.Generator().manual_seed(9))

    def _run(self, transform):
        self.assertIsNotNone(transform.seed)
        self._run_single(transform)
        compose = KDComposeTransform([transform])
        self._run_single(compose)

    def _run_single(self, transform):
        images = self._create_images()
        transformed_history = []
        contexts_history = []
        for _ in range(3):
            transform.reset_seed()
            transformed = []
            contexts = []
            for image in images:
                ctx = {}
                transformed.append(transform(image, ctx=ctx))
                contexts.append(ctx)
            transformed = torch.stack(transformed)
            transformed_history.append(transformed)
            contexts_history.append(contexts)
            self.assertTrue(torch.all(transformed_history[0] == transformed))
            self.assertTrue(contexts_history[0] == contexts)

    def test_color_jitter(self):
        transform = KDColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, seed=5)
        self._run(transform)

