import unittest

import torch

from kappadata.functional.onehot import to_onehot_vector, to_onehot_matrix


class TestOnehot(unittest.TestCase):
    def test_to_onehot_vector(self):
        classes = torch.tensor([0, 1, 2, 3])
        n_classes = classes.max() + 1
        onehot_vectors = [to_onehot_vector(y=c, n_classes=n_classes) for c in classes]
        expected = torch.tensor([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ])
        for onehot_vector in onehot_vectors:
            self.assertEqual((4,), onehot_vector.shape)
        self.assertTrue(torch.all(expected == torch.stack(onehot_vectors)))

    def test_to_onehot_vector_noop(self):
        source = torch.tensor([0., 1., 0.])
        actual = to_onehot_vector(y=source, n_classes=3)
        self.assertTrue(torch.all(source == actual))

    def test_to_onehot_matrix(self):
        classes = torch.tensor([0, 1, 5, 3, 4])
        n_classes = classes.max() + 1
        onehot_matrix = to_onehot_matrix(y=classes, n_classes=n_classes)
        expected = torch.tensor([
            [1., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1.],
            [0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 1., 0.],
        ])
        self.assertEqual((5, 6), onehot_matrix.shape)
        self.assertTrue(torch.all(expected == onehot_matrix))

    def test_to_onehot_matrix_noop(self):
        source = torch.eye(5)
        actual = to_onehot_matrix(y=source, n_classes=5)
        self.assertTrue(torch.all(source == actual))
