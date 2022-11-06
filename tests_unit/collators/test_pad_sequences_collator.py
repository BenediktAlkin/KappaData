import unittest

import torch
from torch.utils.data import DataLoader

from kappadata.collators.base.compose_collator import ComposeCollator
from kappadata.collators.pad_sequences_collator import PadSequencesCollator
from kappadata.wrappers.mode_wrapper import ModeWrapper
from tests_util.sequence_index_dataset import SequenceIndexDataset
from tests_util.sequence_classification_dataset import SequenceClassificationDataset

class TestPadSequencesCollator(unittest.TestCase):
    def test_getitem_x(self):
        rng = torch.Generator().manual_seed(1235)
        maxlen = 5
        lengths = torch.randint(maxlen, size=(4,), generator=rng)
        maxlen = lengths.max().item()
        ds = SequenceIndexDataset(lengths=lengths)
        ds = ModeWrapper(dataset=ds, mode="x")

        collator = ComposeCollator(collators=[PadSequencesCollator()])
        dl = DataLoader(ds, batch_size=len(lengths), collate_fn=collator)
        x = next(iter(dl))

        for i in range(len(lengths)):
            expected = list(range(i, i + lengths[i]))
            actual = x[i][:lengths[i]].tolist()
            self.assertEqual(expected, actual)
            self.assertEqual([0.] * (maxlen - lengths[i]), x[i][lengths[i]:].tolist())

    def test_getitem_x_classes(self):
        rng = torch.Generator().manual_seed(1235)
        n_classes = 4
        expected_seqlen = [4, 2, 3, 1]
        maxlen = max(expected_seqlen)
        expected_classes = [torch.randint(n_classes, size=(seqlen,), generator=rng) for seqlen in expected_seqlen]
        ds = SequenceClassificationDataset(classes=expected_classes)
        ds = ModeWrapper(dataset=ds, mode="x classes seqlen")

        collator = ComposeCollator(collators=[PadSequencesCollator()])
        dl = DataLoader(ds, batch_size=len(expected_classes), collate_fn=collator)
        x, classes, seqlen = next(iter(dl))

        self.assertEqual((4,), seqlen.shape)
        for i in range(len(expected_seqlen)):
            self.assertEqual(expected_seqlen[i], seqlen[i])
            expected_x = list(range(i, i + expected_seqlen[i]))
            actual_x = x[i][:expected_seqlen[i]].tolist()
            self.assertEqual(expected_x, actual_x)
            self.assertEqual([0.] * (maxlen - expected_seqlen[i]), x[i][expected_seqlen[i]:].tolist())

            expected_y = expected_classes[i].tolist()
            actual_y = classes[i][:expected_seqlen[i]].tolist()
            self.assertEqual(expected_y, actual_y)
            self.assertEqual([0.] * (maxlen - expected_seqlen[i]), classes[i][expected_seqlen[i]:].tolist())