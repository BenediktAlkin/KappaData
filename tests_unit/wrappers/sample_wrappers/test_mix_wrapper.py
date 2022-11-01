import torch
import unittest
from tests_util.classification_dataset import ClassificationDataset
from kappadata.wrappers.sample_wrappers.mix_wrapper import MixWrapper
from kappadata.wrappers.mode_wrapper import ModeWrapper
from torch.utils.data import DataLoader

class TestMixWrapper(unittest.TestCase):
    def test_ctor_arg_checks(self):
        def ctor(**kwargs):
            return MixWrapper(dataset=None, **kwargs)
        
        self.assertRaises(AssertionError, lambda: ctor(cutmix_alpha="a", mixup_alpha=1., p=1., cutmix_p=0.5))
        self.assertRaises(AssertionError, lambda: ctor(cutmix_alpha=None, mixup_alpha=1., p=1., cutmix_p=0.5))
        self.assertRaises(AssertionError, lambda: ctor(cutmix_alpha=-0.1, mixup_alpha=1., p=1., cutmix_p=0.5))

        self.assertRaises(AssertionError, lambda: ctor(cutmix_alpha=1., mixup_alpha="a", p=1., cutmix_p=0.5))
        self.assertRaises(AssertionError, lambda: ctor(cutmix_alpha=1., mixup_alpha=None, p=1., cutmix_p=0.5))
        self.assertRaises(AssertionError, lambda: ctor(cutmix_alpha=1., mixup_alpha=-0.1, p=1., cutmix_p=0.5))

        self.assertRaises(AssertionError, lambda: ctor(cutmix_alpha=1., mixup_alpha=1., p="a", cutmix_p=0.5))
        self.assertRaises(AssertionError, lambda: ctor(cutmix_alpha=1., mixup_alpha=1., p=None, cutmix_p=0.5))
        self.assertRaises(AssertionError, lambda: ctor(cutmix_alpha=1., mixup_alpha=1., p=0., cutmix_p=0.5))
        self.assertRaises(AssertionError, lambda: ctor(cutmix_alpha=1., mixup_alpha=1., p=-2., cutmix_p=0.5))

        self.assertRaises(AssertionError, lambda: ctor(cutmix_alpha=1., mixup_alpha=1., p=1., cutmix_p="a"))
        self.assertRaises(AssertionError, lambda: ctor(cutmix_alpha=1., mixup_alpha=1., p=1., cutmix_p=None))
        self.assertRaises(AssertionError, lambda: ctor(cutmix_alpha=1., mixup_alpha=1., p=1., cutmix_p=0.))
        self.assertRaises(AssertionError, lambda: ctor(cutmix_alpha=1., mixup_alpha=1., p=1., cutmix_p=-2.))
        _ = MixWrapper(
            dataset=ClassificationDataset(x=torch.randn(2, 1), classes=list(range(2))), 
            cutmix_alpha=1.,
            mixup_alpha = 1.,
            p=1.,
            cutmix_p=0.5,
        )

    def test_getitem_x_class(self):
        rng = torch.Generator().manual_seed(63)
        data = torch.randn(8, 1, 16, 16, generator=rng)
        classes = torch.randint(4, size=(len(data),), generator=rng)
        ds = ClassificationDataset(x=data, classes=classes)
        cutmix_ds = MixWrapper(dataset=ds, cutmix_alpha=1., mixup_alpha=1., p=1., cutmix_p=0.5, seed=103)
        loader = DataLoader(ModeWrapper(cutmix_ds, mode="x class", return_ctx=True), batch_size=len(cutmix_ds))
        (x, y), ctxs = next(iter(loader))
        lamb = ctxs["mix_lambda"]
        idx2 = ctxs["mix_idx2"]
        use_cutmix = ctxs["mix_usecutmix"]
        top, left, bot, right = ctxs["mix_bbox"]

        for i in range(len(data)):
            # check "interpolated" with itself
            if idx2[i] == i:
                if use_cutmix[i]:
                    self.assertTrue(torch.all(data[i] == x[i]))
                else:
                    self.assertTrue(torch.allclose(data[i], x[i]))
                self.assertTrue(torch.all(lamb[i] * data[i] + (1. - lamb[i]) * data[i] == x[i]))
                self.assertEquals(1., y[i].max())
                self.assertEquals(1., y[i].sum())
            else:
                if use_cutmix[i]:
                    # check replaced patch
                    replaced_expected = data[idx2[i], :, top[i]:bot[i], left[i]:right[i]]
                    replaced_actual = x[i, :, top[i]:bot[i], left[i]:right[i]]
                    self.assertTrue(torch.all(replaced_expected == replaced_actual))
                    # check original patch
                    self.assertTrue(torch.all(data[i, :, :top[i], :] == x[i, :, :top[i], :]))
                    self.assertTrue(torch.all(data[i, :, bot[i]:, :] == x[i, :, bot[i]:, :]))
                    self.assertTrue(torch.all(data[i, :, :, :left[i]] == x[i, :, :, :left[i]]))
                    self.assertTrue(torch.all(data[i, :, :, right[i]:] == x[i, :, :, right[i]:]))
                else:
                    self.assertTrue(torch.all(lamb[i] * data[i] + (1. - lamb[i]) * data[idx2[i]] == x[i]))


    def test_getitem_class(self):
        rng = torch.Generator().manual_seed(63)
        data = torch.randn(8, 1, 16, 16, generator=rng)
        classes = torch.randint(4, size=(len(data),), generator=rng)
        ds = ClassificationDataset(x=data, classes=classes)
        mix_ds = MixWrapper(dataset=ds, cutmix_alpha=1., mixup_alpha=1., p=1., cutmix_p=0.5, seed=103)
        for i in range(len(data)):
            y = mix_ds.getitem_class(i)
            self.assertEquals(1., y.sum())
            self.assertLessEqual((y != 0.).sum(), 2)