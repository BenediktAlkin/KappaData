import numpy as np
from kappadata.datasets.kd_wrapper import KDWrapper
import torch
from pathlib import Path
from kappadata.utils.global_rng import GlobalRng

class KDPseudoLabelWrapper(KDWrapper):
    def __init__(self, dataset, uri, topk=None, tau=None, seed=None, **kwargs):
        super().__init__(dataset=dataset, **kwargs)
        assert len(self.getshape_class()) == 1

        # load pseudo labels
        if uri is not None:
            if not isinstance(uri, Path):
                uri = Path(uri)
            uri = uri.expanduser()
            assert uri.exists(), f"'{uri.as_posix()}' does not exist"
            pseudo_labels = torch.load(uri, map_location="cpu").float()
            assert len(pseudo_labels) == len(self.dataset)
            assert pseudo_labels.ndim == 1 or pseudo_labels.ndim == 2
            if pseudo_labels.ndim == 2 and pseudo_labels.size(1) == 1:
                pseudo_labels = pseudo_labels.squeeze()
            if pseudo_labels.ndim == 2:
                assert pseudo_labels.size(1) == self.getshape_class()[0]
        else:
            raise NotImplementedError

        # set properties
        self.pseudo_labels = pseudo_labels
        self.topk = topk
        self.tau = tau
        self.seed = seed

    @property
    def _global_rng(self):
        return GlobalRng()

    # noinspection PyUnusedLocal
    def getitem_class(self, idx, ctx=None):
        return self._getitem_class(idx).long().item()

    def _getitem_class(self, idx):
        if self.seed is not None:
            # static pseudo labels (they dont change from epoch to epoch)
            rng = np.random.default_rng(seed=self.seed + idx)
        else:
            # dynamic pseudo labels (resampled for every epoch)
            rng = self._global_rng

        if self.topk is not None:
            # sample from topk
            assert self.pseudo_labels.ndim == 2
            topk_probs, topk_idxs = self.pseudo_labels[idx].topk(k=self.topk)
            if self.tau == float("inf"):
                # uniform sample
                choice = rng.integers(self.topk)
            else:
                if self.tau is not None:
                    # labels are logits -> divide by temperature and apply softmax
                    weights = topk_probs.div_(self.tau).softmax(dim=0)
                else:
                    # labels are probabilities
                    weights = topk_probs
                # NOTE: argmax is required because np.random.multinomial "counts" the number of outcomes
                # so if multinomial of 5 values draws the 4th value and 1 trial is used the outcome would
                # be [0, 0, 0, 1, 0] -> with argmax -> 4
                choice = rng.multinomial(1, weights).argmax()
            return topk_idxs[choice]

        # hard labels
        assert self.tau is None
        if self.pseudo_labels.ndim == 1:
            return self.pseudo_labels[idx]
        elif self.pseudo_labels.ndim == 2:
            return self.pseudo_labels[idx].argmax()
        else:
            raise NotImplementedError

    def getall_class(self):
        if self.pseudo_labels.ndim == 1:
            return self.pseudo_labels.tolist()
        raise NotImplementedError