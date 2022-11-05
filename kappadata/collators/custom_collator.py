from torch.utils.data import default_collate

class CustomCollator:
    def __init__(self, collators):
        assert isinstance(collators, list)
        self.collators = collators

    def __call__(self, batch):
        batch = default_collate(batch)
        for collator in self.collators:
            batch = collator(batch)
        return batch