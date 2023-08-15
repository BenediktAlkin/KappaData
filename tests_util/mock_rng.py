from collections import defaultdict

class MockRng:
    def __init__(self, rng_values):
        assert isinstance(rng_values, dict)
        self.rng_values = rng_values
        self.counter = defaultdict(int)

    def random(self):
        values = self.rng_values["random"]
        value = values[self.counter["random"] % len(values)]
        self.counter["random"] += 1
        return value

