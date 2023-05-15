import torch
import numpy as np

def hash_tensor_entries(tensor):
    primes = [662773, 277427, 232861, 433633, 170843, 538481, 656783, 834949]
    for i in range(len(primes) - 2):
        tensor = (tensor + primes[i]) * primes[i + 1] % primes[i + 2]
    return tensor