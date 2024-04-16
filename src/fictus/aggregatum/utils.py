import numpy as np


def define_seeds(base_seed, num_cells):
    np.random.seed(base_seed)
    seeds = np.random.randint(0, 2**32, size=num_cells)
    return seeds

