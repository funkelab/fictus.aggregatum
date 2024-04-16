from fictus.aggregatum.utils import define_seeds
import numpy as np
import pytest


@pytest.mark.parametrize("base_seed", [42, 69, 2306, 5])
def test_random_seeds(base_seed):
    """Checks that all seeds within a dataset are unique."""
    # Set the numpy random seed
    num_cells = 25000 * 4
    all_seeds = define_seeds(base_seed, num_cells)
    assert set(np.unique(all_seeds)) == set(all_seeds)


@pytest.mark.parametrize(
    "seed1, seed2",
    [(0, 1), (42, 69), (5, 2306)]
)
def test_separate_datasets(seed1, seed2): 
    """Checks that two datasets have no overlapping seeds."""
    seeds_1 = define_seeds(seed1, 100000)
    seeds_2 = define_seeds(seed2, 100000)
    assert not np.any(seeds_1 == seeds_2)