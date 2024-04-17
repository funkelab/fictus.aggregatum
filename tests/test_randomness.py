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
    "seed1, seed2, num_cells",
    [
        pytest.param(0, 1, 100000, marks=pytest.mark.xfail(reason="Too many samples")),
        pytest.param(
            42, 69, 100000, marks=pytest.mark.xfail(reason="Too many samples")
        ),
        pytest.param(
            5, 2306, 100000, marks=pytest.mark.xfail(reason="Too many samples")
        ),
        (0, 1, 25000),
        (42, 69, 25000),
        (5, 2306, 25000),
    ],
)
def test_separate_datasets(seed1, seed2, num_cells):
    """Checks that two datasets have no overlapping seeds."""
    seeds_1 = define_seeds(seed1, num_cells)
    seeds_2 = define_seeds(seed2, num_cells)
    assert len(set(seeds_1).intersection(set(seeds_2))) == 0
    assert set(seeds_1).isdisjoint(set(seeds_2))
