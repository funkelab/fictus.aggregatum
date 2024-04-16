from fictus.aggregatum import make_cell
import numpy as np
import pytest


@pytest.mark.parametrize("seed", [0, 1])
def test_reproducibility(seed):
    cell1, _ = make_cell(seed, 560, 570)
    cell2, _ = make_cell(seed, 560, 570)
    assert np.all(cell1 == cell2)
