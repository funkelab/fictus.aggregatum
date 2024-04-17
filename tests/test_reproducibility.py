from fictus.aggregatum import make_cell
import numpy as np
import pytest


@pytest.mark.parametrize("seed", [0, 1])
def test_reproducibility(seed):
    cell1 = make_cell(seed)
    cell2 = make_cell(seed)

    cell1 = np.stack(cell1)
    cell2 = np.stack(cell2)
    assert np.all(cell1 == cell2)
