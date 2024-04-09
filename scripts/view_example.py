"""Small script to view a simple set of example cells."""
from fictus.aggregatum import make_cell
import napari


if __name__ == "__main__":
    viewer = napari.Viewer()
    n_cells = 10
    wavelength_membrane = 532
    wavelength_punctae = 646
    for i in range(0, n_cells):
        cell = make_cell(i, wavelength_membrane, wavelength_punctae)
        viewer.add_image(cell)
    input()