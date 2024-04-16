"""Small script to view a simple set of example cells."""
from fictus.aggregatum import make_cell
from fictus.aggregatum.visualization import apply_spectra
import napari
import time


if __name__ == "__main__":
    viewer = napari.Viewer()
    n_cells = 1
    wavelength_membrane = 532
    wavelength_punctae = 646
    t0 = time.perf_counter()
    for i in range(0, n_cells):
        membrane, punctae, nucleus, segmentation = make_cell(i)
        cell = apply_spectra(
            membrane,
            punctae,
            wavelength_membrane=wavelength_membrane,
            wavelength_punctae=wavelength_punctae,
        )
        viewer.add_image(cell)
        viewer.add_image(nucleus)
        viewer.add_labels(segmentation) 
    time_elapsed = time.perf_counter() - t0
    print(f"Time to load {n_cells} cells: {time_elapsed:.2f} s,")
    print(f" or {time_elapsed/n_cells:.2f} s per cell.")
    input("Press Enter to close...")
