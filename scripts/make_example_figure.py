"""Small script to view a simple set of example cells."""

from fictus.aggregatum import make_cell
from fictus.aggregatum.visualization import apply_spectra
from matplotlib import pyplot as plt
import numpy as np
import napari
import time

elongations = {
    0: 1.5,
    1: 1.5,
    2: 2.5,
    3: 2.5,
}
num_cluster_means = {
    0: 50,
    1: 200,
    2: 50,
    3: 200,
}


def make_2d(
    membrane,
    punctae,
    nucleus,
    slice,
    wavelength_membrane=532,
    wavelength_punctae=646,
    wavelength_nucleus=450,
):
    membrane = apply_spectra(
        membrane[slice] / np.max(membrane[slice]),
        wavelength=wavelength_membrane,
    )
    punctae = apply_spectra(
        punctae[slice] / np.max(punctae[slice]), wavelength=wavelength_punctae
    )
    nucleus = apply_spectra(
        nucleus[slice] / np.max(nucleus[slice]), wavelength=wavelength_nucleus
    )
    cell = membrane + punctae + nucleus
    # cell = cell / np.max(cell, axis=-1, keepdims=True)
    return (255 * cell).astype(np.uint8)


if __name__ == "__main__":
    label = 3
    n_cells = 5

    t0 = time.perf_counter()

    membrane_t = []
    punctae_t = []
    nucleus_t = []

    for i in range(0, n_cells):
        membrane, punctae, nucleus, segmentation = make_cell(
            radius=25.0,
            seed=i,
            elongation_ratio=elongations[label],
            num_cluster_mean=num_cluster_means[label],
        )
        # viewer.add_labels(segmentation, visible=visible)
        membrane_t.append(membrane)
        punctae_t.append(punctae)
        nucleus_t.append(nucleus)

        cell = make_2d(membrane, punctae, nucleus, slice=64)
        plt.imsave(f"docs/class_examples/class_{label}_cell_{i}.png", cell)

    membrane_t = np.stack(membrane_t)
    punctae_t = np.stack(punctae_t)
    nucleus_t = np.stack(nucleus_t)

    time_elapsed = time.perf_counter() - t0
    print(f"Time to load {n_cells} cells: {time_elapsed:.2f} s,")
    print(f" or {time_elapsed/n_cells:.2f} s per cell.")
    # viewer = napari.Viewer()
    # viewer.add_image(
    #     membrane_t,
    #     name=f"membrane",
    #     colormap="green",
    #     blending="additive",
    # )
    # viewer.add_image(punctae_t, name=f"punctae", colormap="red", blending="additive")
    # viewer.add_image(nucleus_t, name=f"nucleus", colormap="blue", blending="additive")
    # input("Press Enter to close...")
