"""Script for creating a large dataset of images and labels"""

from fictus.aggregatum import make_cell
from fictus.aggregatum.visualization import apply_spectra
import numpy as np
from pathlib import Path
import tifffile
import zarr


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


def main(label, num_cells, seed, zarr_path, save_path, split="train"):
    # Get 3D data container
    zarrfile = zarr.open(zarr_path, mode="a")
    label_group = zarrfile.require_group(f"{split}/{str(label)}")
    # Create datasets
    raw = label_group.require_dataset(
        "raw", shape=(num_cells, 3, 128, 128, 128), dtype=np.uint8
    )
    labels = label_group.require_dataset(
        "labels", shape=(num_cells, 128, 128, 128), dtype=np.uint64
    )

    # Get 2D data directory
    label_directory = Path(save_path) / str(label)
    label_directory.mkdir(exist_ok=True)

    for i in range(num_cells):
        membrane, punctae, nucleus, segmentation = make_cell(
            radius=25.0,
            seed=seed,
            elongation_ratio=elongations[label],
            num_cluster_mean=num_cluster_means[label],
        )

        cell = np.stack(
            [
                punctae,
                membrane,
                nucleus,
            ],
            axis=0,
        )
        raw[i] = cell
        labels[i] = segmentation

        # Get 2D versions by taking the middle z-slice
        cell_2d = cell[:, 64]
        # Save as a tiff with axes metadata
        tiff_path = label_directory / f"{i}.tiff"
        tifffile.imwrite(tiff_path, cell_2d, imagej=True, metadata={"axes": "CYX"})


if __name__ == "__main__":
    main(
        label=0,
        num_cells=10,
        seed=0,
        zarr_path="/nrs/funke/adjavond/data/aggregatum_v1.0.zarr",
        save_path="aggregatum",
        split="train",
    )
