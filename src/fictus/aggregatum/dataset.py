import napari
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.spatial.transform import Rotation
from fictus.aggregatum.wav2rgb import wav2RGB


def sample_membrane(
    shape,
    elongation,
    membrane_width,
    membrane_fuzziness,
    membrane_jitter_amplitude,
    membrane_jitter_smoothness,
    seed=0,
):
    np.random.seed(seed)
    # coordinates: (d, h, w, 3)
    coordinates = np.stack(
        np.meshgrid(
            np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
        ),
        axis=-1,
    ).astype(np.float32)

    # set center to (0, 0, 0)
    center = np.array([s / 2 for s in shape], dtype=np.float32)
    coordinates -= center
    # Apply the rotation
    rotation = Rotation.random(random_state=seed)
    coordinates = rotation.apply(coordinates.reshape(-1, 3)).reshape(shape + (3,))
    # compute signed distance to ellipse surface
    ratios = np.array(elongation, dtype=np.float32)
    scaled_coordinates = coordinates**2 / ratios**2
    # note: those distances are anisotropic
    distances = np.sqrt(np.sum(scaled_coordinates, axis=-1)) - 1.0

    interior = distances < 0.0
    # bring distances back to isotropic
    distances = distance_transform_edt(1 - interior) - distance_transform_edt(interior)
    distances = gaussian_filter(distances, 1.0)

    distance_jitter = np.random.normal(
        0, membrane_jitter_amplitude, size=distances.shape
    )
    distance_jitter = gaussian_filter(distance_jitter, membrane_jitter_smoothness)

    distances += distance_jitter

    # transfer distance to membrane intensity
    membrane = 1.0 - 1.0 / (
        1.0 + np.exp(-(np.abs(distances) - membrane_width) / membrane_fuzziness)
    )

    return -distances, membrane


def sample_aggregates(
    boundary_distance,
    num_cluster_mean,
    num_cluster_var,
    boundary_distance_mean,
    boundary_distance_var,
    cluster_var,
    punctae_intensity,
    num_punctae,
    seed=0,
):
    np.random.seed(seed)
    num_clusters = int(np.round(np.random.normal(num_cluster_mean, num_cluster_var)))

    shape = boundary_distance.shape
    size = boundary_distance.size

    if num_clusters == 0:
        return np.zeros(boundary_distance.shape, dtype=np.float32)

    numerator = (boundary_distance - boundary_distance_mean) ** 2
    denominator = boundary_distance_var**2
    boundary_density = np.exp(-numerator / denominator)
    interior_density = boundary_density * (boundary_distance > 0)

    # make sure densities sum up to 1
    interior_density /= np.sum(interior_density)

    indices = np.random.choice(
        size, size=num_clusters, replace=False, p=interior_density.flatten()
    )

    cluster_locations = tuple(np.unravel_index(indices, shape))
    punctae_density = np.zeros(shape, dtype=np.float32)
    np.add.at(punctae_density, cluster_locations, 1.0)

    punctae_density = gaussian_filter(punctae_density, cluster_var)
    punctae_density *= boundary_distance > 0
    punctae_density /= np.sum(punctae_density)

    indices = np.random.choice(size, size=num_punctae, p=punctae_density.flatten())
    punctae_locations = tuple(np.unravel_index(indices, shape))

    punctae = np.zeros(shape, dtype=np.float32)
    np.add.at(punctae, punctae_locations, punctae_intensity)

    return boundary_density, punctae


def apply_spectra(membrane, punctae, wavelength_membrane=560, wavelength_punctae=570):
    membrane_rgb = wav2RGB(wavelength_membrane)  # returns r,g,b values between 0 and 1
    punctae_rgb = wav2RGB(wavelength_punctae)  # returns r,g,b values between 0 and 1

    membrane_image = np.stack([membrane * c for c in membrane_rgb], axis=-1)
    punctae_image = np.stack([punctae * c for c in punctae_rgb], axis=-1)

    return membrane_image + punctae_image


def apply_optics(
    membrane,
    punctae,
    hf_noise_sigma,
    lf_noise_sigma,
    lf_noise_smoothness,
    seed=0,
    psf_sigmas=(5.0, 1.0, 1.0),
):
    np.random.seed(seed)

    membrane = gaussian_filter(membrane, psf_sigmas)
    punctae = gaussian_filter(punctae, psf_sigmas)

    # Apply high-frequency noise
    membrane += np.random.normal(0, hf_noise_sigma, size=membrane.shape)
    punctae += np.random.normal(0, hf_noise_sigma, size=punctae.shape)

    # Apply low-frequency noise to the membrane
    membrane += gaussian_filter(
        np.random.normal(0, lf_noise_sigma, size=membrane.shape), lf_noise_smoothness
    )
    membrane = np.clip(membrane, a_min=0, a_max=None)
    punctae = np.clip(punctae, a_min=0, a_max=None)
    return membrane, punctae


def make_cell(seed, wavelength_membrane=500, wavelength_punctae=700):
    print("Creating membrane...")
    distances, membrane = sample_membrane(
        shape=(128, 128, 128),
        elongation=(30, 30, 60),  # --> number ratio of the axes
        membrane_width=3.0,
        membrane_fuzziness=1.0,
        membrane_jitter_amplitude=500.0,
        membrane_jitter_smoothness=10.0,
        seed=seed,
    )

    print("Creating punctae...")
    boundary_density, punctae = sample_aggregates(
        distances,
        num_cluster_mean=200,  # 1?
        num_cluster_var=1.0,
        boundary_distance_mean=20.0,  # 1
        boundary_distance_var=10.0,  # 2
        cluster_var=1.0,
        punctae_intensity=10.0,
        num_punctae=2000,
        seed=seed,
    )

    print("Applying optics...")
    membrane, punctae = apply_optics(
        membrane,
        punctae,
        hf_noise_sigma=0.1,
        lf_noise_sigma=25.0,
        lf_noise_smoothness=15.0,
        seed=seed,
    )

    print("Applying spectra...")
    cell = apply_spectra(
        membrane,
        punctae,
        wavelength_membrane=wavelength_membrane,
        wavelength_punctae=wavelength_punctae,
    )

    return cell
