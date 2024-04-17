import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.spatial.transform import Rotation


def sample_membrane(
    shape,
    radius,
    elongation_ratio,
    membrane_width,
    membrane_fuzziness,
    membrane_jitter_amplitude,
    membrane_jitter_smoothness,
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)
    elongation = np.array([radius, radius, radius * elongation_ratio], dtype=np.float32)
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


def sample_nucleus(
    boundary_distances,
    nucleus_radius,
    membrane_width=3.0,
    nucleus_jitter_amplitude=500.0,
    nucleus_jitter_smoothness=10.0,
    nucleus_fuzziness=0.5,
    chromatin_intensity=1.0,
    chromatin_intensity_var=5.0,
    chromatin_smoothness=1.0,
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)
    shape = boundary_distances.shape

    # Sample a nucleus point in where distances are highest
    max_distance = np.max(boundary_distances)
    center = np.unravel_index(np.argmax(boundary_distances), boundary_distances.shape)
    nucleus_radius = min(nucleus_radius, max_distance - 3 * membrane_width)

    coordinates = np.stack(
        np.meshgrid(
            np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
        ),
        axis=-1,
    ).astype(np.float32)
    # set center to (0, 0, 0)
    coordinates -= center
    # compute signed distance to sphere surface
    distances = np.sqrt(np.sum(coordinates**2, axis=-1)) - nucleus_radius

    distance_jitter = np.random.normal(
        0, nucleus_jitter_amplitude, size=distances.shape
    )
    distances += gaussian_filter(distance_jitter, nucleus_jitter_smoothness)

    # Transfer distance to nucleus segmentation
    nucleus_segmentation = distances < 0

    # Make chromatin
    chromatin = np.random.normal(
        chromatin_intensity, chromatin_intensity_var, size=shape
    )
    chromatin = gaussian_filter(chromatin, chromatin_smoothness)

    # Create nucleus from a smooth the nucleus segmentation
    nucleus = chromatin * gaussian_filter(
        nucleus_segmentation.astype(np.float32), nucleus_fuzziness
    )
    return nucleus, nucleus_segmentation


def get_segmentation(distances, nucleus_segmentation, membrane_width):
    multi_class = np.zeros(distances.shape, dtype=np.uint64)
    multi_class[distances > 0] = 1
    multi_class[nucleus_segmentation] = 3
    multi_class[np.abs(distances) < membrane_width] = 2
    return multi_class


def sample_aggregates(
    boundary_distance,
    num_cluster_mean,
    num_cluster_var,
    boundary_distance_mean,
    boundary_distance_var,
    cluster_var,
    punctae_intensity,
    num_punctae,
    seed=None,
):
    if seed is not None:
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


def apply_optics(
    membrane,
    punctae,
    nucleus,
    hf_noise_sigma,
    lf_noise_sigma,
    lf_noise_smoothness,
    psf_sigmas=(5.0, 1.0, 1.0),
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)

    membrane = gaussian_filter(membrane, psf_sigmas)
    punctae = gaussian_filter(punctae, psf_sigmas)
    nucleus = gaussian_filter(nucleus, psf_sigmas)

    # Apply high-frequency noise
    membrane += np.random.normal(0, hf_noise_sigma, size=membrane.shape)
    punctae += np.random.normal(0, hf_noise_sigma, size=punctae.shape)
    nucleus += np.random.normal(0, hf_noise_sigma, size=nucleus.shape)

    # Apply low-frequency noise to the membrane
    membrane += gaussian_filter(
        np.random.normal(0, lf_noise_sigma, size=membrane.shape), lf_noise_smoothness
    )
    membrane = np.clip(membrane, a_min=0, a_max=None)
    punctae = np.clip(punctae, a_min=0, a_max=None)
    nucleus = np.clip(nucleus, a_min=0, a_max=None)
    return membrane, punctae, nucleus


def make_cell(
    seed,
    elongation_ratio=2.0,
    num_cluster_mean=50,
    # These should be kept constant
    shape=(128, 128, 128),
    radius=30.0,
    membrane_width=3.0,
    membrane_fuzziness=1.0,
    membrane_jitter_amplitude=500.0,
    membrane_jitter_smoothness=10.0,
    num_cluster_var=1.0,
    avg_punctae_per_cluster=10,
    boundary_distance_mean=20.0,
    boundary_distance_var=10.0,
    cluster_var=1.0,
    punctae_intensity=10.0,
    hf_noise_sigma=0.1,
    lf_noise_sigma=25.0,
    lf_noise_smoothness=15.0,
):
    num_punctae = avg_punctae_per_cluster * num_cluster_mean

    distances, membrane = sample_membrane(
        shape=shape,
        radius=radius,
        elongation_ratio=elongation_ratio,
        membrane_width=membrane_width,
        membrane_fuzziness=membrane_fuzziness,
        membrane_jitter_amplitude=membrane_jitter_amplitude,
        membrane_jitter_smoothness=membrane_jitter_smoothness,
        seed=seed,
    )

    nucleus, nucleus_segmentation = sample_nucleus(
        distances, nucleus_radius=np.round(2 * radius / 3)
    )

    _, punctae = sample_aggregates(
        distances,
        num_cluster_mean=num_cluster_mean,
        num_cluster_var=num_cluster_var,
        boundary_distance_mean=boundary_distance_mean,
        boundary_distance_var=boundary_distance_var,
        cluster_var=cluster_var,
        punctae_intensity=punctae_intensity,
        num_punctae=num_punctae,
        seed=seed,
    )

    membrane, punctae, nucleus = apply_optics(
        membrane,
        punctae,
        nucleus,
        hf_noise_sigma=hf_noise_sigma,
        lf_noise_sigma=lf_noise_sigma,
        lf_noise_smoothness=lf_noise_smoothness,
        seed=seed,
    )

    segmentation = get_segmentation(
        distances, nucleus_segmentation, membrane_width=membrane_width
    )
    return membrane, punctae, nucleus, segmentation
