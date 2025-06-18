import numpy as np
from scipy.cluster.hierarchy import dendrogram, to_tree

def get_uniform_data(n_samples, n_dims=1, seed=0):
    np.random.seed(seed)
    data = np.random.rand(n_samples, n_dims)
    return data


def approport(n, shares):
    """
    Approximate proportions of n to shares. Using largest remainder method.
    """
    if isinstance(shares, int):
        shares = np.ones(shares) / shares
    n = int(n)
    fair_shares = n * shares
    remainders = fair_shares - np.floor(fair_shares)
    counts = np.floor(fair_shares)
    missing = int(n - counts.sum())
    order = np.argpartition(remainders, missing)
    counts[order[:missing]] += 1
    assert counts.sum() == n
    return counts

def counts_to_assignments(n, counts):
    # transform cluster counts to assignment
    first_points = np.concatenate([[0], np.cumsum(counts)]).astype(int)
    assignments = np.zeros(n)
    for i in range(len(first_points) - 1):
        assignments[first_points[i]:first_points[i + 1]] = i

    _, effective_counts = np.unique(assignments, return_counts=True)
    assert np.all(effective_counts == counts), "Effective counts do not match intended counts"
    return assignments

def get_gaussian_mixture(n_samples, n_dims=None, n_clusters=None, n_latent_dims=None, std=0.1, equispaced=False, prior=None, exact_counts=False,
                         seed=0, return_centroids=False, cluster_centers=None, return_assignments=False):


    assert (n_clusters is not None and n_dims is not None) + (cluster_centers is not None) == 1, "Either n_clusters or cluster_centers must be provided."
    if cluster_centers is not None:
        n_clusters = cluster_centers.shape[0]

    if n_latent_dims is None:
        if cluster_centers is not None:
            n_latent_dims = cluster_centers.shape[1]
        else:
            n_latent_dims = n_dims

    if n_dims is None:
        n_dims = n_latent_dims


    np.random.seed(seed)

    if cluster_centers is None:
        # get cluster centers
        if equispaced:
            if n_latent_dims == 1:
                #cluster_centers = np.linspace(0, 1, n_clusters)[:, None]  # distance between points depends on the number of clusters
                cluster_centers = np.arange(n_clusters)[:, None] # cluster centers are always 1 unit apart

            elif n_latent_dims == 2:
                sqrt = np.ceil(np.sqrt(n_clusters))
                x, y = np.mgrid[0:1:sqrt * 1j, 0:1:sqrt * 1j]
                cluster_centers = np.stack([x.flatten(), y.flatten()]).T
                cluster_centers = cluster_centers[:n_clusters]

        else:
            cluster_centers = get_uniform_data(n_clusters, n_latent_dims, seed=seed)


    # actual dimensionality is smaller than ambient
    if n_latent_dims < n_dims:
        if len(cluster_centers.shape) ==1:
            cluster_centers = cluster_centers[:, None]
        cluster_centers = np.concatenate([cluster_centers, np.zeros((n_clusters, n_dims - n_latent_dims))], axis=1)

    # sample cluster assignments
    if prior is None:
        prior = np.ones(n_clusters) / n_clusters
    if exact_counts:
        # make counts so that no clusters off a lot from its fair share of points by using highest remainder method
        counts = approport(n_samples, prior)
        cluster_assignments = counts_to_assignments(n_samples, counts)

    else:
        # just sample cluster assignment according to prior
        cluster_assignments = np.random.choice(n_clusters, n_samples, p=prior)

    # sample the gaussian noise
    data = np.zeros((n_samples, n_dims))
    for c in range(n_clusters):
        data[cluster_assignments == c] = np.random.randn(np.sum(cluster_assignments == c), n_dims) * std + \
                                         cluster_centers[c]

    if return_centroids:
        return data, cluster_centers

    if return_assignments:
        return data, cluster_assignments

    return data


def add_gaussian_noise(data, std, noise_dims, seed=0):
    np.random.seed(seed)
    noise = np.random.normal(0, std, size=(data.shape[0], noise_dims))
    return np.hstack([data, noise])

###################
# synthetic hierarchical dataset
###################
def side_to_radius(side, n):
    """
    Convert the side length of a regular n-polygon to the radius of the circumscribed circle.
    """
    return side / (2 * np.sin(np.pi / n))

def radius_to_side(radius, n):
    """
    Convert the radius of a circumscribed circle of a regular n-polygon to the side length.
    """
    return 2 * radius * np.sin(np.pi / n)


def hierarchical_gaussian_mixture(n, d=2, sigma=0.0, std=0.1, levels=3, c=5, seed=0):
    """

    :param n:
    :param d: Ambient dimensionality (intrinsic is 2)
    :param sigma: Standard deviation in the ambient dimensions
    :param std: Standard deviation of the Gaussian noise added to the points in the finest level.
    :param levels:
    :param c: number of clusters per level
    :param seed:
    :return:
    """

    angles = np.linspace(0, 2 * np.pi, c, endpoint=False)

    # the ratio of the radii of a circumscibed circle of each level to the side length of the polygon of the previous level is rel_std
    # (and finally this serves as std for a normal distribution)
    # so r_{i+1} = rel_std * s_i
    radii = [1]
    sides = [radius_to_side(radii[0], c)]
    for i in range(1, levels):
        radii.append(std * sides[i - 1])
        sides.append(radius_to_side(radii[i], c))

    np.random.seed(seed)

    # get centroids
    # first relative to zero
    centered_centroid_l = []
    for i in range(levels):
        r = radii[i]
        centroids = np.stack([np.cos(angles) * r, np.sin(angles) * r], axis=1)
        centered_centroid_l.append(centroids)

    # add to centroid of the previous level
    centroid_l = [centered_centroid_l[0]]
    centroid_assignments = [np.zeros(c, dtype=int)]

    for i in range(1, levels):
        # add random rotations
        rotation_angle = np.random.uniform(0, 2 * np.pi, size=len(centroid_l[i - 1]))

        rot_matrices = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                 [np.sin(rotation_angle), np.cos(rotation_angle)]]).transpose(2, 0,
                                                                                              1)  # shape n_cluster_prev, 2,2
        repeated_centroids = np.tile(centered_centroid_l[i],
                                     (len(centroid_l[i-1]), 1, 1))  # shape (n_clusters_prev, n_clusters_per_level, 2)

        # apply the rotation
        rotated_centroids = np.einsum('ijk,ikl->ijl', repeated_centroids,
                                      rot_matrices)  # shape (n_clusters_prev, n_clusters_per_level, 2)

        # add the rotated centered centroids to the centroid of the previous level
        shifted_rotated_centroids = centroid_l[i - 1][:, None,
                                    :] + rotated_centroids  # shape (n_clusters_prev, n_clusters_per_level, 2)

        centroid_l.append(np.concatenate(shifted_rotated_centroids))
        n_centroids_prev = len(centroid_l[i - 1])
        centroid_assignments.append(np.repeat(np.arange(n_centroids_prev, dtype=int), c))

        assert len(centroid_l[i]) == len(centroid_assignments[i]), \
            f"Centroid assignments ({len(centroid_assignments[i])}) must match the number of centroids ({len(centroid_l[i])})."

    # the points to sample are those in the very last level.
    data, finest_assignments = get_gaussian_mixture(n,
                                                    cluster_centers=centroid_l[-1],
                                                    n_dims=d,
                                                    std=np.concatenate([np.repeat(sides[-1] * std, 2), np.repeat(sigma, d - 2)]), # different std in ambient dim and data dim
                                                    # / np.sqrt(2* np.log(n_samples / len(centroid_l[-1]))), # E(max(x1, ..., xk)) ~ sqrt(2log(k)) for x_i ~ N(0, 1)
                                                    seed=seed,
                                                    return_assignments=True)

    # get hierarchical gt
    gts = [finest_assignments]
    for i in range(levels - 1, 0, -1):
        # get the assignments for the previous level
        prev_assignments = centroid_assignments[i][gts[levels - 1 - i]]
        gts.append(prev_assignments)
    gts.append(np.zeros(n, dtype=int))  # the very last level is just one cluster
    gts = np.array(gts)

    return data, gts


