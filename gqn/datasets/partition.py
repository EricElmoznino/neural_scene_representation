import random


def partition(images, viewpoints, min_viewpoints=None):
    """
    Partition batch into context and query sets.
    :param images
    :param viewpoints
    :param min_viewpoints: minimum number of context viewpoints to sample
    :return: context images, context viewpoint, query image, query viewpoint
    """
    # Maximum number of context points to use
    b, m = viewpoints.shape[:2]

    # Shuffle views (and sample a random number of them between [min_viewpoints+1, total viewpoints] during training)
    if min_viewpoints is None:
        num_samples = m
    else:
        num_samples = random.randint(min_viewpoints + 1, m)
    indices = random.sample([i for i in range(m)], num_samples)

    # Partition into context and query sets
    context_idx, query_idx = indices[:-1], indices[-1]

    x, v = images[:, context_idx], viewpoints[:, context_idx]
    x_q, v_q = images[:, query_idx], viewpoints[:, query_idx]

    return x, v, x_q, v_q
