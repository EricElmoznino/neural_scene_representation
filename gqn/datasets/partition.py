import random


def partition(images, viewpoints, min_viewpoints=None):
    """
    Partition batch into context and query sets.
    :param images
    :param viewpoints
    :param random_num: whether to sample a random number of views or
    :return: context images, context viewpoint, query image, query viewpoint
    """
    # Maximum number of context points to use
    b, m = viewpoints.shape[:2]

    # Shuffle views (and sample a random number of them between [min_viewpoints+1, total viewpoints] during training)
    if min_viewpoints is None:
        n_context = m - 1
    else:
        n_context = random.randint(min_viewpoints + 1, m - 1)
    indices = random.sample([i for i in range(m)], n_context)

    # Partition into context and query sets
    context_idx, query_idx = indices[:-1], indices[-1]

    x, v = images[:, context_idx], viewpoints[:, context_idx]
    x_q, v_q = images[:, query_idx], viewpoints[:, query_idx]

    return x, v, x_q, v_q
