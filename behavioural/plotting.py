from matplotlib import pyplot as plt


def plot_rdm(array, n_scenes, normalize=True):
    plt.close()
    if normalize:
        array /= array.max()
    plt.matshow(array, cmap='jet')
    plt.xticks(range(0, len(array), len(array) // n_scenes), range(n_scenes))
    plt.yticks(range(0, len(array), len(array) // n_scenes), range(n_scenes))
    plt.colorbar()
    plt.show()
