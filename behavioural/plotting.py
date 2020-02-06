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


def plot_scatter(groups, title):
    plt.close()
    plots = [plt.scatter([p[0] for p in points], [p[1] for p in points]) for points in groups]
    plt.title(title)
    plt.show()
