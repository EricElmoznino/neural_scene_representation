import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def plot_rdm(rdm, n_scenes, normalize=True):
    plt.close()
    f, ax = plt.subplots()

    if normalize:
        rdm /= rdm.max()
    mask = 1 - np.tril(np.ones_like(rdm, dtype=np.bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(rdm, mask=mask, cmap=cmap, square=True,
                linewidths=0.5, cbar_kws={'shrink': 0.5})

    scene_labels = ['Scene ' + str(i) for i in range(n_scenes)]
    n_images = len(rdm)
    imgs_per_scene = n_images / n_scenes

    ax.set_xticks(np.arange(0, n_images + 1, imgs_per_scene))
    ax.set_xticklabels('')
    ax.set_xticks(np.arange(0, n_images, imgs_per_scene) + imgs_per_scene / 2, minor=True)
    ax.set_xticklabels(scene_labels, minor=True)
    ax.set_yticks(np.arange(0, n_images + 1, imgs_per_scene))
    ax.set_yticklabels('')
    ax.set_yticks(np.arange(0, n_images, imgs_per_scene) + imgs_per_scene / 2, minor=True)
    ax.set_yticklabels(scene_labels, minor=True)
    ax.tick_params(which='minor', color='white')

    plt.show()


def plot_scatter(groups, title):
    plt.close()
    f, ax = plt.subplots()

    data = []
    for scene, points in enumerate(groups):
        for x, y in points:
            data.append({'x': x, 'y': y, 'Scene': 'Scene {}'.format(scene)})
    data = pd.DataFrame(data)

    sns.scatterplot(x='x', y='y', hue='Scene', data=data, hue_order=['Scene {}'.format(i) for i in range(len(groups))])
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.title(title)

    plt.show()


def plot_bar(data):
    plt.close()
    f, ax = plt.subplots()

    data = [{'Comparison': c, 'Pearson Correlation': r} for c, r in data.items()]
    data = pd.DataFrame(data)

    sns.barplot(x='Comparison', y='Pearson Correlation', data=data)
    ax.set_xlabel('')
    plt.title('RSA')

    plt.show()
