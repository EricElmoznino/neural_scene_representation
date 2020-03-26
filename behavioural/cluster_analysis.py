from argparse import ArgumentParser
import numpy as np
np.random.seed(27)
from sklearn import manifold
from behavioural.plotting import plot_scatter


def get_clustering_index(rdm, n_scenes):
    # Inter-scene distance
    inter_distances = []
    images_per_scene = rdm.shape[0] // n_scenes
    for scene_a in range(n_scenes - 1):
        for scene_b in range(scene_a + 1, n_scenes):
            i = scene_a * images_per_scene
            j = scene_b * images_per_scene
            inter_distances.append(rdm[i:i+images_per_scene, j:j+images_per_scene].mean())
    mean_inter_distance = np.mean(inter_distances)

    # Intra-scene distance
    intra_distances = []
    for scene in range(n_scenes):
        scene = scene * images_per_scene
        for i in range(scene, scene + images_per_scene - 1):
            for j in range(i + 1, scene + images_per_scene):
                intra_distances.append(rdm[i, j])
    mean_intra_distance = np.mean(intra_distances)

    return mean_inter_distance/mean_intra_distance


def mds_coordinates(rdm, n_scenes):
    rdm = rdm / rdm.max()
    mds = manifold.MDS(n_components=2, dissimilarity="precomputed")
    coords = mds.fit(rdm).embedding_
    scene_coords = np.split(coords, n_scenes)
    return scene_coords


if __name__ == '__main__':
    parser = ArgumentParser(description='Assess the clustering of behavioural data based on rdm')
    parser.add_argument('--rdm', required=True, type=str, help='filename of the rdm to analyze')
    parser.add_argument('--n_scenes', required=True, type=int, help='number of scenes in rdm')
    args = parser.parse_args()

    rdm = np.load('behavioural/results/' + args.rdm)
    clustering_index = get_clustering_index(rdm, args.n_scenes)
    mds_coords = mds_coordinates(rdm, args.n_scenes)
    plot_scatter(mds_coords, 'Clustering Index: {:.3f}'.format(clustering_index))
