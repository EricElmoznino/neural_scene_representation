from argparse import ArgumentParser
import pandas as pd
import numpy as np
from behavioural.plotting import plot_rdm


def load_data(hit_results, keep_catch=False):
    data_raw = pd.read_csv('behavioural/mturk/hit_results/' + hit_results)

    def split_trials(row):
        def split_inner(cell):
            if isinstance(cell, str) and ',' in cell:
                cell = cell.split(',')
                if cell[0].isnumeric():
                    cell = [float(a) for a in cell]
            return cell
        return row.apply(split_inner)
    data = data_raw.apply(split_trials)

    def expand_trials(df, trials_per_subj):
        return pd.DataFrame({field: np.repeat(df[field].values, trials_per_subj)
                             for field in df.columns if not isinstance(df[field].values[0], list)}
                            ).assign(**{field: np.concatenate(df[field].values)
                                        for field in df.columns if isinstance(df[field].values[0], list)})
    trials_per_subj = len(data['response'].values[0])
    data = expand_trials(data, trials_per_subj)

    if not keep_catch:
        data = data[data['isCatch'] == 0]

    n_scenes = int(data['scene1'].max() + 1)

    return data, n_scenes


def compute_accuracy(data):
    same_scene = data['scene1'] == data['scene2']
    intra_correct = same_scene & (data['response'] == 'same')
    inter_correct = ~same_scene & (data['response'] == 'different')
    intra_accuracy = intra_correct.sum() / same_scene.sum()
    inter_accuracy = inter_correct.sum() / (~same_scene).sum()
    return intra_accuracy, inter_accuracy


def compute_rdm(data):
    data = data.copy()
    data['dissimilarity'] = (data['response'] == 'different').astype('float32')
    pair_means = data[['scene1', 'scene2', 'img1', 'img2', 'dissimilarity']]\
        .groupby(['scene1', 'scene2', 'img1', 'img2']).mean().reset_index()

    imgs_per_scene = int(data['img1'].max() + 1)
    n_scenes = int(data['scene1'].max() + 1)
    n_images = int(imgs_per_scene * n_scenes)
    pairwise_dissimilarities = np.zeros((n_images, n_images))
    for _, pair in pair_means.iterrows():
        first_idx = int(pair['scene1'] * imgs_per_scene + pair['img1'])
        second_idx = int(pair['scene2'] * imgs_per_scene + pair['img2'])
        pairwise_dissimilarities[first_idx, second_idx] = pair['dissimilarity']
        pairwise_dissimilarities[second_idx, first_idx] = pair['dissimilarity']

    return pairwise_dissimilarities


if __name__ == '__main__':
    parser = ArgumentParser(description='Human behavioural RDM')
    parser.add_argument('--hit_results', required=True, type=str, help='filename of Mechanical Turk results')
    args = parser.parse_args()

    data, n_scenes = load_data(args.hit_results)

    intra_scene_accuracy, inter_scene_accuracy = compute_accuracy(data)
    print('Intra-scene accuracy: {:.3f}'.format(intra_scene_accuracy))
    print('Inter-scene accuracy: {:.3f}'.format(inter_scene_accuracy))

    rdm = compute_rdm(data)
    np.save('behavioural/results/human_rdm.npy', rdm)
    plot_rdm(rdm, n_scenes)
