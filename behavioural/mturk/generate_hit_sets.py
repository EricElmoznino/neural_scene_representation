from argparse import ArgumentParser
import random
import json

random.seed(27)


def create_pairwise_sampled_dict(n_scenes, imgs_per_scene):
    intra_pairs_sampled = {}
    intra_pairs_sampled_inv = {0: []}
    for scene in range(n_scenes):
        for img1 in range(imgs_per_scene - 1):
            for img2 in range(img1 + 1, imgs_per_scene):
                intra_pairs_sampled[(scene, scene, img1, img2)] = 0
                intra_pairs_sampled_inv[0].append((scene, scene, img1, img2))

    inter_pairs_sampled = {}
    inter_pairs_sampled_inv = {0: []}
    for scene1 in range(n_scenes - 1):
        for scene2 in range(scene1 + 1, n_scenes):
            for img1 in range(imgs_per_scene):
                for img2 in range(imgs_per_scene):
                    inter_pairs_sampled[(scene1, scene2, img1, img2)] = 0
                    inter_pairs_sampled_inv[0].append((scene1, scene2, img1, img2))

    return intra_pairs_sampled, inter_pairs_sampled, intra_pairs_sampled_inv, inter_pairs_sampled_inv


def get_n_least_sampled(pairs_sampled_inv, n):
    least_sampled = []
    sample_nums = sorted(pairs_sampled_inv.keys())
    for n_sampled in sample_nums:
        if len(least_sampled) >= n:
            break
        pairs = pairs_sampled_inv[n_sampled]
        random.shuffle(pairs)
        least_sampled += pairs
    least_sampled = least_sampled[:n]
    return least_sampled


def increment_pairs(pairs_sampled, pairs_sampled_inv, pairs):
    for pair in pairs:
        n_sampled = pairs_sampled[pair]
        pairs_sampled_inv[n_sampled].remove(pair)
        if len(pairs_sampled_inv[n_sampled]) == 0:
            del pairs_sampled_inv[n_sampled]

        n_sampled += 1
        pairs_sampled[pair] = n_sampled
        if n_sampled not in pairs_sampled_inv:
            pairs_sampled_inv[n_sampled] = []
        pairs_sampled_inv[n_sampled].append(pair)


def generate_hit_sets(n_scenes, imgs_per_scene, n_trials, min_hits_per_pair, intra_inter_ratio):
    intra_pairs_sampled, inter_pairs_sampled, intra_pairs_sampled_inv, inter_pairs_sampled_inv = \
        create_pairwise_sampled_dict(n_scenes, imgs_per_scene)

    hits = []
    while min(list(intra_pairs_sampled_inv.keys()) + list(inter_pairs_sampled_inv.keys())) < min_hits_per_pair:
        n_intra_trials = int(n_trials * intra_inter_ratio)
        n_inter_trials = n_trials - n_intra_trials

        intra_pairs = get_n_least_sampled(intra_pairs_sampled_inv, n_intra_trials)
        inter_pairs = get_n_least_sampled(inter_pairs_sampled_inv, n_inter_trials)
        pairs = intra_pairs + inter_pairs
        random.shuffle(pairs)

        hit = [{'scene1': p[0], 'scene2': p[1], 'img1': p[2], 'img2': p[3]} for p in pairs]
        hits.append(hit)

        increment_pairs(intra_pairs_sampled, intra_pairs_sampled_inv, intra_pairs)
        increment_pairs(inter_pairs_sampled, inter_pairs_sampled_inv, inter_pairs)

    return hits


if __name__ == '__main__':
    parser = ArgumentParser(description='Generate sets of stimuli pairs for turk hits')
    parser.add_argument('--n_scenes', required=True, type=int, help='number of scenes')
    parser.add_argument('--imgs_per_scene', required=True, type=int, help='number of images per scene')
    parser.add_argument('--n_trials', default=20, type=int, help='number of trials per hit')
    parser.add_argument('--min_hits_per_pair', default=10, type=int,
                        help='minimum number of hits than will include a given pair')
    parser.add_argument('--intra_inter_ratio', default=0.5, type=float,
                        help='ratio of intra-to-inter scene images presented for each hit (range of 0.0-1.0)')
    args = parser.parse_args()

    hits = generate_hit_sets(args.n_scenes, args.imgs_per_scene, args.n_trials,
                             args.min_hits_per_pair, args.intra_inter_ratio)
    with open('hit_sets.json', 'w') as f:
        f.write(json.dumps({'hit_sets': hits}, indent=2))
