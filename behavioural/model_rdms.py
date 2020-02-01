from argparse import ArgumentParser
import os
import numpy as np
import torch
from gqn.datasets import XYRHDataset
from gqn.models import GenerativeQueryNetwork
from vae.models import VAE
from behavioural.plotting import plot_rdm


def load_model(model_path, vae=False):
    if not vae:
        model = GenerativeQueryNetwork(c_dim=3, v_dim=6, r_dim=256, h_dim=128, z_dim=3, l=8)
    else:
        model = VAE(c_dim=3, r_dim=256, h_dim=128, z_dim=3, l=8)
    model.eval()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    rep = model.representation
    return rep


def load_data(data_dir):
    data = XYRHDataset(os.path.join(data_dir, 'renderings'), resolution=64)
    images, viewpoints = [], []
    for x, v in data:
        images.append(x)
        viewpoints.append(v)
    images = torch.cat(images)
    viewpoints = torch.cat(viewpoints)
    scale_factor = np.load(os.path.join(data_dir, 'scale_factor.npy')).item()
    viewpoints[:, 0:2] /= scale_factor
    n_scenes = len(data)
    return images, viewpoints, n_scenes


def compute_rdm(images, viewpoints, model, metric='eucl', vae=False):
    print('Computing RDM')

    with torch.no_grad():
        if not vae:
            r = model(images, viewpoints)
        else:
            r = model(images)
    r = r.squeeze(3).squeeze(2).numpy()

    if metric == 'eucl':
        d_func = lambda x, y: np.linalg.norm(x - y)
    elif metric == '1-corr':
        d_func = lambda x, y: 1 - np.corrcoef(x, y)[0, 1]
    else:
        raise NotImplementedError('Unimplemented distance metric: {}'.format(metric))

    pairwise_distances = [[d_func(r[i], r[j]) for i in range(len(r))] for j in range(len(r))]
    pairwise_distances = np.array(pairwise_distances)

    return pairwise_distances


if __name__ == '__main__':
    parser = ArgumentParser(description='Model RDMs on behavioural data')
    parser.add_argument('--model_path', required=True, type=str,
                        help='path to saved model')
    parser.add_argument('--data_dir', required=True, type=str, help='directory of the data')
    parser.add_argument('--metric', default='1-corr', choices=['eucl', '1-corr'])
    parser.add_argument('--vae', action='store_true', help='whether to run VAE instead of GQN')
    args = parser.parse_args()

    model = load_model(args.model_path, vae=args.vae)
    images, viewpoints, n_scenes = load_data(args.data_dir)
    rdm = compute_rdm(images, viewpoints, model, metric=args.metric, vae=args.vae)
    save_name = 'gqn_rdm.npy' if not args.vae else 'vae_rdm.npy'
    np.save('behavioural/results/' + save_name, rdm)
    plot_rdm(rdm, n_scenes)
