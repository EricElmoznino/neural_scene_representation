from argparse import ArgumentParser
import os
import json
import numpy as np
import torch
from gqn.datasets import XYRHDataset
from gqn.models import GenerativeQueryNetwork
from vae.models import VAE
from fmri.representation_sequence import *


def read_logs(logs_path):
    with open(logs_path, 'r') as f:
        logs = f.readlines()
    logs = [json.loads(l) for l in logs]
    return logs


def load_model(model_path, vae=False):
    if not vae:
        model = GenerativeQueryNetwork(c_dim=3, v_dim=6, r_dim=256, h_dim=128, z_dim=3, l=8)
    else:
        model = VAE(c_dim=3, r_dim=256, h_dim=128, z_dim=3, l=8)
    model.eval()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    rep = model.representation
    return rep


def load_data(data_dir, room_seq):
    scale_factor = np.load(os.path.join(data_dir, 'scale_factor.npy')).item()
    data = XYRHDataset(os.path.join(data_dir, 'renderings'), resolution=64)
    scene_data = {}
    for scene_path, (x, v) in zip(data.scenes, data):
        if x is None:
            continue
        v[:, 0:2] /= scale_factor
        scene_name = scene_path.split('/')[-1]
        scene_data[scene_name] = x, v
    data = []
    for room in room_seq:
        images, viewpoints = scene_data[room]
        images, viewpoints = [x for x in images], [v for v in viewpoints]
        for x, v in zip(images, viewpoints):
            data.append((room, x, v))
    return data


if __name__ == '__main__':
    parser = ArgumentParser(description='Add viewpoints to scenes according to subject trajectory')
    parser.add_argument('--model_path', required=True, type=str,
                        help='path to saved model')
    parser.add_argument('--logs_path', required=True, type=str, help='path to the subject trajectory log file')
    parser.add_argument('--vae', action='store_true', help='whether to run VAE instead of GQN')
    args = parser.parse_args()

    scenes_path = args.logs_path.replace('.txt', '')
    assert os.path.exists(args.logs_path)
    assert os.path.exists(scenes_path)

    model = load_model(args.model_path, vae=args.vae)
    logs = read_logs(args.logs_path)
    room_seq = list(set([l['room'] for l in logs]))
    scene_data = load_data(scenes_path, room_seq)

    representations = rep_using_rooms(scene_data, model)
    representations = np.array([r.numpy() for r in representations])
    np.save('fmri/results/gqn_representations_room.npy', representations)
