from argparse import ArgumentParser
import os
import json
import pickle
from data_classes.Viewpoint import Viewpoint
from data_classes.Point import Point


def read_logs(logs_path):
    with open(logs_path, 'r') as f:
        logs = f.readlines()
    logs = [json.loads(l) for l in logs]
    return logs


def read_scenes(scenes_path):
    scenes = os.listdir(scenes_path)
    scenes = [s for s in scenes if '.pkl' in s]
    scenes = {s.replace('.pkl', ''): os.path.join(scenes_path, s) for s in scenes}
    for scene_name, path in scenes.items():
        with open(path, 'rb') as f:
            scene = pickle.load(f)
        assert len(scene.viewpoints) == 0
        scenes[scene_name] = scene
    return scenes


def save_scenes(scenes_path, scenes):
    for scene_name, scene in scenes.items():
        path = os.path.join(scenes_path, scene_name + '.pkl')
        with open(path, 'wb') as f:
            pickle.dump(scene, f)


def add_views_to_scenes(scenes, logs):
    for l in logs:
        scene_name = l['room']
        scene = scenes[scene_name]
        v = l['viewpoint']
        loc = v['location']
        viewpoint = Viewpoint(Point(loc['x'], loc['y'], loc['z']), v['rotation'], v['horizon'])
        scene.viewpoints.append(viewpoint)


if __name__ == '__main__':
    parser = ArgumentParser(description='Add viewpoints to scenes according to subject trajectory')
    parser.add_argument('--logs_path', required=True, type=str, help='path to the subject trajectory log file')
    args = parser.parse_args()

    scenes_path = args.logs_path.replace('.txt', '')
    assert os.path.exists(args.logs_path)
    assert os.path.exists(scenes_path)

    logs = read_logs(args.logs_path)
    scenes = read_scenes(scenes_path)
    add_views_to_scenes(scenes, logs)
    save_scenes(scenes_path, scenes)
