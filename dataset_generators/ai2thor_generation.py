from argparse import ArgumentParser
import os
import numpy as np
from PIL import Image
import ai2thor.controller

np.random.seed(27)


def setup_scene(controller, scene_name):
    controller.reset(scene_name)
    event = controller.step(dict(action='Initialize', fieldOfView=args.fov))

    # Remove all small objects in the scene (can filter in other ways)
    for obj in event.metadata['objects']:
        if obj['pickupable']:
            _ = controller.step(dict(action='RemoveFromScene', objectId=obj['objectId']))

    # Get navigable points
    event = controller.step(dict(action='GetReachablePositions'))
    navigable_points = event.metadata['actionReturn']

    # Get (x, z) centre of scene
    xs, zs = [p['x'] for p in navigable_points], [p['z'] for p in navigable_points]
    centre = np.array([(max(xs) - min(xs)) / 2 + min(xs), (max(zs) - min(zs)) / 2 + min(zs)])

    return navigable_points, centre


def random_viewpoints(n_samples, navigable_points, centre, rand_rot=15, rand_hor=20):
    loc = np.random.choice(np.array(navigable_points), n_samples)
    loc = np.array([[p['x'], p['y'], p['z']] for p in loc])

    xz = loc[:, [0, 2]]
    xz_to_centre = centre - xz
    rot = -180 / np.pi * np.arctan2(xz_to_centre[:, 1], xz_to_centre[:, 0]) + 90
    rot += np.random.uniform(-rand_rot, rand_rot, n_samples)

    horizon = np.random.uniform(0, rand_hor, n_samples)

    return loc, rot, horizon


def render_viewpoint(controller, location, rotation, horizon):
    event = controller.step(dict(action='TeleportFull',
                                 x=location[0], y=location[1], z=location[2],
                                 rotation=rotation, horizon=horizon))
    image = event.frame
    image = Image.fromarray(image)
    return image


if __name__ == '__main__':
    parser = ArgumentParser(description='Generative Query Network training')
    parser.add_argument('--data_dir', required=True, type=str,
                        help='directory in which to save the data')
    parser.add_argument('--n_viewpoints', type=int, default=200, help='number of viewpoints to sample per scene')
    parser.add_argument('--fov', type=int, default=75, help='camera field of view')
    args = parser.parse_args()

    os.mkdir(args.data_dir)
    os.mkdir(os.path.join(args.data_dir, 'train'))
    os.mkdir(os.path.join(args.data_dir, 'val'))

    controller = ai2thor.controller.Controller()
    controller.start()
    scene_names = controller.scene_names()
    scene_names = [s.replace('_physics', '') for s in scene_names]
    validation_scenes = ['FloorPlan1', 'FloorPlan2', 'FloorPlan201', 'FloorPlan202',
                         'FloorPlan301', 'FloorPlan302', 'FloorPlan401', 'FloorPlan402']

    for scene_name in scene_names:
        if scene_name in validation_scenes:
            save_dir = os.path.join(args.data_dir, 'val', scene_name)
        else:
            save_dir = os.path.join(args.data_dir, 'train', scene_name)
        os.mkdir(save_dir)

        navigable_points, centre = setup_scene(controller, scene_name)
        locations, rotations, horizons = random_viewpoints(args.n_viewpoints, navigable_points, centre)
        for i, (l, r, h) in enumerate(zip(locations.tolist(), rotations.tolist(), horizons.tolist())):
            image = render_viewpoint(controller, l, r, h)
            image.save(os.path.join(save_dir, '{:05}.png'.format(i)))
        viewpoints = np.concatenate([locations[:, [0, 2]], rotations[:, np.newaxis], horizons[:, np.newaxis]], axis=1)
        np.save(os.path.join(save_dir, 'viewpoints.npy'), viewpoints)

    controller.stop()
