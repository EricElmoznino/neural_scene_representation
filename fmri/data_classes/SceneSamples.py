from typing import List
from PIL import ImageDraw
from data_classes.Scene import Scene
from data_classes.Viewpoint import Viewpoint


class SceneSamples:

    def __init__(self, scenes: List[Scene], viewpoints: List[Viewpoint]):
        self.scenes = scenes
        self.viewpoints = viewpoints

    def visualize(self):
        image, scale = self.scenes[0].visualize(return_scale=True)
        point_radius, arc_radius = 2, 20
        draw = ImageDraw.Draw(image)
        for vp in self.viewpoints:
            centre = (int(vp.location.y * scale), int(vp.location.x * scale))
            draw.pieslice([(centre[0] - arc_radius, centre[1] - arc_radius),
                          (centre[0] + arc_radius, centre[1] + arc_radius)],
                          -vp.rotation + 90 - 30, -vp.rotation + 90 + 30,
                         fill=(0, 125, 0))
            draw.ellipse([(centre[0] - point_radius, centre[1] - point_radius),
                          (centre[0] + point_radius, centre[1] + point_radius)],
                         fill=(0, 200, 0))
        return image

    def serialize(self):
        return {
            'scenes': [s.serialize() for s in self.scenes],
            'viewpoints': [v.serialize() for v in self.viewpoints]
        }
