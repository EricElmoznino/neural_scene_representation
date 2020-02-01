import random
from copy import deepcopy
import numpy as np
from PIL import Image, ImageDraw
from data_classes.Point import Point
from data_classes.Object import Object
from data_classes.Surface import Surface
from data_classes.Orientation import Orientation


class Scene:

    def __init__(self, floor_plan: np.ndarray):
        floor_plan = deepcopy(floor_plan)
        floor_plan = Scene.squeeze_floor_plan(floor_plan)
        self.floor_plan = floor_plan
        self.object_floor_plan = floor_plan.copy()
        self.floor, self.ceiling, self.walls = Scene.make_surfaces(floor_plan)
        self.objects = []
        self.lights = []

    def navigable_points(self, include_objects=True):
        fp = self.object_floor_plan if include_objects else self.floor_plan
        points = []
        for x in range(1, fp.shape[0]):
            for y in range(1, fp.shape[1]):
                if fp[x - 1:x + 1, y - 1:y + 1].all():
                    points.append([x, y])
        return points

    def randomly_place_object(self, type, size):
        rotation = random.choice([Orientation.LEFT, Orientation.FRONT, Orientation.RIGHT, Orientation.BACK])
        object = Object(type, Point(0, 0, 0), rotation, size)
        candidate_locations = self.navigable_points()
        random.shuffle(candidate_locations)
        for x, y in candidate_locations:
            object.location = Point(x, y, 0)
            if self.add_object(object):
                return True
        return False

    def add_object(self, object: Object):
        if not self.fits(object):
            return False
        self.objects.append(object)
        sx, sy = object.size
        if object.rotation in [Orientation.FRONT, Orientation.BACK]:
            sx, sy = sy, sx
        for x in range(int(object.location.x) - sx // 2, int(object.location.x) + sx // 2):
            for y in range(int(object.location.y) - sy // 2, int(object.location.y) + sy // 2):
                self.object_floor_plan[x, y] = False

        return True

    def add_light(self, light):
        self.lights.append(light)

    def fits(self, object: Object):
        sx, sy = object.size
        if object.rotation in [Orientation.FRONT, Orientation.BACK]:
            sx, sy = sy, sx
        for x in range(int(object.location.x) - sx // 2, int(object.location.x) + sx // 2):
            for y in range(int(object.location.y) - sy // 2, int(object.location.y) + sy // 2):
                if not self.object_floor_plan[x, y]:
                    return False
        return True

    def visualize(self, return_scale=False):
        # Floor plan and objects
        floor_plan = self.floor_plan
        objects = floor_plan ^ self.object_floor_plan
        image = floor_plan.astype(np.uint8) * 100
        image = np.stack([image, image, image], axis=-1)
        image[:, :, 2][objects] = 255
        image = Image.fromarray(image)
        width, height = image.width, image.height
        if width > height:
            new_width, new_height = 500, int(500 * height/width)
        else:
            new_width, new_height = int(500 * width/height), 500
        image = image.resize((new_width, new_height))
        scale = new_width / width

        # Lights
        radius = 2
        draw = ImageDraw.Draw(image)
        for l in self.lights:
            centre = (int(l.location.y * scale), int(l.location.x * scale))
            draw.ellipse([(centre[0] - radius, centre[1] - radius), (centre[0] + radius, centre[1] + radius)],
                         fill=(2555, 255, 255))

        if return_scale:
            return image, scale
        return image

    def copy(self):
        return deepcopy(self)

    def serialize(self):
        return {
            'floor': self.floor.serialize(),
            'ceiling': self.ceiling.serialize(),
            'walls': [w.serialize() for w in self.walls],
            'objects': [o.serialize() for o in self.objects],
            'lights': [l.serialize() for l in self.lights]
        }

    @classmethod
    def squeeze_floor_plan(cls, floor_plan):
        x_min = 0
        while not np.any(floor_plan[x_min, :]):
            x_min += 1
        x_max = floor_plan.shape[0] - 1
        while not np.any(floor_plan[x_max, :]):
            x_max -= 1
        y_min = 0
        while not np.any(floor_plan[:, y_min]):
            y_min += 1
        y_max = floor_plan.shape[1] - 1
        while not np.any(floor_plan[:, y_max]):
            y_max -= 1
        floor_plan = floor_plan[x_min - 1:x_max + 2, y_min - 1:y_max + 2]
        return floor_plan

    @classmethod
    def make_surfaces(cls, floor_plan):
        floor = Surface(0, centre=Point(floor_plan.shape[0] // 2, floor_plan.shape[1] // 2, 0),
                        normal=Orientation.UP, size=(floor_plan.shape[0] - 2, floor_plan.shape[1] - 2))
        ceiling = Surface(0, centre=Point(floor_plan.shape[0] // 2, floor_plan.shape[1] // 2, 2),
                          normal=Orientation.DOWN, size=(floor_plan.shape[0] - 2, floor_plan.shape[1] - 2))

        # Find first corner at the top left
        x_global_start, y_global_start = 1, 1
        while not floor_plan[x_global_start, y_global_start]:
            x_global_start += 1
        x_start, y_start = x_global_start, y_global_start
        direction = Orientation.FRONT

        walls = []
        while True:
            x, y = x_start, y_start

            if direction == Orientation.FRONT:
                while not floor_plan[x - 1, y] and floor_plan[x, y]:
                    y += 1
                x_end, y_end = x, y
                walls.append(Surface(0, centre=Point(x, y_start + (y_end - y_start) // 2, 1),
                                     normal=Orientation.RIGHT, size=(y_end - y_start, 2)))
                x_start, y_start = x, y
                if floor_plan[x - 1, y]:
                    direction = Orientation.LEFT
                else:
                    direction = Orientation.RIGHT

            elif direction == Orientation.LEFT:
                while not floor_plan[x - 1, y - 1] and floor_plan[x - 1, y]:
                    x -= 1
                x_end, y_end = x, y
                walls.append(Surface(0, centre=Point(x_start - (x_start - x_end) // 2, y, 1),
                                     normal=Orientation.FRONT, size=(x_start - x_end, 2)))
                x_start, y_start = x, y
                if floor_plan[x - 1, y - 1]:
                    direction = Orientation.BACK
                else:
                    direction = Orientation.FRONT

            elif direction == Orientation.BACK:
                while not floor_plan[x, y - 1] and floor_plan[x - 1, y - 1]:
                    y -= 1
                x_end, y_end = x, y
                walls.append(Surface(0, centre=Point(x, y_start - (y_start - y_end) // 2, 1),
                                     normal=Orientation.LEFT, size=(y_start - y_end, 2)))
                x_start, y_start = x, y
                if floor_plan[x, y - 1]:
                    direction = Orientation.RIGHT
                else:
                    direction = Orientation.LEFT

            else:
                while not floor_plan[x, y] and floor_plan[x, y - 1]:
                    x += 1
                x_end, y_end = x, y
                walls.append(Surface(0, centre=Point(x_start + (x_end - x_start) // 2, y, 1),
                                     normal=Orientation.BACK, size=(x_end - x_start, 2)))
                x_start, y_start = x, y
                if floor_plan[x, y]:
                    direction = Orientation.FRONT
                else:
                    direction = Orientation.BACK

            if (x_start, y_start) == (x_global_start, y_global_start) and direction == Orientation.FRONT:
                break

        return floor, ceiling, walls
