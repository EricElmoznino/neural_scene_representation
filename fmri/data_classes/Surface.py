from data_classes.Point import Point
from data_classes.Orientation import Orientation


class Surface:

    def __init__(self, type: int, centre: Point, normal: Orientation, size: tuple):
        assert len(size) == 2
        assert size[0] % 2 == 0 and size[1] % 2 == 0
        self.type = type
        self.centre = centre
        self.normal = normal
        self.size = size

    def serialize(self):
        return {
            'type': self.type,
            'centre': self.centre.serialize(),
            'normal': self.normal.name,
            'size': self.size
        }
