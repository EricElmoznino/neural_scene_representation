from data_classes.Point import Point
from data_classes.Orientation import Orientation


class Object:

    def __init__(self, type: int, location: Point, rotation: Orientation, size: tuple):
        assert location.z == 0
        assert rotation in [Orientation.LEFT, Orientation.FRONT, Orientation.RIGHT, Orientation.BACK]
        assert len(size) == 2
        assert size[0] % 2 == 0 and size[1] % 2 == 0
        self.type = type
        self.location = location
        self.rotation = rotation
        self.size = size

    def serialize(self):
        return {
            'type': self.type,
            'location': self.location.serialize(),
            'rotation': self.rotation.name,
            'size': self.size
        }
