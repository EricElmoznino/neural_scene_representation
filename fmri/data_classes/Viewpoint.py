from fmri.data_classes.Point import Point


class Viewpoint:

    def __init__(self, location: Point, rotation: float, horizon: float):
        self.location = location
        self.rotation = rotation
        self.horizon = horizon

    def serialize(self):
        return {
            'location': self.location.serialize(),
            'rotation': self.rotation,
            'horizon': self.horizon
        }
