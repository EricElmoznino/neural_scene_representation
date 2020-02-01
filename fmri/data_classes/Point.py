class Point:

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def serialize(self):
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z,
        }
