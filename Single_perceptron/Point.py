import random
import Function


class Point:
    SIZE_POPULATION = random.randint(50, 200)

    def __init__(self):
        self.x = random.uniform(-1, 1)
        self.y = random.uniform(-1, 1)
        self.bias = 1.0
        if self.y >= Function.Function.f(self.x):
            self.label = 1
        else:
            self.label = -1
