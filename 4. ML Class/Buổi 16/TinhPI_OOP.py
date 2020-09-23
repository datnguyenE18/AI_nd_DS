import random
import math
import cv2
import numpy as np

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def distance(self, p):
        return math.sqrt((p.x - self.x)**2 + (p.y - self.y)**2)

# # quan hệ has a
# class Circle:
#     def __init__(self, center, radius):
#         self.center = center
#         self.radius = radius
#     def contain(self, p):
#         return self.center.distance(p) <= self.radius
#     def getCenter(self):
#         return self.center

# quan hệ is a
class Circle(Point):
    def __init__(self, center, radius):
        Point.__init__(self, center.x, center.y)
        self.radius = radius
    def contain(self, p):
        return self.distance(p) <= self.radius
    def getCenter(self):
        return self

class Square:
    def __init__(self, topLeft, size):
        self.topLeft = topLeft
        self.size = size
    def randPoint(self):
        x = random.uniform(self.topLeft.x, self.topLeft.x+self.size)
        y = random.uniform(self.topLeft.y, self.topLeft.y+self.size)
        return Point(x, y)


class Transform:
    def __init__(self, scale=1.0, translate=(0.0,0.0)):
        self.scale = scale
        self.translate = translate
    def forward(self, p):
        x = p.x * self.scale + self.translate[0]
        y = p.y * self.scale + self.translate[1]
        return Point(x,y)
    def forwardScaleOnly(self, v):
        return v*self.scale

class DisplayDevice:
    def __init__(self, size, square, color=(255, 255, 255)):
        self.image = (color - np.zeros(shape=(size, size, 3))).astype(np.uint8)
        self.transform = Transform(size/square.size,
                                   (-square.topLeft.x*size/square.size, -square.topLeft.y*size/square.size))
    def drawPoint(self, p, color):
        p1 = self.transform.forward(p)
        cv2.circle(self.image, (int(p1.x), int(p1.y)), 2, color, -1)

    def drawCircle(self, c, color=(0, 0, 0)):
        R1 = self.transform.forwardScaleOnly(c.radius)
        c1 = self.transform.forward(c.getCenter())
        cv2.circle(self.image, (int(c1.x), int(c1.y)), int(R1), color)

    def show(self, waitedTime):
        cv2.imshow("img", self.image)
        cv2.waitKey(waitedTime)

if __name__ == '__main__':
    square = Square(Point(0, 0), 1)
    circle = Circle(Point(0.4, 0.6), 0.4)

    device = DisplayDevice(500, square)
    device.drawCircle(circle)
    device.show(0)

    nInsides = 0
    nRands = 1000000
    for i in range(nRands):
        p = square.randPoint()

        if circle.contain(p):
            nInsides += 1
            device.drawPoint(p, (0, 255, 0))
        else:
            device.drawPoint(p, (0, 0, 255))
        device.show(100)


