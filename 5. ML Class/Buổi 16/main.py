from TinhPI_OOP import Square, Point, DisplayDevice
import numpy as np
import random
import math
import cv2

# Point 2D ########################
class Point1(Point):
    def add(self, p):
        return Point1(self.x + p.x, self.y + p.y)
    def divide(self, s):
        return Point1(self.x/s, self.y/s)
    def clone(self):
        return Point1(self.x, self.y)
    def nearest(self, pts):
        idx = -1
        minDist = 1000000
        for i in range(len(pts)):
            dist = self.distance(pts[i])
            if dist<minDist:
                minDist = dist
                idx = i
        return idx


class Square1(Square):
    def randPointN(self, mean, stddev):
        x = np.random.normal(mean[0], stddev[0])
        y = np.random.normal(mean[1], stddev[1])

        if x<self.topLeft.x:
            x = self.topLeft.x
        elif x>=(self.topLeft.x+self.size):
            x = self.topLeft.x+self.size-1
        if y<self.topLeft.y:
            y = self.topLeft.y
        elif y>=(self.topLeft.y+self.size):
            y = self.topLeft.y+self.size-1

        return Point1(x, y)

def calcCenter(pts, indexs, id):
    center = Point1(0, 0)
    n = 0
    for i in range(len(indexs)):
        if indexs[i]==id:
            center = center.add(pts[i])
            n = n+1
    center = center.divide(n)
    return center

def testPoint2D():
    pts = []
    square = Square1(Point1(0, 0), 1)
    device = DisplayDevice(500, square)
    # random points
    for i in range(30):
        p = square.randPointN((0.2, 0.7), (0.1, 0.15))
        pts.append(p)
        device.drawPoint(p, (255,0, 0))
    for i in range(40):
        p = square.randPointN((0.4, 0.3), (0.15, 0.1))
        pts.append(p)
        device.drawPoint(p, (0, 255,0))
    for i in range(50):
        p = square.randPointN((0.7, 0.5), (0.1, 0.15))
        pts.append(p)
        device.drawPoint(p, (0, 0, 255))

    device.show(0)

    # k = 3
    # centers = [pts[0].clone(), pts[1].clone(), pts[2].clone()]

    k = 6
    centers = [pts[0].clone(), pts[1].clone(), pts[2].clone(), pts[3].clone(), pts[4].clone(), pts[5].clone()]

    colors = [(255,0, 0), (0, 255,0), (0, 0, 255), (255,255, 0), (0, 255,255), (255, 0, 255)]
    #indexs = [-1 for i in range(len(pts))]
    while True:
        indexs = [pts[i].nearest(centers) for i in range(len(pts))]
        newCenters = [calcCenter(pts, indexs, i) for i in range(k)]

        err = 0
        for i in range(k):
            err = err + centers[i].distance(newCenters[i])
        if err < 0.001:
            break

        centers = newCenters
        for i in range(len(pts)):
            device.drawPoint(pts[i], colors[indexs[i]])

        device.show(0)
    print("stop")
    device.show(0)
# image ######################################################
def getInitCenter(image, centers):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if len(centers)==0:
                return image[i][j]
            else:
                exists = False
                for k_ in range(len(centers)):
                    dist = math.sqrt((image[i][j][0] - centers[k_][0]) ** 2 +
                                     (image[i][j][1] - centers[k_][1]) ** 2 +
                                     (image[i][j][2] - centers[k_][2]) ** 2)
                    if dist<10:
                        exists = True
                if exists==False:
                    return image[i][j]
    return (-1, -1, -1)


def testImage():
    image = cv2.imread("dhtn1.jpg").astype(np.float)
    shw = np.zeros(image.shape, np.uint8)
    k = 3

    centers = []
    for k_ in range(k):
        center = getInitCenter(image, centers)
        if center[0]==-1 or center[1]==-1 or center[2]==-1:
            return
        centers.append(center)
    #print(centers)

    indexs = np.zeros((image.shape[0], image.shape[1]), dtype = np.int)
    loop_id = 0
    while True:
        # 1
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                id = -1
                minDist = 1000000
                for k_ in range(k):
                    dist = math.sqrt((image[i][j][0]-centers[k_][0])**2 +
                                     (image[i][j][1]-centers[k_][1])**2 +
                                     (image[i][j][2] - centers[k_][2]) ** 2)
                    if dist<minDist:
                        minDist = dist
                        id = k_
                indexs[i][j] = id
        #print(indexs)
        # 2
        newCenters = []
        for k_ in range(k):
            newCenter = (0, 0, 0)
            n = 0
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    if indexs[i][j] == k_:
                        newCenter = (newCenter[0] +  image[i][j][0], newCenter[1] + image[i][j][1], newCenter[2] + image[i][j][2])
                        n = n+1
            newCenter = (newCenter[0] / n, newCenter[1] / n, newCenter[2] / n)
            newCenters.append(newCenter)

        err = 0
        for k_ in range(k):
            err = err + math.sqrt((newCenters[k_][0]-centers[k_][0])**2 +
                                     (newCenters[k_][1]-centers[k_][1])**2 +
                                     (newCenters[k_][2] - centers[k_][2]) ** 2)
        if err < 0.001:
            break
        centers = newCenters

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                shw[i][j][0] = int(centers[indexs[i][j]][0])
                shw[i][j][1] = int(centers[indexs[i][j]][1])
                shw[i][j][2] = int(centers[indexs[i][j]][2])
        cv2.imshow(str(loop_id), shw)
        cv2.waitKey(33)
        loop_id = loop_id+1

    print("stop")
    cv2.imshow("img", shw)
    cv2.waitKey(0)

testPoint2D()
# testImage()