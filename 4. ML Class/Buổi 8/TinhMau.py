import random
import math
import cv2
import numpy as np
from TinhPI_OOP import Point, Square, DisplayDevice

def test():
    # vùng hình vuông sẽ sinh điểm
    square = Square(Point(0, 0), 1)
    # cửa sổ hiển thị
    device = DisplayDevice(500, square)
    # danh sách màu
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

    # số điểm sẽ sinh
    nRands = 20
    points = []
    # sinh các điểm và màu
    for i in range(nRands):
        # sinh tọa độ
        p = square.randPoint()
        # sinh màu
        c = random.randint(0, len(colors)-1)
        # đưa vào danh sách
        points.append((p, c))

    # vẽ các điểm
    for i in range(len(points)):
        device.drawPoint(points[i][0], colors[points[i][1]])

    # a. Nhập mới tọa độ 1 điểm, đánh giá màu của điểm này bằng cách tìm điểm gần nhất
    p1 = square.randPoint()
    minDist = 1000000 # khoảng cách nhỏ nhất
    idx = -1 # index của điểm gần nhất
    for i in range(len(points)):
        dist = p1.distance(points[i][0])
        if dist<minDist:
            minDist = dist
            idx = i

    if idx==-1:
        print("khong tim duoc ket qua")
    else:
        print("ket qua: ", colors[points[idx][1]])
        device.drawPoint(p1, colors[points[idx][1]], 4)
        device.show(0)


