# Tạo ngẫu nhiên 1 mảng 2 chiều (hoặc ảnh) nhị phân:
# 1 điểm chỉ nhận 1 trong hai giá trị, thường là 0 và 1
# Coi mỗi điểm trong mảng là 1 đỉnh của đồ thị
# 2 điểm cạnh nhau (ngang, dọc, có thể chéo) thì 2 đỉnh có liên kết cạnh
# Dùng BFS và DFS để minh họa đổ màu

import random
import cv2
import numpy as np

ROWS = 5
COLS = 5

def rand2DArr():
    arr = []
    for i in range( ROWS ):
        row = []
        for j in range( COLS ):
            row.append( random.randint( 0, 1 ) )
        arr.append( row )
    return arr

def printArr( arr ):
    for row in arr:
        print( row )

def showArr( arr, blockSize = 50 ):
    img = np.asarray( arr, dtype=np.float )# ảnh số thực 32bit
    #img = np.asarray(arr, dtype=np.uint8)*255 # ảnh số nguyên 8bit
    img = cv2.resize( img, ( ROWS * blockSize, COLS * blockSize ), interpolation=cv2.INTER_NEAREST )
    cv2.imshow( "img", img )
    cv2.waitKey( 0 )

def inside( r, c ):
    return r >= 0 and c >= 0 and r < ROWS and c < COLS

def DFS( arr, r, c ):
    # Khởi tạo stack rỗng, đưa vị trí xuất phát vào stack
    stack = [( r, c )]

    color = arr[r][c]

    # lặp trong khi stack không rỗng
    while len( stack ) > 0:
        # lấy phần tử ra khỏi stack
        r1, c1 = stack.pop()

        if arr[r1][c1] != color:
            continue

        # thiết lập phần tử là đã duyệt
        arr[r1][c1] = 0.5#1-color

        # Đưa các đỉnh kề vào stack
        # xét 4 láng giềng
        if inside( r1 - 1, c1 ) and arr[r1 - 1][c1] == color:
            stack.append( ( r1 - 1, c1 ) )
        if inside( r1 + 1, c1 ) and arr[r1 + 1][c1] == color:
            stack.append( ( r1 + 1, c1 ) )
        if inside( r1, c1 - 1 ) and arr[r1][c1 - 1] == color:
            stack.append( ( r1, c1 - 1 ) )
        if inside( r1, c1 + 1 ) and arr[r1][c1 + 1] == color:
            stack.append( ( r1, c1 + 1 ) )

        print( r1, c1 )
        print( stack )
        showArr( arr )

def BFS( arr, r, c ):
    # Khởi tạo queue rỗng, đưa vị trí xuất phát vào queue
    queue = [( r, c )]

    color = arr[r][c]

    # lặp trong khi queue không rỗng
    while len( queue ) > 0:
        # lấy phần tử ra khỏi queue
        r1, c1 = queue.pop( 0 )

        if arr[r1][c1] != color:
            continue

        # thiết lập phần tử là đã duyệt
        arr[r1][c1] = 0.5#1-color

        # Đưa các đỉnh kề vào queue
        # xét 4 láng giềng
        if inside( r1 - 1, c1 ) and arr[r1 - 1][c1] == color:
            queue.append( ( r1 - 1, c1 ) )
        if inside( r1 + 1, c1 ) and arr[r1 + 1][c1] == color:
            queue.append( ( r1 + 1, c1 ) )
        if inside( r1, c1 - 1 ) and arr[r1][c1 - 1] == color:
            queue.append( ( r1, c1 - 1 ) )
        if inside( r1, c1 + 1 ) and arr[r1][c1 + 1] == color:
            queue.append( ( r1, c1 + 1 ) )

        print( r1, c1 )
        print( queue )
        showArr( arr )

if __name__ == '__main__':
    arr = rand2DArr()
    printArr( arr )
    showArr( arr )
    BFS( arr, 1, 1 )