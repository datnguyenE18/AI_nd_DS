# Tạo ngẫu nhiên 1 mảng 2 chiều (hoặc ảnh) nhị phân:
# 1 điểm chỉ nhận 1 trong hai giá trị, thường là 0 và 1
# Coi mỗi điểm trong mảng là 1 đỉnh của đồ thị
# 2 điểm cạnh nhau (ngang, dọc, có thể chéo) thì 2 đỉnh có liên kết cạnh
# Dùng BFS và DFS để minh họa đổ màu

import random
import cv2
import numpy as np

# Size (Số hàng & cột):
rows = 5
cols = 5

#============================================================================#

# Tạo ngẫu nhiên mảng 2 chiều nhị phân:
def rand2DArr():
    arr = []
    for i in range( rows ):
        row = []
        for j in range( cols ):
            row.append( random.randint( 0, 1 ) )
        arr.append( row )
    return arr

#============================================================================#

# in ra ma trận ra màn hình:
def disArr( arr ):
    for row in arr:
        print( row )

#============================================================================#

# Minh họa ma trận nhị phân bằng các ô đen trắng:
def showArr( arr, times = 50 ):
    img = np.asarray( arr, dtype = np.float ) # ma trận mô phỏng ảnh số thực 32bit
    img = cv2.resize( img, ( rows * times, cols * times ), interpolation = cv2.INTER_NEAREST ) # tăng kích thước ma trận lên times lần
    cv2.imshow( "img", img )
    cv2.waitKey( 0 )

#============================================================================#

# Kiểm tra tính hợp lệ của vị trí đang xét
def inside( r, c ):
    return r >= 0 and c >= 0 and r < rows and c < cols

#============================================================================#

# Dùng DFS đánh dấu:
def DFS_mark( arr, r, c ):
    # Khởi tạo stack rỗng, đưa vị trí ban đầu vào stack
    stack = [( r, c )] # ô có màu gốc
    color = arr[r][c] # đánh dấu giá trị/ màu vị trí đầu tiên để quét các vị trí có giá trị khác/
                      # tương đương bên dưới
    
    # lặp cho đến khi stack rỗng
    while(len( stack ) > 0):
        r1, c1 = stack.pop() # lấy phần tử trên cùng ra khỏi stack
        
        if arr[r1][c1] != color: # Nếu ô này có màu sắc khác với màu gốc thì đổi màu ô đó
            continue
        
        # thiết lập phần tử là đã duyệt
        arr[r1][c1] = 0.5 # 1-color
        
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

#============================================================================#
def BFS_mark( arr, r, c ):
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

#============================================================================#
if __name__ == '__main__':
    arr = rand2DArr()
    disArr( arr )
    showArr( arr )
    BFS_mark( arr, 1, 1 )