import random
import math
import cv2
import numpy as np

# cạnh hình vuông
d = 20 
# bán kính hình tròn
R = 3
# số lần thử (số lần sinh mẫu điểm)
tries = 1000000
# số điểm nằm trong hình tròn
inside = 0
# tâm hình tròn
x = 5 
y = -3
# giới hạn tọa độ sinh điểm
lmt = d / 2

d2 = 500
rows = d2
cols = d2
# tạo 1 ảnh kích thước (rows, cols), 3 kênh
shw = np.zeros( shape=( rows, cols, 3 ), dtype=np.uint8 )
# chuyển sang màu trắng
for i in range( rows ):
    for j in range( cols ):
        shw[i][j] = ( 255, 255, 255 )

def transform( x, y, tl1x, tl1y, d1, tl2x, tl2y, d2 ):
    x2 = ((x - tl1x) / d1) * d2 + tl2x
    y2 = ((y - tl1y) / d1) * d2 + tl2y
    return int( x2 ), int( y2 )

# vẽ hình tròn
cx, cy = transform( x, y, -d / 2, -d / 2, d, 0, 0, d2 )
R1 = int( R / d * d2 )
cv2.circle( shw, ( cx, cy ), R1, ( 0, 0, 0 ) )

# hiển thị ảnh
cv2.imshow( "img", shw )
cv2.waitKey( 0 )

# lặp để sinh điểm
for i in range( tries ):
    # sinh điểm
    rand_x = random.uniform( -lmt, lmt ) 
    rand_y = random.uniform( -lmt, lmt ) 

    # tọa độ trên ảnh
    x2, y2 = transform( rand_x, rand_y, -d / 2, -d / 2, d, 0, 0, d2 )
    print( x2, y2 )

    # kiểm tra nằm trong hình tròn
    if (math.sqrt( (rand_x - x) ** 2 + (rand_y - y) ** 2 ) <= R):
        cv2.circle( shw, ( x2, y2 ), 2, ( 255, 0, 0 ), -1 )
        inside += 1
    else:
        cv2.circle( shw, ( x2, y2 ), 2, ( 0, 255, 0 ), -1 )

    # hiển thị ảnh
    cv2.imshow( "img", shw )
    cv2.waitKey( 100 )


# tính Pi:
pi = (d ** 2 * inside / tries) / (R ** 2)
print( pi ) # KQ: 3.1329333333333333
