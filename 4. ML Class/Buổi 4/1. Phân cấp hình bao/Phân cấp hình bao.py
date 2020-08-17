import cv2
import numpy as np


# Đọc ảnh và lấy kích thước:
img = cv2.imread( 'img.png' )
h = img.shape[0]
w = img.shape[1]

# Ma trận trạng thái (đã / chưa duyệt):
daDuyet = np.zeros( shape=( h, w, 1 ), dtype=bool )

# Ma trận trạng thái đã được đưa vào danh sách kề:
daVaoDSKe = np.zeros( shape=( h, w, 1 ), dtype=bool )

# Danh sách các nút và cấu trúc các nút (cây):
dsNut = []
dsMucMau = [( 255,0,0 ),( 0,255,0 ),( 0,0,255 ), ( 255,255,0 ),( 0,255,255 ),( 255,0,255 )]

# Hàm so sánh 2 mẫu:
def soSanh( mau1, mau2 ):
    return mau1[0] == mau2[0] and mau1[1] == mau2[1] and mau1[2] == mau2[2]

# Hàm xử lý điểm (j, i)
def xuLy( p, mau, dsHangXom, dsObjCungMau,queue ):
    i = p[0]
    j = p[1]
    if (0 <= i and i < h) and (0 <= j and j < w) and daDuyet[i][j] == False:
        if soSanh( img[i][j], mau ):
            queue.append( ( i, j ) )
            dsObjCungMau.append( ( i, j ) )
            daDuyet[i][j] = True
        elif daVaoDSKe[i][j] == False:
            dsHangXom.append( ( i, j ) )
            daVaoDSKe[i][j] = True

# Loang từ điểm (j, i) ra các điểm cùng màu với nó:
def BFS( i, j ):
    mau = img[i][j]
    queue = [( i, j )]
    dsOBjCungMau = []
    dsHangXom = []
    while(len( queue ) > 0):
        p = queue.pop( 0 )
        i = p[0]
        j = p[1]
        xuLy( ( i - 1, j - 1 ), mau, dsHangXom, dsOBjCungMau, queue )
        xuLy( ( i - 1, j ), mau, dsHangXom, dsOBjCungMau, queue )
        xuLy( ( i - 1, j + 1 ), mau, dsHangXom, dsOBjCungMau, queue )

        xuLy( ( i, j - 1 ), mau, dsHangXom, dsOBjCungMau, queue )
        xuLy( ( i, j + 1 ), mau, dsHangXom, dsOBjCungMau, queue )

        xuLy( ( i + 1, j - 1 ), mau, dsHangXom, dsOBjCungMau, queue )
        xuLy( ( i + 1, j ), mau, dsHangXom, dsOBjCungMau, queue )
        xuLy( ( i + 1, j + 1 ), mau, dsHangXom, dsOBjCungMau, queue )
    return dsOBjCungMau, dsHangXom

# Hàm phân cấp:
def phanCap( dsDiemNen, lv ):
    while len( dsDiemNen ) > 0:
        p = dsDiemNen.pop( 0 )
        i = p[0]
        j = p[1]
        if daDuyet[i][j]:
            continue
        dsNen, dsOBjKe = BFS( i, j )

        for p1 in dsOBjKe:
            i1 = p1[0]
            j1 = p1[1]
            if daDuyet[i1][j1]:
                continue
            dsObj, dsNenTrong = BFS( i1, j1 )
            dsNut.append( ( dsObj, lv ) )
            for p2 in dsObj:
                i2 = p2[0]
                j2 = p2[1]
                img[i2][j2] = dsMucMau[lv]
            cv2.imshow( 'image', img )
            cv2.waitKey( 0 )
            phanCap( dsNenTrong, lv + 1 )

if __name__ == '__main__':
    phanCap( [( 0, 0 )], 0 )


