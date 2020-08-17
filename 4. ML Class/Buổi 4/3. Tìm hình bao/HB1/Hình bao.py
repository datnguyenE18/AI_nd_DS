import numpy as np
import cv2

BACKGROUND = 1
OBJECT = 0

# @param mat mảng 2d numpy
# @param wnd tên cửa sổ
# @param tileSize độ giãn ảnh
def showMat(mat, wnd="img", tileSize=20, wait=0):
    # lấy kích thước mat
    shape = mat.shape
    # giãn theo tileSize
    img = cv2.resize(mat, (shape[1]*tileSize, shape[0]*tileSize), interpolation=cv2.INTER_NEAREST)
    # hiển thị
    cv2.imshow(wnd, img)
    cv2.waitKey(wait)

def inside(mat, r, c):
    shape = mat.shape
    return r >= 0 and c >= 0 and r < shape[0] and c < shape[1]

def get4Neighbors(mat, usedStatus, r1, c1):
    dst = []
    if inside(mat, r1, c1 - 1) and usedStatus[r1][c1 - 1] == 0:
        dst.append((r1, c1 - 1))
    if inside(mat, r1, c1 + 1) and usedStatus[r1][c1 + 1] == 0:
        dst.append((r1, c1 + 1))
    if inside(mat, r1 - 1, c1) and usedStatus[r1 - 1][c1] == 0:
        dst.append((r1 - 1, c1))
    if inside(mat, r1 + 1, c1) and usedStatus[r1 + 1][c1] == 0:
        dst.append((r1 + 1, c1))
    return dst

# loang DFS
# @param mat mảng 2d
# @param usedStatus trạng thái đã duyệt hay chưa, có kích thước giống mat
#       usedStatus[r1][c1] == 1 --> điểm (r1,c1) đã duyệt
#       usedStatus[r1][c1] == 0 --> điểm (r1,c1) chưa duyệt
# @param r,c tọa độ điểm bắt đầu
# @return danh sách điểm đã loang
def DFS(mat, usedStatus, r, c):
    # danh sách điểm kết quả
    blob = []

    if not inside(mat, r, c) or usedStatus[r][c] == 1:
        return blob

    color = mat[r][c]
    stack = [(r, c)]
    while len(stack)>0:
        # lấy điểm ra khỏi stack
        r1, c1 = stack.pop()
        if usedStatus[r1][c1] == 1:
            continue

        # đánh dấu đã duyệt, đưa vào danh sách kết quả
        usedStatus[r1][c1] = 1
        blob.append((r1, c1))

        # lấy tất cả các điểm 4 láng giềng chưa duyệt
        neighbors =get4Neighbors(mat, usedStatus, r1, c1)
        for p in neighbors:
            if mat[p[0]][p[1]]==color:
                stack.append(p)
    return blob

# Hiển thị vùng đối tượng: hiển thị 2 cửa sổ cho vùng ảnh và cho hình bao
# @param shw ảnh hiển thị
# @param blob vùng đối tượng
# @param tileSize độ giãn
def showBlob(shw, blob, tileSize=50):
    # để hiển thị vùng ảnh
    blobImage = shw.copy()

    # tính min max các tọa độ
    minx=100000
    maxx=0
    miny=100000
    maxy=0
    for p in blob:
        # với mỗi điểm trong vùng, ta đổi màu, vàng
        blobImage[p[0]][p[1]] = (0, 255, 255)

        # cập nhật min max
        minx = min(minx, p[1])
        miny = min(miny, p[0])
        maxx = max(maxx, p[1])
        maxy = max(maxy, p[0])

    if minx<maxx and miny<maxy:
        showMat(blobImage, wnd="blob", tileSize=tileSize, wait=100)

        # để hiển thị hình bao
        boundingBox = shw.copy()
        cv2.rectangle(boundingBox, (minx, miny), (maxx, maxy), (0, 255, 0))
        showMat(boundingBox, wnd="bbox", tileSize=tileSize)

if __name__ == '__main__':
    # 1.1 sinh ngẫu nhiên mảng có kích thước (15, 25) có các giá trị là số nguyên >=0, <2
    mat = np.random.randint(2, size=(15, 25))
    # ép kiểu sang float
    mat = mat.astype(dtype=np.float)
    tileSize=20

    # # 1.2 đọc ảnh đen trắng
    # # a. đọc ảnh dưới dạng ảnh xám, 1 điểm ảnh 8 bit => [0, 255]
    # mat = cv2.imread("hinhbao1.bmp", cv2.IMREAD_GRAYSCALE)
    # # b. ép kiểu sang float
    # mat = mat.astype(dtype=np.float)
    # # c. chuyển về [0, 1]
    # mat = mat/255
    # tileSize = 1

    shape = mat.shape

    # hiển thị
    showMat(mat, tileSize=tileSize, wait=0)

    # trạng thái đã duyệt hay chưa, có kích thước giống mat
    usedStatus = np.zeros(shape)

    # tạo ảnh màu để hiển thị
    shw = cv2.cvtColor((mat*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Quá trình
    for i in range(shape[0]):
        for j in range(shape[1]):
            # tìm điểm đối tượng chưa duyệt để duyệt
            if usedStatus[i][j]==0 and mat[i][j]==OBJECT:
                # tìm vùng đối tượng
                blob = DFS(mat, usedStatus, i, j)
                # hiển thị vùng đối tượng
                showBlob(shw, blob, tileSize)



