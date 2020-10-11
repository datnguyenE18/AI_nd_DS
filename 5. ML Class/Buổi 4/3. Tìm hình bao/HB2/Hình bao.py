import cv2
import numpy as np
from queue import LifoQueue

class Point:
    def __init__( self, i, j ):
        self.i = i
        self.j = j

def soSanh( mau1, mau2 ):
    if mau1[0] == mau2[0] and mau1[1] == mau2[1] and mau1[2] == mau2[2]:
        return True
    else:
        return False


###################################################################################################################################
img = cv2.imread( 'img.png' )

###################################################################################################################################
w = img.shape[1]
h = img.shape[0]
daDuyet = np.zeros( shape=( h, w, 1 ), dtype=bool )

###################################################################################################################################
def laUCV( i, j ):
    if (0 <= i and i < h) and (0 <= j and j < w) and (soSanh( img[i][j],( 255,255,255 ) ) == True) and (daDuyet[i][j] == False):
        return True
    else:
        return False

def DSF( i, j ):
    stack = LifoQueue()
    stack.put( Point( i, j ) )
    mini = 10000
    minj = 10000
    maxi = 0
    maxj = 0
    while stack.empty() == False:
        p = stack.get()
        i = p.i
        j = p.j
        daDuyet[i][j] = True
        if mini > i:
            mini = i
        if minj > j:
            minj = j
        if maxi < i:
            maxi = i
        if maxj < j:
            maxj = j

        if laUCV( i + 1, j ):
            stack.put( Point( i + 1, j ) )
        if laUCV( i - 1, j ):
            stack.put( Point( i - 1, j ) )
        if laUCV( i, j + 1 ):
            stack.put( Point( i, j + 1 ) )
        if laUCV( i, j - 1 ):
            stack.put( Point( i, j - 1 ) )

        if laUCV( i - 1, j - 1 ):
            stack.put( Point( i - 1, j - 1 ) )
        if laUCV( i - 1, j + 1 ):
            stack.put( Point( i - 1, j + 1 ) )
        if laUCV( i + 1, j - 1 ):
            stack.put( Point( i + 1, j - 1 ) )
        if laUCV( i + 1, j + 1 ):
            stack.put( Point( i + 1, j + 1 ) )
        #img[i][j] = (0,0,255)

    tl = Point( mini, minj )
    br = Point( maxi, maxj )
    return ( tl, br )

for i in range( h ):
    for j in range( w ):
        if soSanh( img[i][j], ( 255, 255, 255 ) ) and daDuyet[i][j] == False:
            tl, br = DSF( i, j )
            cv2.rectangle( img, ( tl.j, tl.i ), ( br.j, br.i ), ( 0,255,0 ) )
            cv2.imshow( 'image',img )
            cv2.waitKey( 100 )

cv2.waitKey( 0 )