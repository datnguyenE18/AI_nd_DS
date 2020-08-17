import random
import cv2
import numpy as np

class Point:
    def __init__( self, x, y ):
        self.x = x
        self.y = y

class Map:
    def __init__( self, width, height ):
        self.width = width
        self.height = height
        self.arr = []
        self.init()

    def init( self ):
        self.arr = []
        for i in range( self.height ):
            row = []
            for j in range( self.width ):
                row.append( random.randint( 0, 1 ) )
            self.arr.append( row )

    def show( self, tileSize = 50 ):
        img = np.array( self.arr, dtype=np.float )
        img = cv2.resize( img, ( self.height * tileSize, self.width * tileSize ), interpolation=cv2.INTER_NEAREST )
        cv2.imshow( "img", img )
        cv2.waitKey( 0 )

    def inside( self, pt ):
        return pt.x >= 0 and pt.y >= 0 and pt.y < self.height and pt.x < self.width

    def getPixel( self, pt ):
        if self.inside( pt ):
            return self.arr[pt.y][pt.x]
        else:
            raise ValueError()

    def setPixel( self, pt, val ):
        if self.inside( pt ):
            self.arr[pt.y][pt.x] = val
        else:
            raise ValueError()

    def getNeighbors( self, pt, color ):
        dst = []
        p = Point( pt.x, pt.y - 1 )
        if self.inside( p ) and self.getPixel( p ) == color:
            dst.append( p )
        p = Point( pt.x, pt.y + 1 )
        if self.inside( p ) and self.getPixel( p ) == color:
            dst.append( p )
        p = Point( pt.x - 1, pt.y )
        if self.inside( p ) and self.getPixel( p ) == color:
            dst.append( p )
        p = Point( pt.x + 1, pt.y )
        if self.inside( p ) and self.getPixel( p ) == color:
            dst.append( p )
        return dst

def DSF( map, pt ):
    if not map.inside( pt ):
        return

    color = map.getPixel( pt )
    stack = [pt]
    while len( stack ) > 0:
        pt1 = stack.pop()
        if map.getPixel( pt1 ) != color:
            continue
        map.setPixel( pt1, 1 - color )

        neighbors = map.getNeighbors( pt1, color )
        for nb in neighbors:
            stack.append( nb )

        map.show()


def BSF( map, pt ):
    if not map.inside( pt ):
        return

    color = map.getPixel( pt )
    queue = [pt]
    while len( queue ) > 0:
        pt1 = queue.pop( 0 )
        if map.getPixel( pt1 ) != color:
            continue
        map.setPixel( pt1, 1 - color )

        neighbors = map.getNeighbors( pt1, color )
        for nb in neighbors:
            queue.append( nb )

        print( queue )

        map.show()

if __name__ == '__main__':
    # Khởi tạo mảng 2 chiều
    map = Map( 5, 5 )

    map.show()
    DSF( map, Point( 1, 1 ) )
    #BSF(map, Point(1, 1))