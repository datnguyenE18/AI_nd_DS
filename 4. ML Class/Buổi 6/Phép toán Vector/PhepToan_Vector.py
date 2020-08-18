import numpy as np
import cv2
###############################################
def tichVoHuong( v1, v2 ):
    assert v1.shape[0] == v2.shape[0]

    v = 0
    for i in range( v1.shape[0] ):
        v = v + v1[i] * v2[i]
    return v

def test_tichVoHuong():
    v1 = np.random.random( size=(2) )
    v2 = np.random.random( size=(2) )
    print( v1 )
    print( v2 )
    print( tichVoHuong( v1, v2 ) )
###############################################
def tichChapVector( v1, v2 ):
    assert v1.shape[0] <= v2.shape[0]

    v = np.zeros( shape=(v2.shape[0] - v1.shape[0] + 1) )
    for i in range( v.shape[0] ):
        v[i] = tichVoHuong( v1, v2[i:i + v1.shape[0]] )
    return v

def test_tichChapVector():
    v1 = np.random.random( size=(2) )
    v2 = np.random.random( size=(6) )
    print( v1 )
    print( v2 )
    print( tichChapVector( v1, v2 ) )
###############################################
def tichVoHuongMaTran( m1, m2 ):
    assert m1.shape[0] == m2.shape[0] and m1.shape[1] == m2.shape[1]

    v = 0
    for i in range( m1.shape[0] ):
        for j in range( m1.shape[1] ):
            v = v + m1[i][j] * m2[i][j]
    return v

def phepCuon( m1, m2 ):
    s1 = m1.shape
    s2 = m2.shape

    assert s1[0] <= s2[0] and s1[1] <= s2[1]

    m = np.zeros( shape=( s2[0] - s1[0] + 1, s2[1] - s1[1] + 1 ) )
    for i in range( m.shape[0] ):
        for j in range( m.shape[1] ):
            m[i][j] = tichVoHuongMaTran( m1,
                                        m2[i:i + s1[0],j:j + s1[1]] )
    return m

def test_phepCuon():
    m1 = np.random.random( size=( 2, 2 ) )
    m2 = np.random.random( size=( 3, 3 ) )
    print( m1 )
    print( m2 )
    print( phepCuon( m1, m2 ) )
###############################################
def SSD( m1, m2 ):
    assert m1.shape[0] == m2.shape[0] and m1.shape[1] == m2.shape[1]

    v = 0
    for i in range( m1.shape[0] ):
        for j in range( m1.shape[1] ):
            v = v + (m1[i][j] - m2[i][j]) ** 2
    return v

# p1: (x, y)
# p2: (x+dx, y+dy)
def soKhopSSD( m1, m2, p1, p2, r ):
    # cần kiểm tra kích thước ?
    # ...
    return SSD( m1[p1[0] - r:p1[0] + r + 1, p1[1] - r:p1[1] + r + 1], m2[p2[0] - r:p2[0] + r + 1, p2[1] - r:p2[1] + r + 1] )

def test_soKhopSSD():
    m1 = np.random.randint( 2, size=( 5, 5 ) )
    m2 = np.random.randint( 2, size=( 5, 5 ) )
    print( m1 )
    print( m2 )
    print( soKhopSSD( m1, m2, ( 1,1 ), ( 2,2 ), 1 ) )
###############################################



