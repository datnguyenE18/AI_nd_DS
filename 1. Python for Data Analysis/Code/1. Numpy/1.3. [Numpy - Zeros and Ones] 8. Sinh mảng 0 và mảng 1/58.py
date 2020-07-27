# Import Numpy:
import numpy as np

# Mảng 0 một chiều:
arr01 = np.zeros( 3 )
print( arr01 ) #[0.  0.  0.]

# Mảng 0 hai chiều:
arr02 = np.zeros( ( 5,5 ) )
arr02 = np.zeros( ( 5,5 ) )
print( arr02 )

# Mảng 1 một chiều:
arr11 = np.ones( 3 )
print( arr11 ) # [1.  1.  1.]

# Mảng 1 hai chiều:
arr12 = np.ones( ( 5,5 ) )
print( arr12 )

# Truy cập các mảng trên:
print( arr01[1] + arr11[2] ) # 1.0

'''
[0. 0. 0.]
[[0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]]
[1. 1. 1.]
[[1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1.]]
1.0
'''