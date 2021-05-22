#Creat a list:
my_list = [1, 2, 3]

# display that list:
print( my_list )

# Import numpy:
import numpy as np

# case that List as an Array
print( np.array( my_list ) )

# Creat and save that array:
arr = np.array( my_list )
print( arr )

# Note: My_list is not changed:
print( my_list )

'''
[1, 2, 3]
[1 2 3]
[1 2 3]
[1, 2, 3]
'''