import random
import math

d = 20 
R = 3 
tries = 1000000 
inside = 0 
x = 5 
y = -3 
lmt = d / 2

for i in range(tries):
    rand_x = random.uniform(-lmt, lmt) 
    rand_y = random.uniform(-lmt, lmt) 
    
    if (math.sqrt((rand_x - x)**2 + (rand_y - y)**2) <= R):
        inside += 1

pi = (d**2 * inside / tries) / (R**2)
print (pi) # KQ: 3.1329333333333333
