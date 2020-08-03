import random
import math

# Giả sử giao điểm giữa 2 đường chéo của hình vuông có tọa độ (0, 0)
d = 20 # Cạnh hình vuông
R = 3 # Bán kính hình tròn
tries = 1000000 # Số lần thử
inside = 0 
x = 5 # Hoành độ tâm hình tròn
y = -3 # Tung độ tâm hình tròn
lmt = d / 2

for i in range(tries):
    rand_x = random.uniform(-lmt, lmt) 
    rand_y = random.uniform(-lmt, lmt) 
    
    if (math.sqrt((rand_x - x)**2 + (rand_y - y)**2) <= R):
        inside += 1

pi = (d**2 * inside / tries) / (R**2)
print (pi) # KQ: 3.1329333333333333
