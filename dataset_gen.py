import random
import math


DIMENSION = 1
SIZE = 200
LEFT_EDGE = -10
RIGHT_EDGE = 10

def FUNC_TRUE(x):
    return x[0]*x[0] - 4 + random.random()*10
    # if abs(x[1]) + abs(x[0]) < 4: return 1
    # if abs(x[1]) + abs(x[0]) < 7: return 0
    # if abs(x[0] + 6) < 1.25 and abs(x[1] - 5) < 4: return 2
    # if abs(x[1] - 5) < 1.25 and abs(x[0] + 6) < 4: return 2 
    # if abs(x[1] - x[0]) < 2: return 1
    # if x[0] > 5 and x[1] < -5: return 0
    # if x[0] > 7 and x[1] < -2: return 0
    # return -1

f = open("data.txt", 'w')
f.write(str(DIMENSION) + "\n")
f.write(str(SIZE) + "\n")
f.write(str(LEFT_EDGE) + "\n")
f.write(str(RIGHT_EDGE) + "\n")
for i in range(SIZE):
    x = [LEFT_EDGE + random.random()*(RIGHT_EDGE - LEFT_EDGE) for i in range(DIMENSION)]
    if 1:
        for k in range(DIMENSION):
            f.write(str(x[k]) + " ")
    f.write (str(FUNC_TRUE(x)) + "\n")
f.close()