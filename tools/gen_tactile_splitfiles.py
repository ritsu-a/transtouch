import numpy as np
import sys
import os

b = np.zeros(20)
a = np.array([1, 2, 3, 4, 6, 7, 8, 11, 12, 13, 14])
for x in a:
    b[x] = 1

s = sys.argv[1]


file = open("./split_files/tactile_" + s + ".txt", "w")

for i in range(4):
    x = np.random.randint(1, 17)
    while (b[x] == 0): x = np.random.randint(1, 17)
    b[x] = 0
    file.write(s + '-' + str(x) + '\n')

print(os.path.exists("./split_files"))
print(s)