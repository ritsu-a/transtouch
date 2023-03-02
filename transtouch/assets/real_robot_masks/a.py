import numpy as np
import cv2

for i in range(1, 18):
    img = cv2.imread(f"m{i}.png")
    (m, n, k) = img.shape
    img_new = img.copy()
    for x in range(m):
        for y in range(n):
            if (img[max(0, x - 10): min(x + 11, m), max(0, y - 10): min(n, y + 11), 0].sum() == 0):
                img_new[x, y] = (0, 0, 127)
            else:
                img_new[x, y] = (127, 0, 0)
    cv2.imwrite(f"m{i}.png", img_new)