import cv2
import numpy as np


def get_tag_positions(cornors, tag_img):
    grayscale = cv2.cvtColor(tag_img, cv2.COLOR_BGR2GRAY)
    bin_val = binary(grayscale)
    data = []
    for i in range(4):
        data.append(cornors[0][i])

    if bin_val[2, 2] == 1:
        a, b, c, d = data
        x, y, z, w = bin_val[4, 4], bin_val[4, 3], bin_val[3, 3], bin_val[3, 4]
    elif bin_val[5, 2] == 1:
        b, c, d, a = data
        y, z, w, x = bin_val[4, 4], bin_val[4, 3], bin_val[3, 3], bin_val[3, 4]
    elif bin_val[5, 5] == 1:
        c, d, a, b = data
        z, w, x, y = bin_val[4, 4], bin_val[4, 3], bin_val[3, 3], bin_val[3, 4]
    elif bin_val[2, 5] == 1:
        d, a, b, c = data
        w, x, y, z = bin_val[4, 4], bin_val[4, 3], bin_val[3, 3], bin_val[3, 4]
    else:
        a, b, c, d = data
        x, y, z, w = bin_val[4, 4], bin_val[4, 3], bin_val[3, 3], bin_val[3, 4]

    new_corners = np.array([[a, b, c, d]])
    tag_val = w * 8 + z * 4 + y * 2 + z * 1
    return new_corners, tag_val

def binary(x):
    for i in range(len(x)):
        for j in range(len(x)):
            x[i, j] = 0
            if x[i, j] > 150:
                x[i, j] = 1
    return x