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
    elif bin_val[5, 2] == 1:
        b, c, d, a = data
    elif bin_val[5, 5] == 1:
        c, d, a, b = data
    elif bin_val[2, 5] == 1:
        d, a, b, c = data
    else:
        a, b, c, d = data

    new_corners = np.array([[a, b, c, d]])
    return new_corners

def binary(x):
    for i in range(len(x)):
        for j in range(len(x)):
            x[i, j] = 0
            if x[i, j] > 165:
                x[i, j] = 1
    return x