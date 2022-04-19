import numpy as np
from numpy import linalg as lg


def homography(vector1, vector2):
    hlist = []

    for i in range(4):
        hlist.append([-vector1[i][0], -vector1[i][1], -1, 0, 0, 0, vector2[i, 0] * vector1[i][0], vector2[i, 0] * vector1[i][1], vector2[i, 0]])
        hlist.append([0, 0, 0, -vector1[i][0], -vector1[i][1], -1, vector2[i, 1] * vector1[i][0], vector2[i, 1] * vector1[i][1], vector2[i, 1]])
    H = np.array(hlist)

    a, b, xt = lg.svd(H)
    x = xt[8:, ] / xt[8][8]

    homography_matrix = np.reshape(x, (3, 3))
    return homography_matrix
