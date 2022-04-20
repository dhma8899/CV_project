import numpy as np
from numpy import linalg as lg
import math


def projection_matrix(homography):
    x = np.array([[1406.08415449821, 0, 0],
                  [2.20679787308599, 1417.99930662800, 0],
                  [1014.13643417416, 566.347754321696, 1]]).T
    temp = np.dot(lg.inv(x), homography)
    temp2 = lg.inv(x) @ homography
    #print(temp)
    #print("")
    #print(temp2)
    col_data = [temp2[:, 0], temp2[:, 1], temp2[:, 2]]
    eq = 1 / math.sqrt(lg.norm(col_data[0], 2) * lg.norm(col_data[1], 2))
    op = [col_data[0] * eq, col_data[1] * eq, col_data[2] * eq]
    cross = np.cross(op[0]+op[1], np.cross(op[0], op[1]))
    output = [
        np.dot((op[0] + op[1]) / lg.norm(op[0] + op[1], 2) + cross / lg.norm(op[0] + op[1], 2), 1 / math.sqrt(2)),
        np.dot((op[0] + op[1]) / lg.norm(op[0] + op[1], 2) - cross / lg.norm(op[0] + op[1], 2), 1 / math.sqrt(2))]

    ret = np.array((output[0], output[1], np.cross(output[0], output[1]), op[2])).T
    return np.dot(x, ret)
