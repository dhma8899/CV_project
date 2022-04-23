import numpy as np
from numpy import linalg as lg
import math


def projection_matrix(homography):
    x = np.array([[1406.084154, 0, 0],
                  [2.206797, 1417.999306, 0],
                  [1014.136434, 566.347754, 1]]).T
    temp = lg.inv(x) @ lg.inv(homography)

    b1 = temp[:, 0].reshape(3, 1)
    b2 = temp[:, 1].reshape(3, 1)
    b3 = temp[:, 2].reshape(3, 1)

    scalar_value = 2 / (np.linalg.norm(np.dot(lg.inv(x), b1)) +
                        np.linalg.norm(np.dot(lg.inv(x), b2)))
    # Calculating the transverse matrix
    trans_matrix = scalar_value * b3
    # Calculating the rotational matrix
    rot_mat1 = scalar_value * b1
    rot_mat2 = scalar_value * b2
    rot_mat3 = ((np.cross(temp[:, 0], temp[:, 1])) * scalar_value * scalar_value).reshape(3, 1)
    rot_matrix = np.concatenate((rot_mat1, rot_mat2, rot_mat3), axis=1)
    return rot_matrix, trans_matrix, x