import numpy as np
from numpy import linalg as lg
import cv2


def design_shape(shape1, shape2, pm):
    if shape1.lower() == "cube":
        corners_shape = np.float32(
            [[0, 0, 0, 1], [0, 512, 0, 1], [512, 512, 0, 1], [512, 0, 0, 1], [0, 0, -512, 1], [0, 512, -512, 1],
             [512, 512, -512, 1], [512, 0, -512, 1]])
    elif shape1.lower() == "cuboid":
        corners_shape = np.float32(
            [[0, 0, 0, 1], [0, 512, 0, 1], [512, 512, 0, 1], [512, 0, 0, 1], [0, 0, -1024, 1], [0, 512, -1024, 1],
             [512, 512, -1024, 1], [512, 0, -1024, 1]])
    elif shape1.lower() == "pyramid":
        corners_shape = np.float32(
            [[0, 0, 0, 1], [0, 512, 0, 1], [512, 512, 0, 1], [512, 0, 0, 1], [256, 256, -512, 1], [256, 256, -512, 1],
             [256, 256, -512, 1], [256, 256, -512, 1]])

    if shape2.lower() == "cube":
        corners_shape2 = np.float32(
            [[0, 0, -512, 1], [0, 512, -512, 1], [512, 512, -512, 1], [512, 0, -512, 1], [0, 0, -1024, 1],
             [0, 512, -1024, 1],
             [512, 512, -1024, 1], [512, 0, -1024, 1]])
    elif shape2.lower() == "cuboid":
        corners_shape2 = np.float32(
            [[0, 0, -512, 1], [0, 512, -512, 1], [512, 512, -512, 1], [512, 0, -512, 1], [0, 0, -1536, 1], [0, 512, -1536, 1],
             [512, 512, -1536, 1], [512, 0, -1536, 1]])
    elif shape2.lower() == "pyramid":
        corners_shape2 = np.float32(
            [[0, 0, -512, 1], [0, 512, -512, 1], [512, 512, -512, 1], [512, 0, -512, 1], [256, 256, -1024, 1], [256, 256, -1024, 1],
             [256, 256, -1024, 1], [256, 256, -1024, 1]])

    #data = np.matmul(corners_shape, pm.T)
    data = corners_shape @ pm.T
    #data2 = np.matmul(corners_shape2, pm.T)
    data2 = corners_shape2 @ pm.T

    op = []
    op2 = []
    for i in range(8):
        op.append(np.divide(data[i], data[i][2]))
        op2.append(np.divide(data2[i], data2[i][2]))
    shape_data = []
    shape_data2 = []
    for i in range(8):
        #shape_data.append([stack[i][0], stack[i][1]])
        #shape_data2.append([stack2[i][0], stack2[i][1]])
        shape_data.append([op[i][0], op[i][1]])
        shape_data2.append([op2[i][0], op2[i][1]])
    shape_coords = np.array(shape_data)
    shape_coords2 = np.array(shape_data2)
    return shape_coords, shape_coords2


def impose_cube(image, coords1, coords2):
    coords1 = np.int32(coords1).reshape(-1, 2)
    coords2 = np.int32(coords2).reshape(-1, 2)
    image = cv2.drawContours(image, [coords1[:4]], -1, (255, 255, 255), -3)
    image = cv2.drawContours(image, [coords2[:4]], -1, (255, 255, 255), -3)

    for i, j in zip(range(4), range(4, 8)):
        image = cv2.line(image, tuple(coords1[i]), tuple(coords1[j]), (71, 100, 8), 3)
        image = cv2.line(image, tuple(coords2[i]), tuple(coords2[j]), (71, 100, 8), 3)
    image = cv2.drawContours(image, [coords1[4:]], -1, (123, 231, 312), 3)
    image = cv2.drawContours(image, [coords2[4:]], -1, (123, 231, 312), 3)
    return image


def homography(vector1, vector2):
    H = np.array(
        [[vector1[0][0], vector1[0][1], 1, 0, 0, 0, -vector2[0, 0] * vector1[0][0], -vector2[0, 0] * vector1[0][1],
          -vector2[0, 0]],
         [0, 0, 0, vector1[0][0], vector1[0][1], 1, -vector2[0, 1] * vector1[0][0], -vector2[0, 1] * vector1[0][1],
          -vector2[0, 1]],
         [vector1[1][0], vector1[1][1], 1, 0, 0, 0, -vector2[1, 0] * vector1[1][0], -vector2[1, 0] * vector1[1][1],
          -vector2[1, 0]],
         [0, 0, 0, vector1[1][0], vector1[1][1], 1, -vector2[1, 1] * vector1[1][0], -vector2[1, 1] * vector1[1][1],
          -vector2[1, 1]],
         [vector1[2][0], vector1[2][1], 1, 0, 0, 0, -vector2[2, 0] * vector1[2][0], -vector2[2, 0] * vector1[2][1],
          -vector2[2, 0]],
         [0, 0, 0, vector1[2][0], vector1[2][1], 1, -vector2[2, 1] * vector1[2][0], -vector2[2, 1] * vector1[2][1],
          -vector2[2, 1]],
         [vector1[3][0], vector1[3][1], 1, 0, 0, 0, -vector2[3, 0] * vector1[3][0], -vector2[3, 0] * vector1[3][1],
          -vector2[3, 0]],
         [0, 0, 0, vector1[3][0], vector1[3][1], 1, -vector2[3, 1] * vector1[3][0], -vector2[3, 1] * vector1[3][1],
          -vector2[3, 1]]])

    a, b, xt = lg.svd(H)
    x = xt[8:, ] / xt[8][8]

    homography_matrix = np.reshape(x, (3, 3))
    return homography_matrix
