import numpy as np


def design_shape(shape, pm):

    if shape.lower() == "cube":
        corners_shape = np.float32(
            [[0, 0, 0, 1], [0, 512, 0, 1], [512, 512, 0, 1], [512, 0, 0, 1], [0, 0, -512, 1], [0, 512, -512, 1],
             [512, 512, -512, 1], [512, 0, -512, 1]])
    elif shape.lower() == "cuboid":
        corners_shape = np.float32(
            [[0, 0, 0, 1], [0, 512, 0, 1], [512, 512, 0, 1], [512, 0, 0, 1], [0, 0, -1024, 1], [0, 512, -1024, 1],
             [512, 512, -1024, 1], [512, 0, -1024, 1]])
    elif shape.lower() == "pyramid":
        corners_shape = np.float32(
            [[0, 0, 0, 1], [0, 512, 0, 1], [512, 512, 0, 1], [512, 0, 0, 1], [256, 256, -512, 1], [256, 256, -512, 1],
             [256, 256, -512, 1], [256, 256, -512, 1]])

    #data = np.matmul(corners_shape, pm.T)
    data = corners_shape @ pm.T
    op = []
    for i in range(8):
        op.append(data[i] / data[i][2])
    stack = np.vstack(op)

    shape_data = []
    for i in range(8):
        shape_data.append([stack[i][0], stack[i][1]])

    shape_coords = np.array(shape_data)
    return shape_coords

