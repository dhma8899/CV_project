import numpy as np
import cv2


def design_shape(shape, rotation, translation, pm):

    if shape.lower() == "cube":
        corners_shape = np.float32(
            [[0, 0, 0], [0, 100, 0], [100, 100, 0], [100, 0, 0], [0, 0, -25], [0, 100, -25],
             [100, 100, -25], [100, 0, -25]])
    elif shape.lower() == "cuboid":
        corners_shape = np.float32(
            [[0, 0, 0], [0, 100, 0], [100, 100, 0], [100, 0, 0], [0, 0, -50], [0, 100, -50],
             [100, 100, -50], [100, 0, -50]])
    elif shape.lower() == "pyramid":
        corners_shape = np.float32(
            [[0, 0, 0], [0, 100, 0], [100, 100, 0], [100, 0, 0], [50, 50, -25], [50, 50, -25],
             [50, 50, -25], [50, 50, -25]])

    structure, _ = cv2.projectPoints(corners_shape, rotation, translation, pm, np.zeros((1, 4)))

    return structure
