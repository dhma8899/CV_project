import cv2
import numpy as np


def impose_cube(image, coords):
    coords = np.int32(coords).reshape(-1, 2)
    image = cv2.drawContours(image, [coords[:4]], -1, (255, 255, 255), -3)

    for i, j in zip(range(4), range(4, 8)):
        image = cv2.line(image, tuple(coords[i]), tuple(coords[j]), (71, 100 ,8), 3)
    image = cv2.drawContours(image, [coords[4:]], -1, (123, 231, 312), 3)
    return image
