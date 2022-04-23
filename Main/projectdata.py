import cv2
import numpy as np


def project_cube(image, coords):
    coords = np.int32(coords).reshape(-1, 2)

    image = cv2.drawContours(image, [coords[:4]], -1, (255, 255, 255), -3)

    for i in range(4):
        j = i + 4
        image = cv2.line(image, tuple(coords[i]), tuple(coords[j]), (71, 100, 8), 3)
    image = cv2.drawContours(image, [coords[4:]], -1, (123, 231, 312), 3)

    return image


def project_image(frame, joinimage, corners, homography):
    temp = cv2.warpPerspective(joinimage, homography, (frame.shape[1], frame.shape[0]))
    cv2.fillConvexPoly(frame, corners, 0, 16)

    frame = frame + temp
    return frame
