import cv2
import numpy as np
import Homography as H_matrix
import cannyedge
import positions2 as ps
import projectdata as pd
import projection_matrix as pm
import design_shape as ds
import output
import sys
import matplotlib.pyplot as plt
import imageio


try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

def operation():
    video_frames = []
    shape = input("Enter structure Cube, Cuboid, Pyramid, image")
    method = input("Enter Binary, Canny")

    frame = cv2.VideoCapture('input/Tag1.mp4')
    response, image = frame.read()
    old_positions = 0
    pdimage = cv2.imread('lena.jpeg')
    pdimageshape = pdimage.shape
    print(pdimageshape)
    row = col = pdimageshape[0]
    count = 0
    while response:
        dims = image.shape
        size = (dims[1], dims[0])
        print(count)
        count+=1
        # Edge detection process
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_grayscale = cv2.GaussianBlur(grayscale, (3, 3), cv2.BORDER_DEFAULT)



        # threshold and find the contours.
        # Contours can be explained simply as a curve joining
        # all the continuous points (along the boundary), having same color or intensity.


        if method.lower() == "canny":
            #Canny edge method
            edge_detection = cannyedge.canny_edge_detection(image).astype(np.uint8)
            #edge1 = cv2.Canny(image, 60, 180)
        elif method.lower() == "binary":
            #Threshold Method
            ret, edge_detection = cv2.threshold(blur_grayscale, 165, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(edge_detection, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



        corners = corners_identification(hierarchy, contours)

        if len(corners) == 0:
            corners = old_positions


        # Get the homography of the tag
        tag_img_resized, homography_matrix = calculate_homography_wrap(corners, image)


        new_corners = ps.get_tag_positions(corners, tag_img_resized)

        if shape.lower() == "image":
            inptcoords = np.array([[0, 0], [row - 1, 0], [row - 1, col - 1], [0, col - 1]])
            outptcoords = np.concatenate(corners)
            new_homography_matrix = H_matrix.homography(inptcoords, outptcoords)
            image = pd.project_image(image, pdimage, new_corners, new_homography_matrix)
        else:
            inptcoords = np.array([[0, 0], [row - 1, 0], [row - 1, col - 1], [0, col - 1]])
            outptcoords = np.concatenate(new_corners)
            new_homography_matrix = H_matrix.homography(inptcoords, outptcoords)
            image = pd.project_image(image, pdimage, new_corners, new_homography_matrix)
            rotation, translation, pmatrix = pm.projection_matrix(homography_matrix)
            structure = ds.design_shape(shape, rotation, translation, pmatrix)
            image = pd.project_cube(image, structure)

        old_positions = corners
        video_frames.append(image)
        response, image = frame.read()
    return video_frames, size


def corners_identification(hierarchy, contours):
    contours_points = []
    contour_details = [[cv2.contourArea(contour), cv2.approxPolyDP(contour, cv2.arcLength(contour, True) * 0.05, True), meta_data] for meta_data, contour in zip(hierarchy[0], contours)]
    for area, shape, meta_data in contour_details:
        if len(shape) == 4 and area > 1500 and meta_data[0] == -1 and meta_data[1] == -1 and meta_data[3] != -1:
            shape = shape.reshape(-1, 2)
            contours_points.append(shape)

    return contours_points


def calculate_homography_wrap(corners, image):
    tag_dest = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype="float32")
    homography_matrix = H_matrix.homography(corners[0], tag_dest)
    warp1 = cv2.warpPerspective(image.copy(), homography_matrix, (100, 100))
    warp1_blur = cv2.GaussianBlur(warp1, (3, 3), cv2.BORDER_DEFAULT)
    tag_img_resized = cv2.resize(warp1_blur, dsize=None, fx=0.08, fy=0.08)
    return tag_img_resized, homography_matrix

if __name__ == "__main__":
    frames, size = operation()
    output.create_video(frames, size)
