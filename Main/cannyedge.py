import numpy as np
from scipy import ndimage
import cv2
import imageio
import matplotlib.pyplot as plt


def hysteresis(suppression):
    l, h = 0.10, 0.30
    minimum = np.min(suppression)
    maximum = np.max(suppression)
    #minimum = 190
    #maximum = 255
    diff = maximum - minimum
    low_thresh = minimum + l * diff
    high_thresh = minimum + h * diff
    dims = suppression.shape
    backup = np.zeros(dims)
    for row in range(dims[0] - 1):
        for col in range(dims[1] - 1):
            if suppression[row, col] > high_thresh:
                backup[row, col] = 2
            elif low_thresh <= suppression[row, col] <= high_thresh:
                backup[row, col] = 1
    count = np.sum(backup == 2)
    while True:
        for row in range(dims[0]):
            for col in range(dims[1]):
                if backup[row, col] == 1:
                    #check = max(backup[row - 1, col - 1], backup[row - 1, col], backup[row - 1, col + 1],
                                #backup[row, col - 1], backup[row, col + 1], backup[row + 1, col - 1],
                                #backup[row + 1, col], backup[row + 1, col + 1])
                    minrow = max(row-2, 0)
                    maxrow = min(row+2, dims[0])
                    mincol = max(col-2, 0)
                    maxcol = min(col+2, dims[1])
                    check = backup[minrow:maxrow, mincol:maxcol]
                    check = np.max(check)
                    if check == 2:
                        backup[row, col] = 2
        if count == np.sum(backup == 2):
            break
        count = np.sum(backup == 2)

    for row in range(dims[0] - 1):
        for col in range(dims[1] - 1):
            if backup[row, col] == 1:
                backup[row, col] = 0
            elif backup[row, col] == 2:
                backup[row, col] = 255
    return backup


def suppression_and_interpolation(direct, magnitude):
    dims = magnitude.shape

    suppression = np.zeros(dims)
    for row in range(dims[0] - 1):
        for col in range(dims[1] - 1):
            if (-22.5 < direct[row, col] <= 22.5) or (-157.5 >= direct[row, col] > 157.5):
                if ((magnitude[row, col] > magnitude[row, col + 1]) and (
                        magnitude[row, col] > magnitude[row, col - 1])):
                    suppression[row, col] = magnitude[row, col]
                else:
                    suppression[row, col] = 0
            elif (22.5 < direct[row, col] <= 67.5) or (-112.5 >= direct[row, col] > -157.5):
                if ((magnitude[row, col] > magnitude[row + 1, col + 1]) and (
                        magnitude[row, col] > magnitude[row - 1, col - 1])):
                    suppression[row, col] = magnitude[row, col]
                else:
                    suppression[row, col] = 0
            elif (67.5 < direct[row, col] <= 112.5) or (-67.5 >= direct[row, col] > -112.5):
                if ((magnitude[row, col] > magnitude[row + 1, col]) and (
                        magnitude[row, col] > magnitude[row - 1, col])):
                    suppression[row, col] = magnitude[row, col]
                else:
                    suppression[row, col] = 0
            else:
                if ((magnitude[row, col] > magnitude[row + 1, col - 1]) and (
                        magnitude[row, col] > magnitude[row - 1, col + 1])):
                    suppression[row, col] = magnitude[row, col]
                else:
                    suppression[row, col] = 0
    return suppression / np.max(suppression)


def make_gradient(gaussian):
    gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    fx = ndimage.convolve(gaussian, gx)
    fy = ndimage.convolve(gaussian, gy)
    fx = fx / np.max(fx)
    fy = fy / np.max(fy)
    mag = np.hypot(fx, fy)
    mag = mag / np.max(mag)

    return fx, fy, mag


def canny_edge_detection(input_image):
    # Convert image to grayscale
    #grayscale = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    grayscale = np.dot(input_image, [0.2989, 0.5870, 0.1140])
    # create Gaussian blur from grayscale image
    gaussian = cv2.GaussianBlur(grayscale, (0, 0), cv2.BORDER_DEFAULT)
    #gaussian = ndimage.gaussian_filter(grayscale, sigma=1.0)
    # Find the gradients in x and y
    fx, fy, magnitude = make_gradient(gaussian)
    gradient_direction = np.degrees(np.arctan2(fy, fx))
    suppression = suppression_and_interpolation(gradient_direction, magnitude)

    result = hysteresis(suppression)
    #plt.imshow(result, cmap=plt.get_cmap('gray'))
    #plt.show()
    #imageio.imwrite('result/'+vehicle+'/'+name+'.jpeg', result, cmap='gray')

    return result

def name_and_file(name, vehicle):
    input_image = cv2.imread('data/'+vehicle+'/'+name)
    name = name.split(".")[0]
    canny_edge_detection(name, vehicle, input_image)

#if __name__ == "__main__":
#    input_img = cv2.imread('Square0.jpg')
#    canny_edge_detection('Square0.jpeg', 'vehicles', input_img)
