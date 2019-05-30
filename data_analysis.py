import os
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from analysis_tools import *
import numpy as np

def main():
    images = {}
    for image_file in os.listdir('./images'):
        img = cv2.imread('images/' + image_file)
        img = cv2.copyMakeBorder(img, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        contour = detect_contour(img)
        edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 80, 160)

        plt.subplot(122)
        plt.imshow(edges,cmap = 'gray')
        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        plt.show()

        # analysis
        circularity = calculate_circularity(contour)

        images[image_file] = circularity

    print(images)

def pad_img(img):
    whitespace = [[255, 255, 255], [255, 255, 255], [255, 255, 255]]
    new_img = [[whitespace for j in range(math.floor(len(img[0]) * 1.1))]
            for i in range(math.floor(len(img) * 1.1))]
    new_img = np.array(new_img)

    for i in range(math.floor(len(img) * 1.1)):
        for j in range(len(img[0] * 1.1)):
            shift = i - math.floor(len(img) * 0.05)
            if shift < 0 or shift > len(img) - 1:
                new_img[i][j] = whitespace
            else:
                shjft = j - math.floor(len(img[0]) * 0.05)
                if shjft < 0 or shjft > len(img[0]) - 1:
                    new_img[i][j] = whitespace
                else:
                    new_img[i][j] = img[shift][shjft]

    return new_img

def contour_analysis(contour):
    centre, furthest, closest = get_features(contour)

    standard_deviation = radius_statistics(contour, centre)
    #standard_deviation = radius_statistics2(contour, centre)

    return (furthest, centre, closest, standard_deviation)

def detect_contour(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect the contours
    ret, threshold = cv2.threshold(gray, 127, 255, 0)
    contours, heirarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #contours, heirarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # find the contour with the 2nd largest area
    max_contour = []
    second_max_contour = []
    max_area = 0
    second_max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if (area > second_max_area):
            if (area > max_area):
                second_max_area = max_area
                second_max_contour = max_contour
                max_area = area
                max_contour = contour
            else:
                second_max_area = area
                second_max_contour = contour
        print("Max area is: " + str(max_area))
        print("Second max area is: " + str(second_max_area))

    return second_max_contour

def plot_contour(img, contour):
    colour = (255, 0, 0)
    img = cv2.drawContours(img, contour, -1, colour, 3)
    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    main()
