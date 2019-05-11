import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from analysis_tools import *
import numpy as np

def main():
    ellipse_img = cv2.imread('ellipse.jpg')
    ellipse_img = cv2.copyMakeBorder(ellipse_img, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    ellipse = detect_contour(ellipse_img)

    circle_img = cv2.imread('circle.png')
    circle_img = cv2.copyMakeBorder(circle_img, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    circle = detect_contour(circle_img)

    bad_circle_img = cv2.imread('bad_circle.jpg')
    bad_circle_img = cv2.copyMakeBorder(bad_circle_img, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    bad_circle = detect_contour(bad_circle_img)

    # analysis
    ellipse_furthest, ellipse_centre, ellipse_closest, ellipse_std = contour_analysis(ellipse)
    #print(ellipse_std)

    circle_furthest, circle_centre, circle_closest, circle_std = contour_analysis(circle)
    #print(circle_std)

    bad_circle_furthest, bad_circle_centre, bad_circle_closest, bad_circle_std = contour_analysis(bad_circle)
    #print(bad_circle_std)

    ellipse_circularity = calculate_circularity(ellipse)
    circle_circularity = calculate_circularity(circle)
    bad_circle_circularity = calculate_circularity(bad_circle)

    print(ellipse_circularity)
    print(circle_circularity)
    print(bad_circle_circularity)

    # plot em up
    plot_contour(circle_img, circle, circle_closest, circle_furthest)
    plot_contour(bad_circle_img, bad_circle, bad_circle_closest, bad_circle_furthest)
    plot_contour(ellipse_img, ellipse, ellipse_closest, ellipse_furthest)

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

    return max_contour

def plot_contour(img, contour, closest, furthest):
    colour = (255, 0, 0)
    img = cv2.drawContours(img, contour, -1, colour, 3)
    img = cv2.line(img, closest[0], closest[1], colour)
    img = cv2.line(img, furthest[0], furthest[1], colour)
    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    main()
