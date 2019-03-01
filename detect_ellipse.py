import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from analysis_tools import *

def main():
    ellipse_img = cv2.imread('ellipse.jpg')
    ellipse = detect_contour(ellipse_img)

    circle_img = cv2.imread('circle.jpg')
    circle = detect_contour(circle_img)

    bad_circle_img = cv2.imread('bad_circle.jpg')
    bad_circle = detect_contour(bad_circle_img)

    # analysis
    ellipse_furthest, ellipse_centre, ellipse_closest, ellipse_std = contour_analysis(ellipse)
    print(ellipse_std)

    circle_furthest, circle_centre, circle_closest, circle_std = contour_analysis(circle)
    print(circle_std)

    bad_circle_furthest, bad_circle_centre, bad_circle_closest, bad_circle_std = contour_analysis(bad_circle)
    print(bad_circle_std)

    # plot em up
    plot_contour(circle_img, circle, circle_closest, circle_furthest)
    plot_contour(bad_circle_img, bad_circle, bad_circle_closest, bad_circle_furthest)
    plot_contour(ellipse_img, ellipse, ellipse_closest, ellipse_furthest)

def contour_analysis(contour):
    furthest = furthest_points(contour)

    centre = [abs(furthest[0][0] + furthest[1][0]) / 2, abs(furthest[0][1] + furthest[1][1]) / 2]
    centre = np.array(centre, dtype=float)

    closest = closest_points(contour, centre)

    standard_deviation = radius_statistics(contour, centre)

    return (furthest, centre, closest, standard_deviation)

def detect_contour(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect the contours
    ret, threshold = cv2.threshold(gray, 127, 255, 0)
    #contours, heirarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours, heirarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # find the contour with the largest area
    max_contour = []
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if (area > max_area):
            max_area = area
            max_contour = contour

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
