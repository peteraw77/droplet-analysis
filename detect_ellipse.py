import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from analysis_tools import *

def main():
    img = cv2.imread('ellipse.jpg')
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

    # find the smallest and largest points
    furthest = furthest_points(max_contour)
    centre = [abs(furthest[0][0] + furthest[1][0]) / 2, abs(furthest[0][1] + furthest[1][1]) / 2]
    centre = np.array(centre, dtype=float)

    closest = closest_points(max_contour, centre)

    # find the number of slopes that are perpendicular to raydius
    #deformation_score = perpendicular_slopes(max_contour, centre)

    # get that yung variange
    standard_deviation = radius_statistics(max_contour, centre)

    # plot that boi
    colour = (255, 0, 0)
    img = cv2.drawContours(img, max_contour, -1, colour, 3)
    img = cv2.line(img, closest[0], closest[1], colour)
    img = cv2.line(img, furthest[0], furthest[1], colour)
    #cv2.imshow('img', img)
    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    main()
