import os
import cv2
import math
import json
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from analysis_tools import *
import numpy as np

def main():
    images = {}
    for image_file in os.listdir('./images'):
        img = cv2.imread('images/' + image_file)
        #img = cv2.copyMakeBorder(img, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=[255, 255, 255])

        contour = detect_contour(img)

        # analysis
        circularity = calculate_circularity(contour)

        images[image_file] = circularity

    print(images)
    with open('./output.txt', 'w') as f:
        f.write(json.dumps(images))

def contour_analysis(contour):
    centre, furthest, closest = get_features(contour)

    standard_deviation = radius_statistics(contour, centre)

    return (furthest, centre, closest, standard_deviation)

def detect_contour(img):
    # pre-process
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, threshold = cv2.threshold(gray, 95, 255, cv2.THRESH_BINARY)

    # canny business
    #denoised = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    #edges = cv2.Canny(denoised, 40, 80)
    #plot_edges(edges)

    # detect the contours
    contours, heirarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

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

def plot_contour(img, contour):
    colour = (127, 127, 127)
    img = cv2.drawContours(img, contour, -1, colour, 3)
    plt.imshow(img)
    plt.show()

def plot_edges(edges):
    plt.subplot(122)
    plt.imshow()
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()

if __name__ == '__main__':
    main()
