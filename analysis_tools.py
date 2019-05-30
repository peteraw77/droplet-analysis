import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# raise divide by zero errors
np.seterr(divide='raise')

def calculate_roundness(contour, centre):
    # lazy approach, assuming all points have equal angle
    angle = 2 * math.pi / len(contour)
    R = 0
    a = 0
    b = 0

    sin = math.sin(angle)
    cos = math.cos(angle)
    print('sin: ' + str(sin))
    print('cos: ' + str(cos))
    for point in contour:
        r = contour_point_distance(point[0], centre)
        R = R + r
        a = a + r * cos
        b = b + r * sin

    R = R / len(contour)
    a = a * 2 / len(contour)
    b = b * 2 / len(contour)
    print('R: ' + str(R))
    print('a: ' + str(a))
    print('b: ' + str(b))

    deviation = 0
    for point in contour:
        r = contour_point_distance(point[0], centre)
        deviation = deviation + r - R - a * cos - b * sin

    return deviation / len(contour)

def calculate_circularity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    return math.pow(perimeter, 2) / (4 * math.pi * area)

def contour_point_distance(point1, point2):
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

def closest_points(contour, centre):
    # this is the succ
    closest = []
    distance = 0
    min_distance = 9999999999

    # approximate the centre of the ellipse
    for i in range(len(contour)):
        point = contour[i]
        for pair in contour[i+1:]:
            # check if the line is "straight"
            point_vector = centre - point[0]
            pair_vector = pair[0] - centre

            angle1 = math.atan2(point_vector[1], point_vector[0])
            angle2 = math.atan2(pair_vector[1], pair_vector[0])

            #if point_length == pair_length and angle == math.pi:
            if angle1 == angle2:
                distance = contour_point_distance(point[0], pair[0])

                if distance < min_distance and distance > 0:
                    closest = [(point[0][0], point[0][1]), (pair[0][0], pair[0][1])]
                    min_distance = distance

    return closest

def get_features(contour):
    furthest = furthest_points(contour)

    # don't think we would need to interpret the centre as centre of mass
    centre = get_centroid(contour)

    closest = closest_points(contour, centre)

    return (centre, furthest, closest)

def get_centroid(contour):
    centroid = [0, 0]
    for point in contour:
        centroid[0] = centroid[0] + point[0][0]
        centroid[1] = centroid[1] + point[0][1]

    return [point / len(contour) for point in centroid]

def furthest_points(contour):
    # optimize this later
    furthest = []
    distance = 0
    max_distance = 0

    for i in range(len(contour)):
        point = contour[i]
        for pair in contour[i+1:]:
            distance = contour_point_distance(point[0], pair[0])
            if distance > max_distance:
                furthest = [(point[0][0], point[0][1]), (pair[0][0], pair[0][1])]
                max_distance = distance

    return furthest

def radius_statistics(contour, centre):
    # compute the radii
    radii = [contour_point_distance(point[0], centre) for point in contour]
    radii = pd.DataFrame(radii)

    # get the standard deviation
    standard_deviation = radii.std()
    return standard_deviation

def radius_statistics2(contour, centre):
    # compute the radii
    radii = [contour_point_distance(point[0], centre) for point in contour]

    # compute the absolute deltas
    avgDelta = 0
    for i in range(1, len(radii)):
        avgDelta = avgDelta + abs(radii[i] - radii[i - 1])

    return avgDelta / (len(radii) - 1)
