import math
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# raise divide by zero errors
np.seterr(divide='raise')

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

            #distance1 = contour_point_distance(point[0], centre)
            #distance2 = contour_point_distance(centre, pair[0])
            #total_distance = contour_point_distance(point[0], pair[0])

            #if point_length == pair_length and angle == math.pi:
            if angle1 == angle2:
            #if distance1 + distance2 == total_distance:
                distance = contour_point_distance(point[0], pair[0])

                if distance < min_distance and distance > 0:
                    closest = [(point[0][0], point[0][1]), (pair[0][0], pair[0][1])]
                    min_distance = distance

    return closest

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
