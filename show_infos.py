 
import numpy as np
import itertools as it
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import cv2
import time
import uuid
import numpy as np
import os
import cv2
import datetime
from shutil import copy2
import tccfunctions as t
import format_frame as f
import math_functions as m
import colors as color

def pairwise(iterable):
    # r"s -> (s0, s1), (s1, s2), (s2, s3), ..."
    a, b = it.tee(iterable)
    next(b, None)
    return zip(a, b)

def print_contours(parameters, frame, contours, color=color.BLUE):
    if parameters.get('SHOW_CONTOURS'):
        cv2.drawContours(frame, contours, -1, color, 2, 8)
    return

def print_trail(trail, frame):
    for (a, b) in pairwise(trail):
        cv2.line(frame, a, b, color.GREEN, 3)
        cv2.circle(frame, a, 5, color.RED, -1)
    return


def print_roi(parameters, frame, RESIZE_RATIO):
    if parameters.get('SHOW_ROI'):
      f.apply_roi(frame, RESIZE_RATIO)
    return


def print_tracking_area(parameters, frame, frame_width, upper_limit, bottom_limit, line_color=(255, 255, 255)):
    if parameters.get('SHOW_TRACKING_AREA'): 
        cv2.line(frame, (0, upper_limit), (frame_width, upper_limit), line_color, 2)
        cv2.line(frame, (0, bottom_limit), (frame_width, bottom_limit), line_color, 2)
    return


def print_real_speeds(parameters, frame, ratio, dict_lane1, dict_lane2, dict_lane3):

    def print_speed(lane, position):
        cv2.rectangle(frame, (position[0] - 10, position[1] - 20), (position[0] + 135, position[1] + 10), color.BLACK, -1)
        cv2.putText(frame, f"speed: {lane.get('speed')}", position, 2, .6, color.CIAN, 1)
        return

    lane1_pos = (m.resize(143,ratio), m.resize(43,ratio))
    lane2_pos = (m.resize(628,ratio), m.resize(43,ratio))
    lane3_pos = (m.resize(1143,ratio), m.resize(43,ratio))
    if parameters.get('SHOW_REAL_SPEEDS'):
        if dict_lane1.get('speed'):  
            print_speed(dict_lane1, lane1_pos)

        if dict_lane2.get('speed'):  
            print_speed(dict_lane2, lane2_pos)

        if dict_lane3.get('speed'):
            print_speed(dict_lane3, lane3_pos)
    return



if __name__ == '__main__':
    print('arquivo show_infos executado. \n Esse arquivo possui apenas funcoes')