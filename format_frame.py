
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
import colors as color
import math_functions as m


def crop(img, start_pos, final_pos):
    return img[start_pos[0]:start_pos[1], final_pos[0]:final_pos[1]]


def bgr_to_gray(bgr_frame):
    return cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)


def apply_roi(frame, resize_ratio):

    def r(number):
        return int(number*resize_ratio)

    pts = np.array([[r(1920), r(790)], [r(1290), 0], [r(1920), 0]], np.int32)
    cv2.fillPoly(frame, [pts], color.BLACK)
    # triângulo lado esquerdo
    pts3 = np.array([[0, r(620)], [r(270), 0], [0, 0]], np.int32)
    cv2.fillPoly(frame, [pts3], color.BLACK)
    # Line between lane 1 and 2
    pts1 = np.array([[r(480), r(1080)], [r(560), r(0)],
                     [r(640), r(0)], [r(570), r(1080)]], np.int32)
    cv2.fillPoly(frame, [pts1], color.BLACK)
    # Line between lane 3 and 3
    pts7 = np.array([[r(1310), r(1080)], [r(900), r(0)],
                     [r(990), r(0)], [r(1410), r(1080)]], np.int32)
    cv2.fillPoly(frame, [pts7], color.BLACK)
    return frame


def apply_CLAHE(gray_frame, clipLimit=3.0, tileGridSize=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit, tileGridSize)
    hist = clahe.apply(gray_frame)
    return hist


def apply_perpective(frame, lane, resize_ratio, output_size=(640, 1080)):

    def r(number):
        return int(number*resize_ratio)

    width = r(output_size[0])
    height = r(output_size[1])

    if lane == 1:
        points = np.array([[[r(-150), r(1080)], [r(480), r(1080)],
                            [r(560), r(0)], [r(270), 0]]], np.int32)
        pt4 = [r(35), 0]
        pt3 = [r(610), 0]
        pt2 = [r(640), r(1080)]
        pt1 = [0, r(1080)]

        target_pts = np.array([pt1, pt2, pt3, pt4], np.float32)
        H, mask_crop = cv2.findHomography(points, target_pts, cv2.RANSAC)
        warped_frame = cv2.warpPerspective(frame, H, (width, height))
        return warped_frame

    elif lane == 2:
        points = np.array([[[r(570), r(1080)],  [r(1310), r(1080)],
                            [r(900), r(0)], [r(640), r(0)]]], np.int32)
        pt4 = [r(50), 0]
        pt3 = [r(570), 0]
        pt2 = [r(640), r(1080)]
        pt1 = [0, r(1080)]

        target_pts = np.array([pt1, pt2, pt3, pt4], np.float32)
        H, mask_crop = cv2.findHomography(points, target_pts, cv2.RANSAC)
        warped_frame = cv2.warpPerspective(frame, H, (width, height))
        return warped_frame

    elif lane == 3:
        points = np.array([[[r(1410), r(1080)], [r(2170), r(1080)],
                            [r(1320), r(0)], [r(990), r(0)]]], np.int32)
        pt4 = [r(15), 0]
        pt3 = [r(670), 0]
        pt2 = [r(640), r(1080)]
        pt1 = [0, r(1080)]

        target_pts = np.array([pt1, pt2, pt3, pt4], np.float32)
        H, mask_crop = cv2.findHomography(points, target_pts, cv2.RANSAC)
        warped_frame = cv2.warpPerspective(frame, H, (width, height))
        return warped_frame
    return


def apply_erode(mask, kernel_erode, iterations=1):
    return cv2.erode(mask, kernel_erode, iterations)


def apply_dilate(mask, kernel_dilate, iterations=1):
    return cv2.dilate(mask, kernel_dilate, iterations)

# GET FRAME INFO


def apply_convexHull(contours):
    hull = []
    for i in range(len(contours)):
        hull.append(cv2.convexHull(contours[i], False))
    return hull


def find_contours(mask):
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return (contours, hierarchy)

# Create Images


def create_empty_image(frame):  # create an empty black image
    return np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)


def crop(img, start_pos, final_pos):
    return img[start_pos[0]:start_pos[1], final_pos[0]:final_pos[1]]


if __name__ == '__main__':
    print('arquivo format_frame executado. \n Esse arquivo possui apenas funcoes')
