from format_frame import *
import numpy as np
import cv2
import format_frame as f


class FormattedFrame():

    def __init__(self, frame, kernel_erode, kernel_dilate, resize_ratio):
        self.frame = frame
        self.kernel_erode = kernel_erode
        self.kernel_dilate = kernel_dilate
        self.contours = []
        self.hull = []
        # self.foreground_mask = bgs_MOG.apply(self.frame, None, 0.01)
        self.eroded_mask = 0
        self.dilated_mask = 0

    def apply_erode(self, frame=None, kernel_erode=None, iterations=1):
        if frame is None:
            frame = self.frame
        if kernel_erode is None:
            kernel_erode = self.kernel_erode
        self.eroded_mask = cv2.erode(frame, kernel_erode, iterations)
        return self.eroded_mask

    def apply_dilate(self, frame=None, kernel_dilate=None, iterations=1):
        if frame is None:
            frame = self.eroded_mask
        if kernel_dilate is None:
            kernel_dilate = self.kernel_dilate

        self.dilated_mask = cv2.dilate(frame, kernel_dilate, iterations)
        return self.dilated_mask

    def apply_perspective(self, frame):
        self.frame = f.apply_perpective(frame, 3, self.resize_ratio)
        return self.frame

    def find_contours(self, frame=None):
        if frame is None:
            frame = self.dilated_mask

        self.contours, hierarchy = cv2.findContours(frame, cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)
        return self.contours

    def apply_convexHull(self, contours=None):
        if contours is None:
            contours = self.contours

        for i in range(len(contours)):
            self.hull.append(cv2.convexHull(self.contours[i], False))
        return self.hull
    #     return self.frame

        # self.resize_ratio = resize_ratio
        # self.bottom_limit_track = 910
        # self.upper_limit_track = 395
        # self.correction_factor = 2.10
        # self.kernel_erode = np.ones((self.r(12), self.r(12)), np.uint8)
        # self.kernel_dilate = np.ones((self.r(120), self.r(400)), np.uint8)
        # self.lane_info = {}
        # self.tracked_blobs = []
        # self.results = {}

    def r(self, num):
        return int(num*self.resize_ratio)


if __name__ == '__main__':
    print(FormattedFrame().teste())
