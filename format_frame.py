 
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

def bgr_to_gray(bgr_frame):
    return cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)


def apply_roi(frame, resize_ratio):
    def r(numero):  # Faz o ajuste de escala
        return int(numero*resize_ratio)
    pts = np.array([[r(1920), r(790)], [r(1290), 0], [r(1920), 0]], np.int32)
    cv2.fillPoly(frame, [pts], color.BLACK)
    # triângulo lado esquerdo
    pts3 = np.array([[0, r(620)], [r(270), 0], [0, 0]], np.int32)
    cv2.fillPoly(frame, [pts3], color.BLACK)
    # Linha entre faixas 1 e 2
    pts1 = np.array([[r(480), r(1080)], [r(560), r(0)],
                     [r(640), r(0)], [r(570), r(1080)]], np.int32)
    cv2.fillPoly(frame, [pts1], color.BLACK)
    # Linha entre faixas 2 e 3
    pts7 = np.array([[r(1310), r(1080)], [r(900), r(0)],
                     [r(990), r(0)], [r(1410), r(1080)]], np.int32)
    cv2.fillPoly(frame, [pts7], color.BLACK)
    return frame







if __name__ == '__main__':
    print('arquivo format_frame executado. \n Esse arquivo possui apenas funcoes')