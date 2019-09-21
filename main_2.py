''' -*- coding: utf-8 -*-'''
import time
import uuid
import numpy as np
import os
import cv2
import tccfunctions as t
import show_infos as s
import format_frame as f
import datetime
from shutil import copy2
import math_functions as m
import xml_functions as x
from util import *

from sys import exit
#######  CONSTANT VALUES ###################################################
VIDEO = 1
VIDEO_FILE = './Dataset/video{}.avi'.format(VIDEO)
XML_FILE = './Dataset/video{}.xml'.format(VIDEO)

RESIZE_RATIO = .22222  # 0.7697  720p=.6667 480p=.4445 360p=.33333 240p=.22222 144p=.13333
if RESIZE_RATIO > 1:
    exit('ERRO: AJUSTE O RESIZE_RATIO')
CLOSE_VIDEO = 770  # 2950 #5934  # 1-6917 # 5-36253
ARTG_FRAME = 0  # 254  # Frame q usei para exemplo no Artigo

SHOW_PARAMETERS = {
    'SHOW_ROI': True,
    'SHOW_TRACKING_AREA': True,
    'SHOW_TRAIL': True,
    'SHOW_REAL_SPEEDS': True,
    'SHOW_CONTOURS': True,
    'SHOW_CAR_RECTANGLE': True
}
SHOW_CAR_RECTANGLE = True

SHOW_FRAME_COUNT = True

SKIP_VIDEO = True
SEE_CUTTED_VIDEO = False  # ver partes retiradas, precisa de SKIP_VIDEO = True
# ---- Tracking Values --------------------------------------------------------
# The maximum distance a blob centroid is allowed to move in order to
# consider it a match to a previous scene's blob.
BLOB_LOCKON_DIST_PX_MAX = 150  # default = 50 p/ ratio 0.35
BLOB_LOCKON_DIST_PX_MIN = 5  # default 5
MIN_AREA_FOR_DETEC = 30000  # Default 40000
# Limites da Área de Medição, área onde é feita o Tracking
# Distancia de medição: default 915-430 = 485

# Faixa 1
BOTTOM_LIMIT_TRACK = 910  # 850  # Default 900
UPPER_LIMIT_TRACK = 395  # 350  # Default 430
# Faixa 2
BOTTOM_LIMIT_TRACK_L2 = 940  # 1095  # Default 940
UPPER_LIMIT_TRACK_L2 = 425  # 408 # Default 420
# Faixa 3
BOTTOM_LIMIT_TRACK = 930  # 1095  # Default 915
UPPER_LIMIT_TRACK = 430  # 408 # Default 430

MIN_CENTRAL_POINTS = 10  # qnt mínima de pontos centrais para calcular a velocidade
# The number of seconds a blob is allowed to sit around without having
# any new blobs matching it.
BLOB_TRACK_TIMEOUT = 0.1  # Default 0.7
# ---- Speed Values -----------------------------------------------------------
CF_LANE1 = 2.10  # 2.10  # default 2.5869977 # Correction Factor
CF_LANE2 = 2.32  # default 2.32    3.758897
CF_LANE3 = 2.3048378  # default 2.304837879578
# ----  Save Results Values ---------------------------------------------------
SAVE_RESULTS = False  # Salva os Gráficos
SAVE_FRAME_F1 = False  # Faixa 1
SAVE_FRAME_F2 = False  # Faixa 2
SAVE_FRAME_F3 = False  # Faixa 3
# ####### END - CONSTANT VALUES ###############################################
cap = cv2.VideoCapture(VIDEO_FILE)
#FPS = cap.get(cv2.CAP_PROP_FPS)
FPS = 30.15
WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Retorna a largura do video
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Retorna a largura do video

bgs_MOG = cv2.createBackgroundSubtractorMOG2(
    history=10, varThreshold=50, detectShadows=0)

# Variant Values
dict_lane1 = {}  # Armazena os valores de "speed, frame_start, frame_end" da FAIXA 1
dict_lane2 = {}  # Armazena os valores de "speed, frame_start, frame_end" da FAIXA 2
dict_lane3 = {}  # Armazena os valores de "speed, frame_start, frame_end" da FAIXA 3
tracked_blobs = []  # Lista que salva os dicionários dos tracked_blobs
tracked_blobs_lane2 = []  # Lista que salva os dicionários dos tracked_blobs
tracked_blobs_lane3 = []

prev_len_speed = []
prev_speed = 1.0

frameCount = 0  # Armazena a contagem de frames processados do video
out = 0  # Armazena o frame com os contornos desenhados
#final_ave_speed = 0
ave_speed = 0

results_lane1 = {}
results_lane2 = {}
results_lane3 = {}

area = []

process_times = []

# ##############  FUNÇÕES #####################################################


def r(numero):
    return int(numero*RESIZE_RATIO)


# ########## FIM  FUNÇÕES #####################################################
now = datetime.datetime.now()
DATE = f'video{VIDEO}_{now.day}-{now.month}-{now.year}_{now.hour}-{now.minute}-{now.second}'

if not os.path.exists(f"results/{DATE}"):
    os.makedirs(f"results/{DATE}/graficos/pdfs")
    os.makedirs(f"results/{DATE}/planilhas/")
    os.makedirs(f"results/{DATE}/imagens/faixa1")
    os.makedirs(f"results/{DATE}/imagens/faixa2")
    os.makedirs(f"results/{DATE}/imagens/faixa3")

# Dicionário que armazena todas as informações do xml
vehicle = t.read_xml(XML_FILE, VIDEO, DATE)

KERNEL_ERODE = np.ones((r(12), r(12)), np.uint8)
KERNEL_DILATE = np.ones((r(100), r(320)), np.uint8)

while True:
    ret, frame = t.get_frame(cap, RESIZE_RATIO)
    frame_time = time.time()

    if SKIP_VIDEO:
        skip = t.skip_frames(frameCount, VIDEO, frame)
        if SEE_CUTTED_VIDEO and not skip:
            frameCount += 1
            if frameCount == CLOSE_VIDEO:  # fecha o video
                break
            continue
        else:
            if skip:
                frameCount += 1
                if frameCount == CLOSE_VIDEO:  # fecha o video
                    break
                continue
    start_frame_time = time.time()

    s.print_roi(SHOW_PARAMETERS, frame, RESIZE_RATIO)
    s.print_tracking_area(SHOW_PARAMETERS, frame, WIDTH, r(
        UPPER_LIMIT_TRACK), r(BOTTOM_LIMIT_TRACK))

    # Image manipulation
    gray_frame = f.bgr_to_gray(frame)
    f.apply_roi(gray_frame, RESIZE_RATIO)

    hist_equalization = f.apply_CLAHE(gray_frame)  # Histogram Equalization

    lane3_perspective = f.apply_perpective(hist_equalization, 3, RESIZE_RATIO)

    if ret:
        t.update_info_xml(frameCount, vehicle, dict_lane1,
                          dict_lane2, dict_lane3)
        s.print_real_speeds(SHOW_PARAMETERS, frame,
                            RESIZE_RATIO, dict_lane1, dict_lane2, dict_lane3)

        lane3 = FormattedFrame(
            lane3_perspective, KERNEL_ERODE, KERNEL_DILATE, RESIZE_RATIO)

        fgmask = bgs_MOG.apply(lane3.frame, None, 0.01)
        lane3.apply_erode(fgmask)
        lane3.apply_dilate()
        contours = lane3.find_contours()
        hull = lane3.apply_convexHull()

        s.print_contours(SHOW_PARAMETERS, lane3.frame, contours)

        drawing = f.create_empty_image(lane3.frame)

        # draw contours and hull points
        for i in range(len(contours)):
            if cv2.contourArea(contours[i]) > r(MIN_AREA_FOR_DETEC):
                # draw ith convex hull object
                out = cv2.drawContours(
                    drawing, hull, i, t.WHITE, -1, 8)

                (x, y, w, h) = cv2.boundingRect(hull[i])
                center = (int(x + w/2), int(y + h/2))

                if w < r(340) and h < r(340):  # ponto que da pra mudar
                    continue
                # Área de medição do Tracking
                if center[1] > r(BOTTOM_LIMIT_TRACK) or center[1] < r(UPPER_LIMIT_TRACK):
                    continue

                if center[1] > r(UPPER_LIMIT_TRACK):
                    PADDING = r(1270)
                    s.print_vehicle_rectangle(
                        SHOW_PARAMETERS, frame, (x+PADDING, y), (w, h))
                    s.print_vehicle_rectangle(
                        SHOW_PARAMETERS, lane3.frame, (x, y), (w, h))

                # ################## TRACKING #################################
                # Look for existing blobs that match this one
                closest_blob = None
                if tracked_blobs_lane3:
                    # Sort the blobs we have seen in previous frames by pixel distance from this one
                    closest_blobs = sorted(
                        tracked_blobs_lane3, key=lambda b3: cv2.norm(b3['trail'][0], center))

                    # Starting from the closest blob, make sure the blob in question is in the expected direction
                    distance = 0.0
                    for close_blob in closest_blobs:
                        distance = cv2.norm(
                            center, close_blob['trail'][0])

                        # Check if the distance is close enough to "lock on"
                        if distance < r(BLOB_LOCKON_DIST_PX_MAX) and distance > r(BLOB_LOCKON_DIST_PX_MIN):
                            closest_blob = close_blob
                        #    continue # retirar depois
                            # If it's close enough, make sure the blob was moving in the expected direction
                            # verifica se esta na dir up
                            if close_blob['trail'][0][1] < center[1]:
                                continue
                            else:
                                closest_blob = close_blob
                                continue  # defalut break

                    if closest_blob:
                        # If we found a blob to attach this blob to, we should
                        # do some math to help us with speed detection
                        prev_center = closest_blob['trail'][0]
                        if center[1] < prev_center[1]:  # It's moving up
                            closest_blob['trail'].insert(
                                0, center)  # Add point
                            closest_blob['last_seen'] = frame_time

                            if len(closest_blob['trail']) > MIN_CENTRAL_POINTS:
                                cf = CF_LANE3
                                closest_blob['speed'].insert(0, m.calculate_speed(
                                    closest_blob['trail'], FPS, RESIZE_RATIO, CF_LANE3))
                                lane = 3
                                ave_speed = np.mean(closest_blob['speed'])
                                abs_error, per_error = t.write_results_on_image(frame, frameCount, ave_speed, lane, closest_blob['id'], RESIZE_RATIO, VIDEO,
                                                                                dict_lane1, dict_lane2, dict_lane3)

                                try:
                                    results_lane3[str(closest_blob['id'])] = dict(ave_speed=round(ave_speed, 2),
                                                                                     speeds=closest_blob['speed'],
                                                                                     frame=frameCount,
                                                                                     real_speed=float(
                                                                                         dict_lane3['speed']),
                                                                                     abs_error=round(
                                                                                         abs_error, 2),
                                                                                     per_error=round(
                                                                                         per_error, 3),
                                                                                     trail=closest_blob['trail'],
                                                                                     car_id=closest_blob['id'])
                                    abs_error = []
                                    per_error = []

                                    if SAVE_FRAME_F3:
                                        cv2.imwrite('results/{}/imagens/faixa3/{}_{}_F{}_{}.png'.format(
                                            DATE, VIDEO, dict_lane3['frame_start'], lane, closest_blob['id']), frame)
                                except:
                                    pass

                if not closest_blob:  # Cria as variaves
                    # If we didn't find a blob, let's make a new one and add it to the list
                    b3 = dict(id=str(uuid.uuid4())[:8], first_seen=frame_time,
                              last_seen=frame_time, trail=[
                                  center], speed=[0],
                              size=[0, 0],)
                    # Agora tracked_blobs não será False
                    tracked_blobs_lane3.append(b3)
                # ################# END TRACKING ##############################
                # ################# END FAIXA 3  ##############################
                # #############################################################

        if tracked_blobs_lane3:
            # Prune out the blobs that haven't been seen in some amount of time
            for i in range(len(tracked_blobs_lane3) - 1, -1, -1):
                # Deleta caso de timeout
                if frame_time - tracked_blobs_lane3[i]['last_seen'] > BLOB_TRACK_TIMEOUT:
                    print("Removing expired track {}".format(
                        tracked_blobs_lane3[i]['id']))
                    del tracked_blobs_lane3[i]

        for blob3 in tracked_blobs_lane3:  # Desenha os pontos centrais
            if SHOW_PARAMETERS['SHOW_TRAIL']:
                s.print_trail(blob3['trail'], lane3.frame)

            if blob3['speed'] and blob3['speed'][0] != 0:
                prev_len_speed.insert(0, len(blob3['speed']))
                # limpa prev_len_speed se estiver muito grande
                # deixa no máx 20 valores
                if len(prev_len_speed) > 20:
                    while len(prev_len_speed) > 20:
                        del prev_len_speed[19]
                # remove zero elements on the speed list
                blob3['speed'] = [item for item in blob3['speed'] if item != 0.0]
                print('========= speed list =========', blob3['speed'])
                prev_speed = ave_speed
                ave_speed = np.mean(blob3['speed'])
                print('========= prev_speed =========',
                      float("{0:.5f}".format(prev_speed)))
                print('========= ave_speed ==========',
                      float("{0:.5f}".format(ave_speed)))

                # ############### FIM PRINTA OS BLOBS  ########################

        print('*************************************************')

        if SHOW_FRAME_COUNT:
            PERCE = str(int((100*frameCount)/vehicle['videoframes']))
            cv2.putText(frame, f'frame: {frameCount} {PERCE}%', (r(
                14), r(1071)), 0, .65, t.WHITE, 2)
        # ########## MOSTRA OS VIDEOS  ########################################

    #    cv2.imshow('fgmask', fgmask)
    #    cv2.imshow('erodedmask_mask',erodedmask_mask)
    #    cv2.imshow('dilated_mask', dilated_mask)

    #    cv2.imshow('out',out)

    #    cv2.imshow('lane3.frame', lane3.frame)
        cv2.imshow('frame', frame)

        frameCount += 1    # Conta a quantidade de Frames

        end_frame_time = time.time()
    #    process_times.append(process_end - process_start)
        process_times.append(end_frame_time - start_frame_time)

        if frameCount == CLOSE_VIDEO:  # fecha o video
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Tecla Q para fechar
            break
    else:  # sai do while: ret == False
        break


cap.release()
cv2.destroyAllWindows()
