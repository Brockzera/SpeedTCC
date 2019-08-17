''' -*- coding: utf-8 -*-'''
import time
import uuid
import numpy as np
#import matplotlib.pyplot as plt
import os
import cv2
import tccfunctions as t
import show_infos as s
import format_frame as f
import datetime
from shutil import copy2
import math_functions as m

from sys import exit
#import math
#######  CONSTANT VALUES ###################################################
VIDEO = 1
VIDEO_FILE = './Dataset/video{}.avi'.format(VIDEO)
XML_FILE = './Dataset/video{}.xml'.format(VIDEO)

RESIZE_RATIO = .4445 #0.7697  720p=.6667 480p=.4445 360p=.33333 240p=.22222 144p=.13333
if RESIZE_RATIO > 1:
    exit('ERRO: AJUSTE O RESIZE_RATIO')
CLOSE_VIDEO = 770 #2950 #5934  # 1-6917 # 5-36253
ARTG_FRAME = 0  # 254  # Frame q usei para exemplo no Artigo

SHOW_PARAMETERS = {
    'SHOW_ROI' : True,
    'SHOW_TRACKING_AREA' : True,
    'SHOW_TRAIL' : True,
    'SHOW_REAL_SPEEDS' : True,
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
BOTTOM_LIMIT_TRACK = 910  #850  # Default 900
UPPER_LIMIT_TRACK = 395 #350  # Default 430
# Faixa 2
BOTTOM_LIMIT_TRACK_L2 = 940 #1095  # Default 940
UPPER_LIMIT_TRACK_L2 = 425 #408 # Default 420
# Faixa 3
BOTTOM_LIMIT_TRACK_L3 = 930 #1095  # Default 915
UPPER_LIMIT_TRACK_L3 = 430 #408 # Default 430

MIN_CENTRAL_POINTS = 10 # qnt mínima de pontos centrais para calcular a velocidade
# The number of seconds a blob is allowed to sit around without having
# any new blobs matching it.
BLOB_TRACK_TIMEOUT = 0.1  # Default 0.7
# ---- Speed Values -----------------------------------------------------------
CF_LANE1 = 2.10 #2.10  # default 2.5869977 # Correction Factor
CF_LANE2 = 2.32  # default 2.32    3.758897 
CF_LANE3 = 2.3048378 # default 2.304837879578
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

bgsMOG = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=50, detectShadows=0)

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

area_L1 = []
area_L2 = []
area_L3 = []

process_times = []

# ##############  FUNÇÕES #####################################################
def r(numero):
    return int(numero*RESIZE_RATIO)

def crop(img,start_pos, final_pos):
    return img[start_pos[0]:start_pos[1], final_pos[0]:final_pos[1]]

def calculate_speed(trails, fps):
    med_area_meter = 3.9  # metros (Valor estimado)
    med_area_pixel = r(485)
    qntd_frames =  11 #len(trails)  # default 11
    dist_pixel = cv2.norm(trails[0], trails[10])  
    dist_meter = dist_pixel*(med_area_meter/med_area_pixel)
    speed = (dist_meter*3.6*cf)/(qntd_frames*(1/fps))
    return speed

# ########## FIM  FUNÇÕES #####################################################
now = datetime.datetime.now()
DATE = f'video{VIDEO}_{now.day}-{now.month}-{now.year}_{now.hour}-{now.minute}-{now.second}'

if not os.path.exists(f"results/{DATE}"):
    os.makedirs(f"results/{DATE}/graficos/pdfs")
    os.makedirs(f"results/{DATE}/planilhas/")
    os.makedirs(f"results/{DATE}/imagens/faixa1")
    os.makedirs(f"results/{DATE}/imagens/faixa2")
    os.makedirs(f"results/{DATE}/imagens/faixa3")

vehicle = t.read_xml(XML_FILE, VIDEO, DATE)  # Dicionário que armazena todas as informações do xml

KERNEL_ERODE = np.ones((r(12), r(12)), np.uint8)  # Default (r(12), r(12))
KERNEL_DILATE = np.ones((r(120), r(400)), np.uint8)  # Default (r(120), r(400))

KERNEL_ERODE_L2 = np.ones((r(8), r(8)), np.uint8)  # Default (r(8), r(8))  np.ones((r(8), r(8)), np.uint8)
KERNEL_DILATE_L2 = np.ones((r(100), r(400)), np.uint8)  # Default (r(100), r(400))

KERNEL_ERODE_L3 = np.ones((r(12), r(12)), np.uint8)  # Default (r(12), r(12))
KERNEL_DILATE_L3 = np.ones((r(100), r(320)), np.uint8)  # Default (r(100), r(320))

while True:
    ret, frame = t.get_frame(cap, RESIZE_RATIO)
    frame_time = time.time()
    
    if SKIP_VIDEO:
        skip = t.skip_frames(frameCount, VIDEO, frame)
        if SEE_CUTTED_VIDEO:
            if not skip:
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
    s.print_tracking_area(SHOW_PARAMETERS, frame, WIDTH, r(UPPER_LIMIT_TRACK), r(BOTTOM_LIMIT_TRACK))

    # Image manipulation
    gray_frame = f.bgr_to_gray(frame)
    f.apply_roi(gray_frame, RESIZE_RATIO)
    
    hist = f.apply_CLAHE(gray_frame) # Histogram Equalization

    
    frame_lane1 = f.apply_perpective(hist, 1, RESIZE_RATIO)
    frame_lane2 = f.apply_perpective(hist, 2, RESIZE_RATIO)
    frame_lane3 = f.apply_perpective(hist, 3, RESIZE_RATIO)
    
    if ret:
        t.update_info_xml(frameCount, vehicle, dict_lane1, dict_lane2, dict_lane3)
        s.print_real_speeds(SHOW_PARAMETERS,frame, RESIZE_RATIO, dict_lane1, dict_lane2, dict_lane3)
              
        fgmask = bgsMOG.apply(frame_lane1, None, 0.01)
        eroded_mask = f.apply_erode(fgmask, KERNEL_ERODE)
        dilate_dmask = f.apply_dilate(eroded_mask, KERNEL_DILATE)
        contours, hierarchy = f.find_contours(dilate_dmask)
        hull = f.apply_convexHull(contours)

        s.print_contours(SHOW_PARAMETERS,frame, contours)

        drawing = f.create_empty_image(dilate_dmask)
    
        #draw contours and hull points
        for i in range(len(contours)):
            if cv2.contourArea(contours[i]) > r(MIN_AREA_FOR_DETEC):
                out = cv2.drawContours(drawing, hull, i, t.WHITE, -1, 8)
            #    area.append(cv2.contourArea(contours[i]))
            #    areahull.append(cv2.contourArea(hull[i]))
                (x, y, w, h) = cv2.boundingRect(hull[i])
                center = m.get_center_of_rectangle((x, y), ( w, h))
                # CONDIÇÕES PARA CONTINUAR COM TRACKING
            #    if h > r(HEIGHT)*.80 or w > r(WIDTH)*.40:
                #    continue

                if w < r(340) and h < r(340):  # ponto que da pra mudar
                    continue
                # Área de medição do Tracking
                if center[1] > r(BOTTOM_LIMIT_TRACK) or center[1] < r(UPPER_LIMIT_TRACK):
                    continue
                
                
                if center[1] > r(UPPER_LIMIT_TRACK):
                    area_L1.append(w*h)
                    s.print_vehicle_rectangle(SHOW_PARAMETERS, frame, (x, y), (w, h))
                    s.print_vehicle_rectangle(SHOW_PARAMETERS, frame_lane1, (x, y), (w, h))
                    

                # ################## TRACKING #################################
                # Look for existing blobs that match this one
                closest_blob = None
                if tracked_blobs:
                    # Sort the blobs we have seen in previous frames by pixel distance from this one
                    closest_blobs = sorted(tracked_blobs, key=lambda b: cv2.norm(b['trail'][0], center))

                    # Starting from the closest blob, make sure the blob in question is in the expected direction
                    distance = 0.0
                    for close_blob in closest_blobs:
                        distance = cv2.norm(center, close_blob['trail'][0])

                        # Check if the distance is close enough to "lock on"
                        if distance < r(BLOB_LOCKON_DIST_PX_MAX) and distance > r(BLOB_LOCKON_DIST_PX_MIN):
                            closest_blob = close_blob
                            continue # retirar depois
                            # If it's close enough, make sure the blob was moving in the expected direction
                        #    if close_blob['trail'][0][1] < center[1]:  # verifica se esta na dir up
                        #        continue
                        #    else:
                        #        closest_blob = close_blob
                        #        continue  # defalut break

                    if closest_blob:
                        # If we found a blob to attach this blob to, we should
                        # do some math to help us with speed detection
                        prev_center = closest_blob['trail'][0]
                        if center[1] < prev_center[1]:  # It's moving up
                            closest_blob['trail'].insert(0, center)  # Add point
                            closest_blob['last_seen'] = frame_time
                            
                            if len(closest_blob['trail']) > MIN_CENTRAL_POINTS:
                                cf = CF_LANE1
                                closest_blob['speed'].insert(0, calculate_speed(closest_blob['trail'], FPS))
                                lane = 1
                                ave_speed = np.mean(closest_blob['speed'])
                                abs_error, per_error = t.write_results_on_image(frame, frameCount, ave_speed, lane, closest_blob['id'], RESIZE_RATIO, VIDEO,
                                                                                dict_lane1, dict_lane2, dict_lane3)
                                try:
                                    results_lane1[str(closest_blob['id'])] = dict(ave_speed = round(ave_speed, 2),
                                                                             speeds = closest_blob['speed'],
                                                                             frame = frameCount, 
                                                                             real_speed = float(dict_lane1['speed']),
                                                                             abs_error = round(abs_error, 2),
                                                                             per_error = round(per_error, 3),
                                                                             trail = closest_blob['trail'],
                                                                             car_id = closest_blob['id'])
                                
                                    abs_error = []
                                    per_error = []
                                    if SHOW_FRAME_COUNT:
                                        PERCE = str(int((100*frameCount)/vehicle['videoframes']))
                                        cv2.putText(frame, f'frame: {frameCount} {PERCE}%', (r(14), r(1071)), 0, .65, t.WHITE, 2)                                    
                                    if SAVE_FRAME_F1:
                                        cv2.imwrite('results/{}/imagens/faixa1/{}_{}_F{}_{}.png'.format(DATE, VIDEO, dict_lane3['frame_start'], lane, closest_blob['id']), frame)
                                except:
                                    pass

                if not closest_blob: # Cria as variaves
                    # If we didn't find a blob, let's make a new one and add it to the list
                    b = dict(id=str(uuid.uuid4())[:8], first_seen=frame_time,
                             last_seen=frame_time, trail=[center], speed=[0],
                             size=[0, 0],)
                    tracked_blobs.append(b)  # Agora tracked_blobs não será False
                # ################# END TRACKING ##############################
                # ################# END FAIXA 3  ##############################
                # #############################################################
                # #############################################################
                # #############################################################


        fgmask_lane2 = bgsMOG.apply(frame_lane2, None, 0.01)
        erodedmask_lane2 = cv2.erode(fgmask_lane2, KERNEL_ERODE_L2, iterations=1)
        dilatedmask_lane2 = cv2.dilate(erodedmask_lane2, KERNEL_DILATE_L2, iterations=1)
        contours_L2, hierarchy = cv2.findContours(dilatedmask_lane2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #contornos =  cv2.drawContours(frame, contours, -1, BLUE, 2, 8, hierarchy)
        hull_L2 = []
        for i in range(len(contours_L2)):  # calculate points for each contour
            # creating convex hull object for each contour
            hull_L2.append(cv2.convexHull(contours_L2[i], False))
        # create an empty black image
        drawing_L2 = np.zeros((dilatedmask_lane2.shape[0], dilatedmask_lane2.shape[1], 3), np.uint8)
    #    area = []
    #    areahull = []
        #draw contours and hull points
        for i in range(len(contours_L2)):
            if cv2.contourArea(contours_L2[i]) > r(MIN_AREA_FOR_DETEC):
                # draw ith contour
                #cv2.drawContours(drawing, contours, i, t.GREEN, 0, 8, hierarchy)
                # draw ith convex hull object
                out_L2 = cv2.drawContours(drawing_L2, hull_L2, i, t.WHITE, -1, 8)
            #    area.append(cv2.contourArea(contours_L2[i]))
            #    areahull.append(cv2.contourArea(hull[i]))
                (x_L2, y_L2, w_L2, h_L2) = cv2.boundingRect(hull_L2[i])
                center_L2 = (int(x_L2 + w_L2/2), int(y_L2 + h_L2/2))
                #out_L2 = cv2.rectangle(out_L2, (x_L2, y_L2), (x_L2 + w_L2, y_L2 + h_L2), t.t.GREEN, 2) # printa na mask
                # CONDIÇÕES PARA CONTINUAR COM TRACKING
            #    if h_L2 > r(HEIGHT)*.80 or w_L2 > r(WIDTH)*.40:
            #        continue

                if w_L2 < r(340) and h_L2 < r(340):  # ponto que da pra mudar
                    continue
                # Área de medição do Tracking
                if center_L2[1] > r(BOTTOM_LIMIT_TRACK_L2) or center_L2[1] < r(UPPER_LIMIT_TRACK_L2):
                    continue
                
                
                if SHOW_CAR_RECTANGLE:
                    PADDING = r(600)
                    if center_L2[1] > r(UPPER_LIMIT_TRACK):
                        cv2.rectangle(frame_lane2, (x_L2, y_L2), (x_L2+w_L2, y_L2+h_L2), t.GREEN, 2)
                        cv2.rectangle(frame, (x_L2+PADDING, y_L2), (x_L2+w_L2+PADDING, y_L2+h_L2), t.GREEN, 2)
                        area_L2.append(w_L2*h_L2)
                    else:
                        cv2.rectangle(frame, (x_L2, y_L2), (x_L2+w_L2, y_L2+h_L2), t.PINK, 2)
                        cv2.rectangle(frame, (x_L2+PADDING, y_L2), (x_L2+w_L2+PADDING, y_L2+h_L2), t.PINK, 2)

                # ################## TRACKING #################################
                # Look for existing blobs that match this one
                closest_blob_L2 = None
                if tracked_blobs_lane2:
                    # Sort the blobs we have seen in previous frames by pixel distance from this one
                    closest_blobs_L2 = sorted(tracked_blobs_lane2, key=lambda b2: cv2.norm(b2['trail'][0], center_L2))

                    # Starting from the closest blob, make sure the blob in question is in the expected direction
                    distance = 0.0
                    for close_blob_L2 in closest_blobs_L2:
                        distance = cv2.norm(center_L2, close_blob_L2['trail'][0])

                        # Check if the distance is close enough to "lock on"
                        if distance < r(BLOB_LOCKON_DIST_PX_MAX) and distance > r(BLOB_LOCKON_DIST_PX_MIN):
                            closest_blob_L2 = close_blob_L2
                        #    continue # retirar depois
                            # If it's close enough, make sure the blob was moving in the expected direction
                            if close_blob_L2['trail'][0][1] < center_L2[1]:  # verifica se esta na dir up
                                continue
                            else:
                                closest_blob_L2 = close_blob_L2
                                continue  # defalut break

                    if closest_blob_L2:
                        # If we found a blob to attach this blob to, we should
                        # do some math to help us with speed detection
                        prev_center_L2 = closest_blob_L2['trail'][0]
                        if center_L2[1] < prev_center_L2[1]:  # It's moving up
                            closest_blob_L2['trail'].insert(0, center_L2)  # Add point
                            closest_blob_L2['last_seen'] = frame_time

                            if len(closest_blob_L2['trail']) > MIN_CENTRAL_POINTS:                                                                    
                                cf = CF_LANE2
                                closest_blob_L2['speed'].insert(0, calculate_speed(closest_blob_L2['trail'], FPS))
                                lane = 2
                                ave_speed = np.mean(closest_blob_L2['speed'])
                                abs_error, per_error = t.write_results_on_image(frame, frameCount, ave_speed, lane, closest_blob_L2['id'], RESIZE_RATIO, VIDEO,
                                                                                dict_lane1, dict_lane2, dict_lane3)                                
                                try:
                                    results_lane2[str(closest_blob_L2['id'])] = dict(ave_speed = round(ave_speed, 2),
                                                                             speeds = closest_blob_L2['speed'],
                                                                             frame = frameCount, 
                                                                             real_speed = float(dict_lane2['speed']),
                                                                             abs_error = round(abs_error, 2),
                                                                             per_error = round(per_error, 3),
                                                                             trail = closest_blob_L2['trail'],
                                                                             car_id = closest_blob_L2['id'])
                                
                                    abs_error = []
                                    per_error = []
                                    if SHOW_FRAME_COUNT:
                                        PERCE = str(int((100*frameCount)/vehicle['videoframes']))
                                        cv2.putText(frame, f'frame: {frameCount} {PERCE}%', (r(14), r(1071)), 0, .65, t.WHITE, 2)                                    
                                    if SAVE_FRAME_F2:
                                        cv2.imwrite('results/{}/imagens/faixa2/{}_{}_F{}_{}.png'.format(DATE, VIDEO, dict_lane3['frame_start'], lane, closest_blob_L2['id']), frame)
                                except:
                                    pass    
    
                if not closest_blob_L2: # Cria as variaves
                    # If we didn't find a blob, let's make a new one and add it to the list
                    b2 = dict(id=str(uuid.uuid4())[:8], first_seen=frame_time,
                             last_seen=frame_time, trail=[center_L2], speed=[0],
                             size=[0, 0],)
                    tracked_blobs_lane2.append(b2)  # Agora tracked_blobs não será False
                # ################# END TRACKING ##############################
                # ################# END FAIXA 2  ##############################
                # #############################################################
                # #############################################################
                # #############################################################

        fgmask_L3 = bgsMOG.apply(frame_lane3, None, 0.01)
        erodedmask_L3 = cv2.erode(fgmask_L3, KERNEL_ERODE_L3, iterations=1)
        dilatedmask_L3 = cv2.dilate(erodedmask_L3, KERNEL_DILATE_L3, iterations=1)
        contours_L3, hierarchy = cv2.findContours(dilatedmask_L3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #contornos =  cv2.drawContours(frame, contours, -1, BLUE, 2, 8, hierarchy)
        hull_L3 = []
        for i in range(len(contours_L3)):  # calculate points for each contour
            # creating convex hull object for each contour
            hull_L3.append(cv2.convexHull(contours_L3[i], False))
        # create an empty black image
        drawing_L3 = np.zeros((dilatedmask_L3.shape[0], dilatedmask_L3.shape[1], 3), np.uint8)
    #    areahull = []
        #draw contours_L3 and hull points
        for i in range(len(contours_L3)):
            if cv2.contourArea(contours_L3[i]) > r(MIN_AREA_FOR_DETEC):
                # draw ith contour
                #cv2.drawContours(drawing_L3, contours_L3, i, t.GREEN, 0, 8, hierarchy)
                # draw ith convex hull object
                out_L3 = cv2.drawContours(drawing_L3, hull_L3, i, t.WHITE, -1, 8)
            #    area.append(cv2.contourArea(contours_L3[i]))
            #    areahull.append(cv2.contourArea(hull[i]))
                (x_L3, y_L3, w_L3, h_L3) = cv2.boundingRect(hull_L3[i])
                center_L3 = (int(x_L3 + w_L3/2), int(y_L3 + h_L3/2))
                #out = cv2.rectangle(out, (x_L3, y_L3), (x_L3 + w_L3, y_L3 + h_L3), t.t.GREEN, 2) # printa na mask
                # CONDIÇÕES PARA CONTINUAR COM TRACKING
            #    if h_L3 > r(HEIGHT)*.80 or w_L3 > r(WIDTH)*.40:
            #        continue

                if w_L3 < r(340) and h_L3 < r(340):  # ponto que da pra mudar
                    continue
                # Área de medição do Tracking
                if center_L3[1] > r(BOTTOM_LIMIT_TRACK_L3) or center_L3[1] < r(UPPER_LIMIT_TRACK_L3):
                    continue
                
                
                if SHOW_CAR_RECTANGLE:
                    PADDING = r(1270)
                    if center_L3[1] > r(UPPER_LIMIT_TRACK):
                        cv2.rectangle(frame_lane3, (x_L3, y_L3), (x_L3+w_L3, y_L3+h_L3), t.GREEN, 2)
                        cv2.rectangle(frame, (x_L3+PADDING, y_L3), (x_L3+w_L3+PADDING, y_L3+h_L3), t.GREEN, 2)
                        area_L3.append(w_L3*h_L3)
                    else:
                        cv2.rectangle(frame_lane3, (x_L3, y_L3), (x_L3+w_L3, y_L3+h_L3), t.PINK, 2)
                        cv2.rectangle(frame, (x_L3+PADDING, y_L3), (x_L3+w_L3+PADDING, y_L3+h_L3), t.PINK, 2)

                # ################## TRACKING #################################
                # Look for existing blobs that match this one
                closest_blob_L3 = None
                if tracked_blobs_lane3:
                    # Sort the blobs we have seen in previous frames by pixel distance from this one
                    closest_blobs_L3 = sorted(tracked_blobs_lane3, key=lambda b3: cv2.norm(b3['trail'][0], center_L3))

                    # Starting from the closest blob, make sure the blob in question is in the expected direction
                    distance = 0.0
                    for close_blob_L3 in closest_blobs_L3:
                        distance = cv2.norm(center_L3, close_blob_L3['trail'][0])

                        # Check if the distance is close enough to "lock on"
                        if distance < r(BLOB_LOCKON_DIST_PX_MAX) and distance > r(BLOB_LOCKON_DIST_PX_MIN):
                            closest_blob_L3 = close_blob_L3
                        #    continue # retirar depois
                            # If it's close enough, make sure the blob was moving in the expected direction
                            if close_blob_L3['trail'][0][1] < center_L3[1]:  # verifica se esta na dir up
                                continue
                            else:
                                closest_blob_L3 = close_blob_L3
                                continue  # defalut break

                    if closest_blob_L3:
                        # If we found a blob to attach this blob to, we should
                        # do some math to help us with speed detection
                        prev_center_L3 = closest_blob_L3['trail'][0]
                        if center_L3[1] < prev_center_L3[1]:  # It's moving up
                            closest_blob_L3['trail'].insert(0, center_L3)  # Add point
                            closest_blob_L3['last_seen'] = frame_time

                            if len(closest_blob_L3['trail']) > MIN_CENTRAL_POINTS:
                                cf = CF_LANE3                                    
                                closest_blob_L3['speed'].insert(0, calculate_speed(closest_blob_L3['trail'], FPS))
                                lane = 3
                                ave_speed = np.mean(closest_blob_L3['speed'])
                                abs_error, per_error = t.write_results_on_image(frame, frameCount, ave_speed, lane, closest_blob_L3['id'], RESIZE_RATIO, VIDEO,
                                                                                dict_lane1, dict_lane2, dict_lane3)
                                
                                try:
                                    results_lane3[str(closest_blob_L3['id'])] = dict(ave_speed = round(ave_speed, 2),
                                                                                 speeds = closest_blob_L3['speed'],
                                                                                 frame = frameCount, 
                                                                                 real_speed = float(dict_lane3['speed']),
                                                                                 abs_error = round(abs_error, 2),
                                                                                 per_error = round(per_error, 3),
                                                                                 trail = closest_blob_L3['trail'],
                                                                                 car_id = closest_blob_L3['id'])
                                    abs_error = []
                                    per_error = []
                               
                                    if SHOW_FRAME_COUNT:
                                        PERCE = str(int((100*frameCount)/vehicle['videoframes']))
                                        cv2.putText(frame, f'frame: {frameCount} {PERCE}%', (r(14), r(1071)), 0, .65, t.WHITE, 2)                                    
                                    if SAVE_FRAME_F3:
                                        cv2.imwrite('results/{}/imagens/faixa3/{}_{}_F{}_{}.png'.format(DATE, VIDEO, dict_lane3['frame_start'], lane, closest_blob_L3['id']), frame)
                                except:
                                    pass
    
                if not closest_blob_L3: # Cria as variaves
                    # If we didn't find a blob, let's make a new one and add it to the list
                    b3 = dict(id=str(uuid.uuid4())[:8], first_seen=frame_time,
                             last_seen=frame_time, trail=[center_L3], speed=[0],
                             size=[0, 0],)
                    tracked_blobs_lane3.append(b3)  # Agora tracked_blobs não será False
                # ################# END TRACKING ##############################
                # ################# END FAIXA 3  ##############################
                # #############################################################
                # #############################################################
                # #############################################################


        if tracked_blobs:
            # Prune out the blobs that haven't been seen in some amount of time
            for i in range(len(tracked_blobs) - 1, -1, -1):
                if frame_time - tracked_blobs[i]['last_seen'] > BLOB_TRACK_TIMEOUT: # Deleta caso de timeout
                    print("Removing expired track {}".format(tracked_blobs[i]['id']))
                #    prev_speed = ave_speed
                #    final_ave_speed = 0.0
                    del tracked_blobs[i]

        if tracked_blobs_lane2:
            # Prune out the blobs that haven't been seen in some amount of time
            for i in range(len(tracked_blobs_lane2) - 1, -1, -1):
                if frame_time - tracked_blobs_lane2[i]['last_seen'] > BLOB_TRACK_TIMEOUT: # Deleta caso de timeout
                    print("Removing expired track {}".format(tracked_blobs_lane2[i]['id']))
                #    prev_speed = ave_speed
                #    final_ave_speed = 0.0
                    del tracked_blobs_lane2[i]

        if tracked_blobs_lane3:
            # Prune out the blobs that haven't been seen in some amount of time
            for i in range(len(tracked_blobs_lane3) - 1, -1, -1):
                if frame_time - tracked_blobs_lane3[i]['last_seen'] > BLOB_TRACK_TIMEOUT: # Deleta caso de timeout
                    print("Removing expired track {}".format(tracked_blobs_lane3[i]['id']))
                #    prev_speed = ave_speed
                #    final_ave_speed = 0.0
                    del tracked_blobs_lane3[i]

        # ################ PRINTA OS BLOBS ####################################
        for blob in tracked_blobs:  # Desenha os pontos centrais
            if SHOW_PARAMETERS['SHOW_TRAIL']:
                s.print_trail(blob['trail'], frame_lane1)

        for blob2 in tracked_blobs_lane2:  # Desenha os pontos centrais
            if SHOW_PARAMETERS['SHOW_TRAIL']:
                s.print_trail(blob2['trail'], frame_lane2)
                
        for blob3 in tracked_blobs_lane3:  # Desenha os pontos centrais
            if SHOW_PARAMETERS['SHOW_TRAIL']:
                s.print_trail(blob3['trail'], frame_lane3)

            if blob['speed'] and blob['speed'][0] != 0:
                prev_len_speed.insert(0, len(blob['speed']))
                # limpa prev_len_speed se estiver muito grande
                # deixa no máx 20 valores
                if len(prev_len_speed) > 20:
                    while len(prev_len_speed) > 20:
                        del prev_len_speed[19]
                # remove zero elements on the speed list
                blob['speed'] = [item for item in blob['speed'] if item != 0.0]
                print('========= speed list =========', blob['speed'])
                prev_speed = ave_speed
                ave_speed = np.mean(blob['speed'])
                print('========= prev_speed =========', float("{0:.5f}".format(prev_speed)))
                print('========= ave_speed ==========', float("{0:.5f}".format(ave_speed)))
            #    print('========prev_final_ave_speed==', float("{0:.5f}".format(final_ave_speed)))
            #    if ave_speed == prev_speed and final_ave_speed != 1:
            #        final_ave_speed = ave_speed
            #        print('===== final_ave_speed =====', float("{0:.5f}".format(final_ave_speed)))
            #        cv2.imwrite('img/{}speed_{}.png'.format(frameCount,final_ave_speed), frame)

                # ############### FIM PRINTA OS BLOBS  ########################


        print('*************************************************')
        
    #    cv2.line(frame, (, final_pt, t.ORANGE, 5)
    #    points = np.array([[[r(1410), r(1080)], [r(2170), r(1080)],
    #      [r(1320), r(0)], [r(990), r(0)]]])
    #    cv2.fillPoly(frame, points, t.RED)

        
        if SHOW_FRAME_COUNT:
            PERCE = str(int((100*frameCount)/vehicle['videoframes']))
            cv2.putText(frame, f'frame: {frameCount} {PERCE}%', (r(14), r(1071)), 0, .65, t.WHITE, 2)
        # ########## MOSTRA OS VIDEOS  ########################################
    #    cv2.imshow('equ', equ)
    #    cv2.imshow('res', res)
    #    cv2.imshow('fgmask', fgmask)
    #    cv2.imshow('erodedmask',erodedmask)
    #    cv2.imshow('dilatedmask', dilatedmask)
        
    #    cv2.imshow('fgmask_lane2', fgmask_lane2)
    #    cv2.imshow('erodedmask_lane2',erodedmask_lane2)
    #    cv2.imshow('dilatedmask_lane2', dilatedmask_lane2)

    #    cv2.imshow('fgmask_L3', fgmask_L3)
    #    cv2.imshow('erodedmask_L3',erodedmask_L3)
    #    cv2.imshow('dilatedmask_L3', dilatedmask_L3)
        
    #    cv2.imshow('contornos',contornos)
    #    cv2.imshow('out_L2',out_L2)
    #    cv2.imshow('out_L3',out_L3)

    #    cv2.imshow('res', res)
        cv2.imshow('frame_lane1', frame_lane1)
    #    cv2.imshow('frame_lane2', frame_lane2)
    #    cv2.imshow('frame_lane3', frame_lane3)
        cv2.imshow('frame', frame)
        
    #    final = np.hstack((erodedmask, dilatedmask))
    #    cv2.imshow('final', final)
    #    cv2.imshow('mask_eroded', np.concatenate((fgmask, dilatedmask),0))
    #    crop_img = outputFrame[70:320, 0:640]
        if frameCount > 857 and frameCount < 894:
    #    if frameCount == 114:
            cv2.imwrite('img/teste/{}.png'.format(frameCount), frame_lane3)
            cv2.imwrite('img/teste/dilatedmask_L3_{}.png'.format(frameCount), dilatedmask_L3)
            cv2.imwrite('img/teste/dilatedmask_L3_{}.png'.format(frameCount), dilatedmask_L3)
            cv2.imwrite('img/teste/out_L3_{}.png'.format(frameCount), out_L3)
        #    cv2.imwrite('img/teste/L3_{}.png'.format(frameCount), np.hstack((erodedmask_L3, out_L3)))
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

# ###### RESULTADOS ###########################################################
if SAVE_RESULTS:
    # Listas com sinais + e -
    abs_error_list1 = []
    abs_error_list2 = []
    abs_error_list3 = []
    # Módulo das Listas acima ( sem sinal )
    abs_error_list1_mod = []
    abs_error_list2_mod = []
    abs_error_list3_mod = []
    # Listas para faixa de valores
    # faixa 1
    #list_3km1 = []  # erros até 3km/h
    #list_5km1 = []  # erros até 5km/h
    #list_maior5km1 = []  # maiores que 5km/h
    ## faixa 2
    #list_3km2 = []  # erros até 3km/h
    #list_5km2 = []  # erros até 5km/h
    #list_maior5km2 = []  # maiores que 5km/h
    ## faixa 3
    #list_3km3 = []  # erros até 3km/h
    #list_5km3 = []  # erros até 5km/h
    #list_maior5km3 = []  # maiores que 5km/h
    # lista erros percentuais
    per_error_list1 = []
    per_error_list2 = []
    per_error_list3 = []
    
    for errors in results_lane1:
        abs_error_list1.append(results_lane1[errors]['abs_error'])
        abs_error_list1_mod.append(abs(results_lane1[errors]['abs_error']))
        per_error_list1.append(results_lane1[errors]['per_error'])
    ave_abs_error1 = round(np.mean(abs_error_list1_mod), 3)
    ave_per_error1 = round(np.mean(per_error_list1), 3)
    
    
    for errors in results_lane2:
        abs_error_list2.append(results_lane2[errors]['abs_error'])
        abs_error_list2_mod.append(abs(results_lane2[errors]['abs_error']))
        per_error_list2.append(results_lane2[errors]['per_error'])
    ave_abs_error2 = round(np.mean(abs_error_list2_mod), 3)
    ave_per_error2 = round(np.mean(per_error_list2), 3)
    
    for errors in results_lane3:
        abs_error_list3.append(results_lane3[errors]['abs_error'])
        abs_error_list3_mod.append(abs(results_lane3[errors]['abs_error']))
        per_error_list3.append(abs(results_lane3[errors]['per_error']))
    ave_abs_error3 = round(np.mean(abs_error_list3_mod), 3)
    ave_per_error3 = round(np.mean(per_error_list3), 3)
    
    list_3km1, list_5km1, list_maior5km1 = t.separar_por_kmh(abs_error_list1_mod)
    list_3km2, list_5km2, list_maior5km2 = t.separar_por_kmh(abs_error_list2_mod)
    list_3km3, list_5km3, list_maior5km3 = t.separar_por_kmh(abs_error_list3_mod)
    
    # Medidas pelo código
    total_cars_lane1 = len(results_lane1)
    total_cars_lane2 = len(results_lane2)
    total_cars_lane3 = len(results_lane3)
    
    # Taxa de Detecção
    rate_detec_lane1 = round(total_cars_lane1/vehicle['total_cars_lane1']*100, 2)
    rate_detec_lane2 = round(total_cars_lane2/vehicle['total_cars_lane2']*100, 2)
    rate_detec_lane3 = round(total_cars_lane3/vehicle['total_cars_lane3']*100, 2)
    
    t.plot_graph(abs_error_list1, ave_abs_error1, ave_per_error1, rate_detec_lane1, 
                   vehicle['total_cars_lane1'], total_cars_lane1, DATE, 1, VIDEO, CF_LANE1, True,
                   list_3km1, list_5km1, list_maior5km1) 
    
    t.plot_graph(abs_error_list2, ave_abs_error2, ave_per_error2, rate_detec_lane2, 
                   vehicle['total_cars_lane2'], total_cars_lane2, DATE, 2, VIDEO, CF_LANE2, True,
                   list_3km2, list_5km2, list_maior5km2)
        
    t.plot_graph(abs_error_list3, ave_abs_error3, ave_per_error3, rate_detec_lane3, 
                   vehicle['total_cars_lane3'], total_cars_lane3, DATE, 3, VIDEO, CF_LANE3, True,
                   list_3km3, list_5km3, list_maior5km3)
    
    # TOTAL
    total_abs_errors = abs_error_list1 + abs_error_list2 + abs_error_list3
    total_abs_errors_mod = abs_error_list1_mod + abs_error_list2_mod + abs_error_list3_mod
    total_per_errors = per_error_list1 + per_error_list2 + per_error_list3
    
    
    list_3km_tot, list_5km_tot, list_maior5km_tot = t.separar_por_kmh(total_abs_errors_mod)
    
    total_ave_abs = round(np.mean(total_abs_errors_mod), 3)
    total_ave_per = round(np.mean(total_per_errors), 3)
    total_cars = vehicle['total_cars_lane1']+vehicle['total_cars_lane2']+ vehicle['total_cars_lane3']
    total_rate_detec = round(len(total_abs_errors)/(total_cars)*100, 2)
    
    
    t.plot_graph(total_abs_errors, total_ave_abs, total_ave_per, total_rate_detec, 
                   total_cars, len(total_abs_errors), DATE, 'total', VIDEO, '---', True,
                   list_3km_tot, list_5km_tot, list_maior5km_tot)


#for i in range(len(abs_error_list)):
#    if x[i] > 50 and x[i] < 54:
#        x2.append(x[i])
#        y2.append(y[i])
        
#    abs_error.append(round(x[i]-y[i], 4))
#    erro_3km.append((3,-3))
    copy2('testes_homografica.py', f'results/{DATE}/')
    copy2('tccfunctions.py', f'results/{DATE}/')

    file = open(f'results/{DATE}/constantes.txt', 'w')
    file.write(f'VIDEO_FILE = {VIDEO_FILE} \n'
            f'XML_FILE = {XML_FILE} \n'
            f'FPS = {FPS} \n'
            f'RESIZE_RATIO = {RESIZE_RATIO} \n'
            f'CLOSE_VIDEO = {CLOSE_VIDEO} \n\n'
            f'BLOB_LOCKON_DIST_PX_MAX = {BLOB_LOCKON_DIST_PX_MAX} \n'
            f'BLOB_LOCKON_DIST_PX_MIN = {BLOB_LOCKON_DIST_PX_MIN} \n'
            f'MIN_AREA_FOR_DETEC = {MIN_AREA_FOR_DETEC} \n\n'
            f'BOTTOM_LIMIT_TRACK = {BOTTOM_LIMIT_TRACK} \n'
            f'UPPER_LIMIT_TRACK = {UPPER_LIMIT_TRACK} \n'
            f'MIN_CENTRAL_POINTS = {MIN_CENTRAL_POINTS} \n\n'
            f'BLOB_TRACK_TIMEOUT = {BLOB_TRACK_TIMEOUT} \n\n'
            f'CF_LANE1 = {CF_LANE1} \n'
            f'CF_LANE2 = {CF_LANE2} \n'
            f'CF_LANE3 = {CF_LANE3} \n\n'
            f'Tempo de processamento de cada frame \n'
            f'qndt de frames processados: {len(process_times)} \n'
            f'valor máximo: {max(process_times)}  segundos\n'
            f'valor mínimo: {min(process_times)}  segundos\n'
            f'valor médio: {np.mean(process_times)}  segundos\n\n' )
    file.close()

cap.release()
cv2.destroyAllWindows()

