
"""
Created on Tue Oct  2 21:15:05 2018

@author: broch
"""

#
import cv2
import numpy as np
import time
import uuid
import math
import xml.etree.ElementTree as ET
import os
from tccfunctions import *
#from itertools import *

##########  CONSTANT VALUES ##################################################
VIDEO_FILE = '../../Dataset/video1.avi' # Local do video a ser analisado
XML_FILE = '../../Dataset/video1.xml'
#XML_FILE = '3-xmlreader/video1.xml' # Igor

KERNEL_ERODE = np.ones((3, 3), np.uint8)  # Matriz (3,3) com 1 em seus valores -- Usa na funcao de erode
KERNEL_DILATE = np.ones((30, 15), np.uint8)  # Matriz (15,15) com 1 em seus valores -- Usa na funcao de dilate
# KERNEL_ERODE_SCND = np.ones((3,3), np.uint8)  # Matriz (8,8) com 1 em seus valores -- Usa na 2nd funcao de erode
RESIZE_RATIO = 0.35 # Resize, valores entre 0 e 1 | 1=Tamanho original do video
CLOSE_VIDEO = 6917#295  # Fecha o video no frame 400

# The maximum distance a blob centroid is allowed to move in order to
# consider it a match to a previous scene's blob.
BLOB_LOCKON_DISTANCE_PX = 50 # default = 80 # Aqui pode ser um limitador da veloc máx
LINE_THICKNESS = 1
# The number of seconds a blob is allowed to sit around without having
# any new blobs matching it.
BLOB_TRACK_TIMEOUT = 0.7 # Default 0.7

# Colors
WHITE  = (255, 255, 255)
BLACK  = (0  , 0  ,  0 )
BLUE   = (255, 0  ,  0 )
GREEN  = (0  , 255,  0 )
RED    = (0  , 0  , 255)
YELLOW = (0  , 255, 255)
CIAN   = (255, 255,  0 )
PINK   = (255, 0  , 255)

###############################################################
cap = cv2.VideoCapture(VIDEO_FILE)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Retorna a largura do video

bgsMOG = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold = 50, detectShadows=0)

# Variant Values
dict_lane1 = {}  # Dict que armazena os valores de "speed, frame_start, frame_end" da FAIXA 1
dict_lane2 = {}  # Dict que armazena os valores de "speed, frame_start, frame_end" da FAIXA 2
dict_lane3 = {}  # Dict que armazena os valores de "speed, frame_start, frame_end" da FAIXA 3
tracked_blobs = []  # Lista que salva os dicionários dos tracked_blobs
average_speed = [1]
meas_speed_lane1 = [0]
meas_speed_lane2 = [0]
meas_speed_lane3 = [0]
real_speed_lane1 = [0]
real_speed_lane2 = [0]
real_speed_lane3 = [0]

prev_len_speed = []

total_cars = {'lane_1':0, 'lane_2':0, 'lane_3':0}
speed = 1.0

frameCount = 0  # Armazena a contagem de frames processados do video
rectCount  = 0
rectCount2 = 0
out = 0  # Armazena o frame com os contornos desenhados

# ##############  FUNÇÕES ####################################################################
                     
def r(numero):  # Faz o ajuste de escala das posições de textos e retangulos
    return int(numero*RESIZE_RATIO)

def calculate_speed (trails, fps):
    # distance: distance on the frame
	# location: x, y coordinates on the frame
	# fps: framerate
	# mmp: meter per pixel
#	dist = cv2.norm(trails[0], trails[10])
	dist_x = trails[0][0] - trails[10][0]
	dist_y = trails[0][1] - trails[10][1]

	mmp_y = 0.125 / (3 * (1 + (3.22 / 432)) * trails[0][1])
#    mmp_y = 0.2 / (3 * (1 + (3.22 / 432)) * trails[0][1])  # Default
	mmp_x = 0.125 / (5 * (1 + (1.5 / 773)) * (width - trails[0][1]))  
#    mmp_x = 0.2 / (5 * (1 + (1.5 / 773)) * (width - trails[0][1]))  # Default
	real_dist = math.sqrt(dist_x * mmp_x * dist_x * mmp_x + dist_y * mmp_y * dist_y * mmp_y)

	return real_dist * fps * 250 / 3.6

def roi(frame):
    ###########  Region of Interest ###############################################
    # Retângulo superior
    cv2.rectangle(frame, (0,0), (r(1920), r(120)), BLUE , -1)
    # triângulo lado direito
    pts = np.array([[r(1920), r(750)], [r(1320),0], [r(1920),0]], np.int32)
    cv2.fillPoly(frame,[pts], BLUE)
    # triângulo lado esquerdo
    pts3 = np.array([[0, r(620)], [r(270),0], [0,0]], np.int32)
    cv2.fillPoly(frame,[pts3], BLUE)
    # Linha entre faixas 1 e 2
    pts1 = np.array([[r(510), r(1080)], [r(570),r(250)], 
                     [r(600),r(250)], [r(550),r(1080)]], np.int32)
    cv2.fillPoly(frame,[pts1], BLUE)
    # Linha entre faixas 2 e 3 SUPERIOR
    pts5 = np.array([[r(570), r(250)], [r(600),r(250)], 
                     [r(615),r(0)], [r(590),r(0)]], np.int32)
    cv2.fillPoly(frame,[pts5], BLUE)
    
    # Linha entre faixas 2 e 3
    pts2 = np.array([[r(1340), r(1080)], [r(1030),r(250)], 
                     [r(1060),r(250)], [r(1390),r(1080)]], np.int32)
    cv2.fillPoly(frame,[pts2], BLUE)
    # Linha entre faixas 2 e 3 SUPERIOR
    pts4 = np.array([[r(1030),r(250)], [r(1060),r(250)], 
                     [r(960),0], [r(930),0]], np.int32)
    cv2.fillPoly(frame,[pts4], BLUE)
    return frame
########### FIM Region of Interest ############################################

# ########## FIM  FUNÇÕES ####################################################################

vehicle = read_xml(XML_FILE)  # Dicionário que armazena todas as informações do xml

qntd_faixa1 = 0
qntd_faixa2 = 0
qntd_faixa3 = 0
#for vehicles in vehicle:
#    if vehicle[vehicles] == 6918:
#        break
#    if vehicle[vehicles]['lane'] == str(1):
#        qntd_faixa1 += 1
#    if vehicle[vehicles]['lane'] == str(2):
#        qntd_faixa2 += 1
#    if vehicle[vehicles]['lane'] == str(3):
#        qntd_faixa3 += 1


# Deleta os arquivos dos resultados, caso existam
if os.path.exists("results/real_speed_lane1.csv"):
  os.remove("results/real_speed_lane1.csv")
if os.path.exists("results/real_speed_lane2.csv"):
  os.remove("results/real_speed_lane2.csv")
if os.path.exists("results/real_speed_lane3.csv"):
  os.remove("results/real_speed_lane3.csv")

if os.path.exists("results/mea_speed_lane1.csv"):
  os.remove("results/mea_speed_lane1.csv")
if os.path.exists("results/mea_speed_lane2.csv"):
  os.remove("results/mea_speed_lane2.csv")  
if os.path.exists("results/mea_speed_lane3.csv"):
  os.remove("results/mea_speed_lane3.csv")
  
if os.path.exists("results/real_speed.csv"):
  os.remove("results/real_speed.csv")

while(True):
#    ret , frame = cap.read()
    ret, frame = get_frame(cap, RESIZE_RATIO)
    frame_time = time.time()
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi(frameGray)
    
#    crop_img = frameGray[0:250, 0:640]
#    crop_img = frameGray[0:320, 0:640]
#    crop_img = rotate_bound(crop_img, -5)
        
#    equ = cv2.equalizeHist(frameGray)
#    res = np.hstack((frameGray,equ))
#    frameGray = equ
    
    # Equalizar Contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(7,7))
    hist = clahe.apply(frameGray)
#    res = np.hstack((frameGray, cl1))
#    frameGray = hist
    
    if ret == True:
        update_info_xml(frameCount, vehicle, dict_lane1, dict_lane2, dict_lane3)
        # Cria a máscara
#        fgmask = bgsMOG.apply(crop_img, None, 0.01)
        fgmask = bgsMOG.apply(hist, None, 0.01)
        erodedmask = cv2.erode(fgmask, KERNEL_ERODE , iterations=1) # usa pra tirar os pixels isolados (ruídos)
        dilatedmask = cv2.dilate(erodedmask, KERNEL_DILATE , iterations=1) # usa para evidenciar o objeto em movimento
#        erodedmask = cv2.erode(fgmask, KERNEL_ERODE_SCND ,iterations=1) # usa pra tirar os pixels isolados (ruídos)
        # Fim da máscara
        _, contours, hierarchy = cv2.findContours(dilatedmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#        contornos =  cv2.drawContours(frame, contours, -1, (255,0,0), 2, 8, hierarchy) # IGOR
        
        # create hull array for convex hull points
        hull = []
        for i in range(len(contours)):  # calculate points for each contour
                # creating convex hull object for each contour
            hull.append(cv2.convexHull(contours[i], False))
            
        # create an empty black image
        drawing = np.zeros((dilatedmask.shape[0], dilatedmask.shape[1], 3), np.uint8)
        
        area = []
        areahull = []
        #draw contours and hull points
        rectCount2 = 0
        for i in range(len(contours)): #default  for i in range(len(contours))
#            if cv2.contourArea(contours[i]) > 50000:
#                rectCount += 1
#                break
#            if cv2.contourArea(contours[i]) < 1500:
#                rectCount += 1
#                break

#            if rectCount2 < 2:
            if cv2.contourArea(contours[i]) > 14000: # Default if cv2.contourArea(contours[i]) > 14000:
                # draw ith contour
                cv2.drawContours(drawing, contours, i, GREEN, 0, 8, hierarchy)
                # draw ith convex hull object
                out = cv2.drawContours(drawing, hull, i, WHITE, -1, 8)
                area.append(cv2.contourArea(contours[i]))
                areahull.append(cv2.contourArea(hull[i]))
                (x, y, w, h) = cv2.boundingRect(hull[i])
#                out = cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), 2) # printa na mask
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2) # printa na frame
                
                rectCount2 += 1
                
#                if y > 70:
##                    cv2.line(frame,(0,140),(640,140),(255,255,0),5)
##                    cv2.line(frame,(0,320),(640,320),(255,255,0),5)
##                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 4) # printa na mask
#                    pass
                
                if w < 60 and h < 60:  # ponto que da pra mudar
                    continue
                center = (int(x + w/2), int(y + h/2))
                
                if center[1] > 320 or center[1] < 150:  # ponto que da pra mudar
                    continue

#        outputFrame = cv2.drawContours(frame, contours, -1, (0,255,0),-1)
        
        
#        try: hierarchy = hierarchy[0]
#        except: hierarchy = []
#        a = []
#        for contour, hier in zip(contours, hierarchy):
#            (x, y, w, h) = cv2.boundingRect(contour)
#
#            if w < 60 and h < 60:
#                continue
##            if w > 400 and h > 280:
##                continue
##            area = h * w
##            if area > 10000 :
##                continue
#
#            center = (int(x + w/2), int(y + h/2))
#
#            if center[1] > 320 or center[1] < 150:
#                continue
#
#				# Optionally draw the rectangle around the blob on the frame that we'll show in a UI later
#            outputFrame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
#            
#            crop_img = frame[y:y+h , x:x+w]
#            cv2.imwrite('imagens/negatives/negativesframe{}.jpg'.format(frameCount),frame)
#
#            rectCount += 1
#          
#################### TRACKING ################################################

			# Look for existing blobs that match this one
                closest_blob = None
                if tracked_blobs:
                    # Sort the blobs we have seen in previous frames by pixel distance from this one
                    closest_blobs = sorted(tracked_blobs, key=lambda b: cv2.norm(b['trail'][0], center))
    
                    # Starting from the closest blob, make sure the blob in question is in the expected direction
                    distance = 0.0
                    distance_five = 0.0
                    for close_blob in closest_blobs:
                        distance = cv2.norm(center, close_blob['trail'][0])
                        if len(close_blob['trail']) > 10:
                            distance_five = cv2.norm(center, close_blob['trail'][10])
    					
                        # Check if the distance is close enough to "lock on"
                        if distance < BLOB_LOCKON_DISTANCE_PX:
                            # If it's close enough, make sure the blob was moving in the expected direction
                            expected_dir = close_blob['dir']
                            if expected_dir == 'up' and close_blob['trail'][0][1] < center[1]:
                                continue
#                            elif expected_dir == 'right' and close_blob['trail'][0][0] > center[0]:
#                                continue
                            else:
                                closest_blob = close_blob
                                break
    
                    if closest_blob:
                        # If we found a blob to attach this blob to, we should
                        # do some math to help us with speed detection
                        prev_center = closest_blob['trail'][0]
                        if center[1] < prev_center[1]:
                            # It's moving up
                            closest_blob['dir'] = 'up'
                            closest_blob['bumper_x'] = y
#                           
    					# ...and we should add this centroid to the trail of
    					# points that make up this blob's history.
                            closest_blob['trail'].insert(0, center)
                            closest_blob['last_seen'] = frame_time
                            if len(closest_blob['trail']) > 10:
                                closest_blob['speed'].insert(0, calculate_speed (closest_blob['trail'], fps))
    
                if not closest_blob: # Cria as variaves
    				# If we didn't find a blob, let's make a new one and add it to the list
                    b = dict(
    					id=str(uuid.uuid4())[:8],
    					first_seen=frame_time, # Coloca o mesmo valor na first_seen e last_seen
    					last_seen=frame_time, # Coloca o mesmo valor na first_seen e last_seen
    					dir=None,
    					bumper_x=None,
    					trail=[center],
    					speed=[0],  # Zera a velocidade
    					size=[0, 0], # Zera a tupla de tamanho
    				)
                    tracked_blobs.append(b)   # coloca as informacoes de "b" na lista "tracked_blobs"
                    # Agora tracked_blobs não será False
                    
################### FIM TRACKING ################################################
                    
                    
################## PRINTA OS BLOBS #############################################
        if tracked_blobs:
			# Prune out the blobs that haven't been seen in some amount of time
            for i in range(len(tracked_blobs) - 1, -1, -1): 
                if frame_time - tracked_blobs[i]['last_seen'] > BLOB_TRACK_TIMEOUT: # Deleta caso de timeout
                    print ("Removing expired track {}".format(tracked_blobs[i]['id']))
#                    save_csv()
                    speed = np.mean(tracked_blobs[i]['speed'])
                    del tracked_blobs[i]

        # Draw information about the blobs on the screen
        #print ('tracked_blobs', tracked_blobs)
        for blob in tracked_blobs:
            for (a, b) in pairwise(blob['trail']):
                cv2.circle(frame, a, 5, (255, 0, 0), LINE_THICKNESS)

                # print ('blob', blob)
                if blob['dir'] == 'up':
                    cv2.line(frame, a, b, CIAN, LINE_THICKNESS)
                else:
                    cv2.line(frame, a, b, YELLOW, LINE_THICKNESS)            
################# FIM PRINTA OS BLOBS  ##########################################
                
            if blob['speed'] and blob['speed'][0] != 0:
                prev_len_speed.insert(0, len(blob['speed']))
                # limpa prev_len_speed se estiver muito grande
                # deixa no máx 20 valores
                if len(prev_len_speed) > 20:
                    while len(prev_len_speed) > 20:
                        del prev_len_speed[19]
                # remove zero elements on the speed list
                blob['speed'] = [item for item in blob['speed'] if item != 0.0]
                print ('========= speed list =========', blob['speed'])
                ave_speed = np.mean(blob['speed'])
                print ('========= ave_speed =========', float("{0:.5f}".format(ave_speed)))
#                cv2.putText(frame, str(int(ave_speed)) + 'km/h', (blob['trail'][0][0] - 10, blob['trail'][0][1] + 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), thickness=1, lineType=2)

                if blob['trail'][0][0] < 200: # entao ta na faixa 1
                    cv2.putText(frame, str(float("{0:.2f}".format(ave_speed))) + 'km/h', (blob['trail'][0][0] + 20, blob['trail'][0][1] + 50), 
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), thickness=1, lineType=2)
                    cv2.putText(frame, 'Faixa 1', (blob['trail'][0][0] - 10, blob['trail'][0][1] + 70), 
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), thickness=1, lineType=2)
                    lane = 1
                elif blob['trail'][0][0] >= 200 and blob['trail'][0][0] < 400:  # entao ta na faixa 2
                    cv2.putText(frame, str(float("{0:.2f}".format(ave_speed))) + 'km/h', (blob['trail'][0][0] + 20, blob['trail'][0][1] + 50),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), thickness=1, lineType=2)
                    cv2.putText(frame, 'Faixa 2', (blob['trail'][0][0] - 10, blob['trail'][0][1] + 70), 
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), thickness=1, lineType=2)
                    lane = 2
                elif blob['trail'][0][0] >= 400 and blob['trail'][0][0] < 670:  # entao ta na faixa 3
                    lane = 3
                    cv2.putText(frame, str(float("{0:.2f}".format(ave_speed))) + 'km/h', (blob['trail'][0][0] + 20, blob['trail'][0][1] + 50),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), thickness=1, lineType=2)
                    cv2.putText(frame, 'Faixa 3', (blob['trail'][0][0] - 10, blob['trail'][0][1] + 70), 
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), thickness=1, lineType=2)
                    
     
                # Parte que coloca as velocidades reais no arquivo CSV
                # separados por faixas
                try:  # FAIXA 1
                    if dict_lane1['speed'] == real_speed_lane1[0]:
                        pass
                    else:
                        real_speed_lane1.insert(0, dict_lane1['speed'])
                        total_cars['lane_1'] += 1
                        file = open('results/real_speed_lane1.csv', 'a')
                        file.write('Carro {},{} \n'.format(total_cars['lane_1'], dict_lane1['speed']))
                        file.close()
                except:
                    pass
                
                try:  # FAIXA 2
                    if dict_lane2['speed'] == real_speed_lane2[0]:
                        pass
                    else:
                        real_speed_lane2.insert(0, dict_lane2['speed'])
                        total_cars['lane_2'] += 1
                        file = open('results/real_speed_lane2.csv', 'a')
                        file.write('Carro {},{} \n'.format(total_cars['lane_2'], dict_lane2['speed']))
                        file.close()
                except:
                    pass
                
                try:  # FAIXA 3
                    if dict_lane3['speed'] == real_speed_lane3[0]:
                        pass
                    else:
                        real_speed_lane3.insert(0, dict_lane3['speed'])
                        total_cars['lane_3'] += 1
                        file = open('results/real_speed_lane3.csv', 'a')
                        file.write('Carro {},{} \n'.format(total_cars['lane_3'], dict_lane3['speed']))
                        file.close()
                except:
                    pass

                
                 # CSV PART
                try:
                    if float("{0:.3f}".format(ave_speed)) == meas_speed_lane1[0] or len(blob['speed']) >= 2:
                        pass
                    elif lane == 1 and prev_len_speed[0] == prev_len_speed[1]:
                        meas_speed_lane1.insert(0, float("{0:.3f}".format(ave_speed)))
                        file = open('results/mea_speed_lane1.csv', 'a')
                        file.write('Carro {},{} \n'.format(total_cars['lane_1'], dict_lane1['speed']))
                        file.close()
                except:
                    pass
                
                try:
                    if float("{0:.3f}".format(ave_speed)) == meas_speed_lane2[0] or len(blob['speed']) < 2:
                        pass    
                    elif lane == 2 and prev_len_speed[0] == prev_len_speed[1]:
                        meas_speed_lane2.insert(0, float("{0:.3f}".format(ave_speed)))
                        file = open('results/mea_speed_lane2.csv', 'a')
                        file.write('Carro {},{} \n'.format(total_cars['lane_2'], dict_lane2['speed']))
                        file.close()
                except:
                    pass
                
                try:
                    if float("{0:.3f}".format(ave_speed)) == meas_speed_lane3[0] or len(blob['speed']) < 2:
                        pass       
                    elif lane == 3 and prev_len_speed[0] == prev_len_speed[1]:
                        meas_speed_lane3.insert(0, float("{0:.3f}".format(ave_speed)))
                        file = open('results/mea_speed_lane3.csv', 'a')
                        file.write('Carro {},{},{} \n'.format(total_cars['lane_3'], dict_lane3['speed'], meas_speed_lane3[0]))
                        file.close()
                except:
                    pass
                
                
                
                
#                with open('velocidades.csv', 'w') as csvfile:
#                    fieldnames = ['Valor Real', 'Valor Estimado']
#                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#                    
#                    writer.writeheader()
#                    writer.writerow({'Valor Real': int(ave_speed), 'Valor Estimado': int(ave_speed)})
#                    writer.writerow({'Valor Real': 1234, 'Valor Estimado': 'Spam'})
#                    writer.writerow({'Valor Real': 'Wonderful', 'Valor Estimado': 'Spam'})
                
#                with open('eggs.csv', 'w', newline='') as csvfile:
#                    spamwriter = csv.writer(csvfile, delimiter=' ',
#                                            quotechar=',', quoting=csv.QUOTE_MINIMAL)
#                    spamwriter.writerow(['Spam'] * 5 + ['Baked Beans'])
#                    spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])
                print('lane {}'.format(lane))        
        print ('*********************************************************************')
  
        # Mostra a qntd de frames processados e a % do video
        outputFrame = cv2.putText(frame, 'frame: {} {}%'.format(frameCount, 
                                    str(int((100*frameCount)/vehicle['videoframes']))), 
                                    (r(14), r(1071)), 0, .65, (255, 255, 255), 2)
        outputFrame = cv2.putText(frame, 'Retangulos descartados: {}'.format(rectCount), 
                                    (r(571),r(1071)), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 2)

        roi(outputFrame) # Ver como esta a ROI
        
        print_xml_values(outputFrame, RESIZE_RATIO, dict_lane1, dict_lane2, dict_lane3)

#        cv2.rectangle(outputFrame, (0,70), (640,320), (255,255,255) , 2)

#        crop_img = outputFrame[70:320, 0:640]
        
        # ########## MOSTRA OS VIDEOS  ################################################
#        cv2.imshow('crop_img', crop_img)
        
#        cv2.imshow('res', res)
        cv2.imshow('fgmask', fgmask)
        cv2.imshow('erodedmask',erodedmask)
        cv2.imshow('dilatedmask', dilatedmask)
#        cv2.imshow('contornos',contornos)     
#        cv2.imshow('out',out)
        cv2.imshow('outputFrame', outputFrame)
#        final = np.hstack((erodedmask, dilatedmask))
#        cv2.imshow('final', final)
        
        # Salva imagens para por no Markdown do GITHUB
        if frameCount ==285:
            cv2.imwrite('1frameGray.png', frameGray)
            cv2.imwrite('2hist.png', hist)
            cv2.imwrite('3fgmask.png', fgmask)
            cv2.imwrite('4erodedmask.png',erodedmask)
            cv2.imwrite('5dilatedmask.png', dilatedmask)
            cv2.imwrite('6resultado.png', outputFrame)

#        
        frameCount = frameCount + 1    # Conta a quantidade de Frames
        
        if frameCount == CLOSE_VIDEO: #  fecha o video
            break
        
        if cv2.waitKey(1) & 0xFF == ord('q'): #Pressiona a tecla Q para fechar o video
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
