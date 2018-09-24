
"""
Created on Mon Sep 17 13:24:34 2018

@author: broch
"""

import cv2
import numpy as np
#import os
import time
import uuid
import math

cap = cv2.VideoCapture("../Dataset/kit.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))      # Retorna a largura do video
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    # Retorna a altura do video
# length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))     # Retorna a quantidade de frames

kernel1 = np.ones((3, 3), np.uint8) # Matriz (3,3) com 1 em seus valores -- Usa na funcao de erode
kernel2 = np.ones((15, 15), np.uint8) # # Matriz (8,8) com 1 em seus valores -- Usa na funcao de dilate
kernel3 = np.ones((3,3), np.uint8) # # Matriz (8,8) com 1 em seus valores -- Usa na funcao de dilate

RESIZE_RATIO = 0.55 #


bgsMOG = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold = 50, detectShadows=0)

frameCount = 0
rectCount = 0

# A list of "tracked blobs".
tracked_blobs = []
# The maximum distance a blob centroid is allowed to move in order to
# consider it a match to a previous scene's blob.
BLOB_LOCKON_DISTANCE_PX = 80 # default = 80
LINE_THICKNESS = 1
# The number of seconds a blob is allowed to sit around without having
# any new blobs matching it.
BLOB_TRACK_TIMEOUT = 0.7


def get_frame():
	#" Grabs a frame from the video vcture and resizes it. "
	ret, frame = cap.read()
	if ret:
		(h, w) = frame.shape[:2]
		frame = cv2.resize(frame, (int(w * RESIZE_RATIO), int(h * RESIZE_RATIO)), interpolation=cv2.INTER_CUBIC)
	return ret, frame

def calculate_speed (trails, fps):
    # distance: distance on the frame
	# location: x, y coordinates on the frame
	# fps: framerate
	# mmp: meter per pixel
#	dist = cv2.norm(trails[0], trails[10])
	dist_x = trails[0][0] - trails[10][0]
	dist_y = trails[0][1] - trails[10][1]

	mmp_y = 0.2 / (3 * (1 + (3.22 / 432)) * trails[0][1])
	mmp_x = 0.2 / (5 * (1 + (1.5 / 773)) * (width - trails[0][1]))
	real_dist = math.sqrt(dist_x * mmp_x * dist_x * mmp_x + dist_y * mmp_y * dist_y * mmp_y)

	return real_dist * fps * 250 / 3.6

from itertools import *


def pairwise(iterable):
	r"s -> (s0,s1), (s1,s2), (s2, s3), ..."
	a, b = tee(iterable)
	next(b, None)
	return zip(a, b)

while(True):
#    ret , frame = cap.read()
    ret, frame = get_frame()
    frame_time = time.time()
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    edged = cv2.Canny(frameGray, 30, 200)
    
    if ret == True:
        # Cria a máscara
        fgmask = bgsMOG.apply(frameGray, None, 0.01)
        erodedmask = cv2.erode(fgmask, kernel1 ,iterations=1) # usa pra tirar os pixels isolados (ruídos)
        dilatedmask = cv2.dilate(erodedmask, kernel2 ,iterations=1) # usa para evidenciar o objeto em movimento
#        erodedmask = cv2.erode(fgmask, kernel3 ,iterations=1) # usa pra tirar os pixels isolados (ruídos)
        # Fim da máscara
        _, contours, hierarchy = cv2.findContours(dilatedmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#        outputFrame = cv2.drawContours(frame, contours, -1, (0,255,0),3)
        
        
        try: hierarchy = hierarchy[0]
        except: hierarchy = []
        a = []
        for contour, hier in zip(contours, hierarchy):
            (x, y, w, h) = cv2.boundingRect(contour)

            if w < 60 and h < 60:
                continue
#            if w > 400 and h > 280:
#                continue
#            area = h * w
#            if area > 10000 :
#                continue

            center = (int(x + w/2), int(y + h/2))

            if center[1] > 320 or center[1] < 150:
                continue

				# Optionally draw the rectangle around the blob on the frame that we'll show in a UI later
            # outputFrame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            crop_img = frame[y:y+h , x:x+w]
            cv2.imwrite('imagens/negatives/negativesframe{}.jpg'.format(frameCount),frame)

            rectCount += 1
            


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
                        if expected_dir == 'left' and close_blob['trail'][0][0] < center[0]:
                            continue
                        elif expected_dir == 'right' and close_blob['trail'][0][0] > center[0]:
                            continue
                        else:
                            closest_blob = close_blob
                            break

                if closest_blob:
                    # If we found a blob to attach this blob to, we should
                    # do some math to help us with speed detection
                    prev_center = closest_blob['trail'][0]
                    if center[0] < prev_center[0]:
                        # It's moving left
                        closest_blob['dir'] = 'left'
                        closest_blob['bumper_x'] = x
                    else:
						# It's moving right
                        closest_blob['dir'] = 'right'
                        closest_blob['bumper_x'] = x + w

					# ...and we should add this centroid to the trail of
					# points that make up this blob's history.
                    closest_blob['trail'].insert(0, center)
                    closest_blob['last_seen'] = frame_time
                    if len(closest_blob['trail']) > 10:
                        closest_blob['speed'].insert(0, calculate_speed (closest_blob['trail'], fps))

            if not closest_blob:
				# If we didn't find a blob, let's make a new one and add it to the list
                b = dict(
					id=str(uuid.uuid4())[:8],
					first_seen=frame_time,
					last_seen=frame_time,
					dir=None,
					bumper_x=None,
					trail=[center],
					speed=[0],
					size=[0, 0],
				)
                tracked_blobs.append(b)        
# PRINTA OS BLOBS
        if tracked_blobs:
			# Prune out the blobs that haven't been seen in some amount of time
            for i in range(len(tracked_blobs) - 1, -1, -1):
                if frame_time - tracked_blobs[i]['last_seen'] > BLOB_TRACK_TIMEOUT:
                    print ("Removing expired track {}".format(tracked_blobs[i]['id']))
                    del tracked_blobs[i]

        # Draw information about the blobs on the screen
        print ('tracked_blobs', tracked_blobs)
        for blob in tracked_blobs:
            for (a, b) in pairwise(blob['trail']):
                cv2.circle(frame, a, 3, (255, 0, 0), LINE_THICKNESS)

                # print ('blob', blob)
                if blob['dir'] == 'left':
                    pass
                    cv2.line(frame, a, b, (255, 255, 0), LINE_THICKNESS)
                else:
                    pass
                    cv2.line(frame, a, b, (0, 255, 255), LINE_THICKNESS)            
# FIM PRINTA OS BLOBS
                
            if blob['speed'] and blob['speed'][0] != 0:
                # remove zero elements on the speed list
                blob['speed'] = [item for item in blob['speed'] if item != 0.0]
                print ('========= speed list =========', blob['speed'])
                ave_speed = np.mean(blob['speed'])
                print ('========= ave_speed =========', ave_speed)
                cv2.putText(frame, str(int(ave_speed)) + 'km/h', (blob['trail'][0][0] - 10, blob['trail'][0][1] + 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), thickness=1, lineType=2)

        print ('*********************************************************************')
        
        outputFrame = cv2.putText(frame, 'frame: {}'.format(frameCount), (5,375), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 2)
        outputFrame = cv2.putText(frame, 'Retangulos: {}'.format(rectCount), (200,375), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 2)
        
        if frameCount >= 71 and frameCount <= 111:
            outputFrame=  cv2.putText(frame, '56.65', (100,375), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 2)


        if frameCount == 3000: # fecha o video
            break
        
        
        
        # cv2.imshow('erodedmask',erodedmask)
        # cv2.imshow('dilatedmask', dilatedmask)
        # cv2.imshow('outputFrame', outputFrame)
        
        frameCount = frameCount + 1    # Conta a quantidade de Frames
        
        
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'): #Pressiona a tecla Q para fechar o video
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
