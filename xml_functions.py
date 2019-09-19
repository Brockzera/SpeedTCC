import numpy as np
#import os
import itertools as it
#import math
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import cv2
#from tccfunctions import *

##### XML FUNCTIONS ###########################################################
def read_xml(xml_file, video, DATE):
# Funcão que lê o .xml e guarda as informações em um dicionário "iframe"
    tree = ET.parse(xml_file)
    root = tree.getroot()
    iframe = {}
    lane1_count = 0
    lane2_count = 0
    lane3_count = 0
    for child in root:
        if child.get('radar') == 'True':
            for subchild in child:
#                print('subchild {}'.format(subchild.attrib))
                if subchild.tag == 'radar':
                    iframe[child.get('iframe')] = subchild.attrib # salva frame_end, frame_start, speed
                    iframe[child.get('iframe')]['lane'] = child.get('lane')
                    iframe[child.get('iframe')]['radar'] = child.get('radar')
                    iframe[child.get('iframe')]['moto'] = child.get('moto')
#                    iframe[child.get('iframe')]['plate'] = child.get('plate')  # Desnecessário
#                    iframe[child.get('iframe')]['sema'] = child.get('sema')  # Desnecessário
                    speed = iframe[child.get('iframe')]['speed']
                    frame_start = iframe[child.get('iframe')]['frame_start']
                    frame_end = iframe[child.get('iframe')]['frame_end']
                    
                    if child.get('lane') == '1':
                        lane1_count += 1
                    if child.get('lane') == '2':
                        lane2_count += 1
                    if child.get('lane') == '3':
                        lane3_count += 1
                    
                    
                    if iframe[child.get('iframe')]['lane'] == '1':
                        file = open(f'results/{DATE}/planilhas/video{video}_real_lane1.csv', 'a')
                        file.write(f'frame_start, {frame_start}, frame_end, {frame_end}, speed, {speed} \n')
                        file.close()
                    if iframe[child.get('iframe')]['lane'] == '2':
                        file = open(f'results/{DATE}/planilhas/video{video}_real_lane2.csv', 'a')
                        file.write(f'frame_start, {frame_start}, frame_end, {frame_end}, speed, {speed} \n')
                        file.close()
                    if iframe[child.get('iframe')]['lane'] == '3':
                        file = open(f'results/{DATE}/planilhas/video{video}_real_lane3.csv', 'a')
                        file.write(f'frame_start, {frame_start}, frame_end, {frame_end}, speed, {speed} \n')
                        file.close()
        if child.tag == 'videoframes':
            iframe[child.tag] = int(child.get('total'))
    
    iframe['total_cars_lane1'] = lane1_count
    iframe['total_cars_lane2'] = lane2_count
    iframe['total_cars_lane3'] = lane3_count
        
    file_name = xml_file[xml_file.rfind('/')+1:]  
    print('Arquivo "{}" foi armazenado com sucesso !!'.format(file_name))
    
    return iframe


def update_info_xml(frameCount, vehicle, dict_lane1, dict_lane2, dict_lane3):
    try:
        # Verifica se naquele frame tem uma chave correspondente no dicionário "vehicle"
        if vehicle[str(frameCount)]['frame_start'] == str(frameCount):
            lane_state = vehicle[str(frameCount)]['lane']

            if lane_state == '1':  # Se for na faixa 1, armazena as seguintes infos
                # print('Faixa 1')
                dict_lane1['speed'] = vehicle[str(frameCount)]['speed']
                dict_lane1['frame_start'] = vehicle[str(frameCount)]['frame_start']
                dict_lane1['frame_end'] = vehicle[str(frameCount)]['frame_end']
            elif lane_state == '2':
                #print('Faixa 2')
                dict_lane2['speed'] = vehicle[str(frameCount)]['speed']
                dict_lane2['frame_start'] = vehicle[str(frameCount)]['frame_start']
                dict_lane2['frame_end'] = vehicle[str(frameCount)]['frame_end']
            elif lane_state == '3':
                #print('Faixa 3')
                dict_lane3['speed'] = vehicle[str(frameCount)]['speed']
                dict_lane3['frame_start'] = vehicle[str(frameCount)]['frame_start']
                dict_lane3['frame_end'] = vehicle[str(frameCount)]['frame_end']
    except KeyError:
#            print('KeyError: Key Não encotrada no dicionário')
        pass
