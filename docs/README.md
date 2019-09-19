# SpeedTCC

###  1. Módulos
````python
''' -*- coding: utf-8 -*-'''
import time
import uuid
import numpy as np
import cv2
import tccfunctions as t
````
- `import time` é usado para armazenar o tempo do primeiro ponto central que foi processado até o último. 
- `import uuid` é usado para gerar um id único para cada **trail**.
- `import numpy as np` é usado para criar/trabalhar com matrizes.
- `import tccfunctions as t` é usado para importar o as funções que fiz. Pra acessá-las usa-se **`t.nome_da_funcao()`** ou **`t.nome_da_variável`**
---

###  2. Valores Constantes
```python
# ########  CONSTANT VALUES ###################################################
VIDEO = 1
VIDEO_FILE = '../Dataset/video{}.avi'.format(VIDEO)
XML_FILE = '../Dataset/video{}.xml'.format(VIDEO)

RESIZE_RATIO = 0.35  # Resize, valores entre 0 e 1 | 1= ize original do video
CLOSE_VIDEO = 953  # 6917 # Fecha o video no frame indicado
```
````python
SHOW_ROI = True  # Mostra ou não a região de interesse. 
SHOW_TRACKING_AREA = True 
SHOW_TRAIL = True # Mostra os pontos centrais
SHOW_LINEAR_REGRESSION = True
SHOW_CAR_RECTANGLE = True
````
- `SHOW_TRACKING_AREA` Mostra os limites da área de medição, limites que são definidos por `BOTTOM_LIMIT_TRACK` e `UPPER_LIMIT_TRACK`.

````python
SHOW_REAL_SPEEDS = True # Mostra as velocidades que foram lidas do arquivo **.xml**
SHOW_FRAME_COUNT = True

SKIP_VIDEO = True
SEE_CUTTED_VIDEO = False  # ver partes retiradas, needs SKIP_VIDEO = True

````
- `SKIP_VIDEO` Define se irá ou não pular as partes ondem não tem carro ou os carros estão parados.  
- `SEE_CUTTED_VIDEO` Se for `True` vai mostrar apenas as partes que foram retiradas do video.

````python
# ---- Tracking Values --------------------------------------------------------
# The maximum distance a blob centroid is allowed to move in order to
# consider it a match to a previous scene's blob.
BLOB_LOCKON_DIST_PX_MAX = 150  # default = 50 p/ ratio 0.35
BLOB_LOCKON_DIST_PX_MIN = 5  # default 5
MIN_AREA_FOR_DETEC = 40000  # Default 40000 (não detecta Moto)
# Limites da Área de Medição, área onde é feita o Tracking
# Distancia de medição: default 915-430 = 485
BOTTOM_LIMIT_TRACK = 915  # Default 915
UPPER_LIMIT_TRACK = 430  # Default 430
# The number of seconds a blob is allowed to sit around without having
# any new blobs matching it.
BLOB_TRACK_TIMEOUT = 0.7  # Default 0.7
````
- `BLOB_LOCKON_DIST_PX_MIN` e `BLOB_LOCKON_DIST_PX_MAX` são **parâmetros importantes**. Pois, exemplo: se um _trail_ tem no momento 5 pontos centrais, e sexto ponto não satisfaz nenhuma das distância mínima ou máxima. Esse sexto ponto, irá dar início à um novo _trail_. Portanto, aquele _trail_ com 5 pontos não terá a velocidade calculada e nem o novo _trail_ terá pontos suficientes para calcular a velocidade. Resumindo, perdemos esse veículo.
- `BLOB_TRACK_TIMEOUT` define quanto tempo o _trail_ ficará esperando por um novo ponto central.  Depois desse tempo, o _trail_ é removido.

````python
# ---- Speed Values -----------------------------------------------------------
CF_LANE1 = 2.5869977  # default 2.5869977 # Correction Factor
CF_LANE2 = 2.5869977  # default 2.5869977
CF_LANE3 = 2.30683  # default 2.3068397
````
- Os **fatores de correção**, na minha opinião devem ser a última coisa a se mexer. Pois sempre que faço alguma mudança no código os valores de velocidade se alteram daí tem que ficar acertando o fator de correção de novo.

````python
# ----  Save Results Values ---------------------------------------------------
SAVE_FRAME_F1 = False  # Faixa 1
SAVE_FRAME_F2 = False  # Faixa 2
SAVE_FRAME_F3 = False  # Faixa 3
# ####### END - CONSTANT VALUES ###############################################
````
- As variáveis `SAVE_FRAME_F` vão salvar todos os frames que contém velocidades medidas. 
- **DICA:** altere o valor de `BLOB_TRACK_TIMEOUT` para **0.1**, pois assim as salvará menos frames.
    - Com `BLOB_TRACK_TIMEOUT = 0.7`, salva uns 14 frames por carro.
    - Com `BLOB_TRACK_TIMEOUT = 0.1`, salva uns 6 frames por carro.
- As imagens serão salvas em `img/novo/`.
- O nome dos arquivos serão assim `{1}-{2}_F1_Carro_{3}.png`
    - {1} será o número do video.
    - {2} é o número do frame.
    - F1 - Faixa 1, F2 - Faixa 2, F3 - Faixa 3
    - {3} é a número do carro que passou naquela faixa. **OBS**: Não confiar nesse número, pode ver quando salvar, que as vezes um carro é contado como dois.
> Para comparar os resultados, copio as imagens e que foram salvas em `img/novo` para `img/velho`, daí mudo os parâmetros e vejo como ficou, comparando as imagens de `img/novo` e `img/velho`.

````python
cap = cv2.VideoCapture(VIDEO_FILE)
FPS = cap.get(cv2.CAP_PROP_FPS)
Fra = cap.get(cv2.CAP_PROP_FRAME_COUNT)
WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Retorna a largura do video
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Retorna a largura do video

bgsMOG = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=50, detectShadows=0)

# Variant Values
dict_lane1 = {}  # Armazena os valores de "speed, frame_start, frame_end" da FAIXA 1
dict_lane2 = {}  # Armazena os valores de "speed, frame_start, frame_end" da FAIXA 2
dict_lane3 = {}  # Armazena os valores de "speed, frame_start, frame_end" da FAIXA 3
tracked_blobs = []  # Lista que salva os dicionários dos tracked_blobs
average_speed = [1]
meas_speed_lane1 = [0]
meas_speed_lane2 = [0]
meas_speed_lane3 = [0]
real_speed_lane1 = [0]
real_speed_lane2 = [0]
real_speed_lane3 = [0]

prev_len_speed = []
total_cars = {'lane_1': 0, 'lane_2': 0, 'lane_3': 0}
prev_speed = 1.0

frameCount = 0  # Armazena a contagem de frames processados do video
out = 0  # Armazena o frame com os contornos desenhados
final_ave_speed = 0
ave_speed = 0
````
### 3. Funções
````python
# ##############  FUNÇÕES #####################################################
def r(numero):
    return int(numero*RESIZE_RATIO)

def crop(img):
    return img[30:370, 205:620]

def calculate_speed(trails, fps):
    med_area_meter = 3.5  # metros (Valor estimado)
    med_area_pixel = r(485)
    qntd_frames = len(trails)  # default 11
    initial_pt, final_pt = t.linearRegression(trails, qntd_frames)  # Usando Regressão Linear
    dist_pixel = cv2.norm(final_pt, initial_pt)
#    dist_pixel1 = cv2.norm(trails[0], trails[10])  # Sem usar Regressão linear
    if SHOW_LINEAR_REGRESSION:
        cv2.line(frame, initial_pt, final_pt, t.ORANGE, 5)
#    cv2.line(frame,trails[0],trails[10], t.GREEN, 2)
#    cv2.imwrite('img/regressao1_{}.png'.format(frameCount), frame)
    dist_meter = dist_pixel*(med_area_meter/med_area_pixel)
    speed = (dist_meter*3.6*cf)/(qntd_frames*(1/fps))
    return speed
# ########## FIM  FUNÇÕES #####################################################
````

````python
vehicle = t.read_xml(XML_FILE)  # Dicionário que armazena todas as informações do xml
t.remove_old_csv_files()  # remove os arquivos csv que são criado em cada execução do código. 
# Esses arquivos armazenam as velocidades calculadas, mas ainda não estão confiáveis.
````
- Kernel são usados nas **Operações morfológicas**

````py
KERNEL_ERODE = np.ones((r(9), r(9)), np.uint8)
KERNEL_DILATE = np.ones((r(100), r(50)), np.uint8)  # Default (r(100), r(50))
````

### 4. Começo do código
````py
while True:
    ret, frame = t.get_frame(cap, RESIZE_RATIO)  # Lê o frame, ret == True se existe frame
    frame_time = time.time()
    
    if SKIP_VIDEO:  # pula as partes que não precisam
        skip = t.skip_video(frameCount, VIDEO, frame)
        if SEE_CUTTED_VIDEO:
            if not skip:
                frameCount += 1
                continue
        else:
            if skip:
                frameCount += 1
                continue
            
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Transforma o frame em Escala de Cinza
    t.region_of_interest(frameGray, RESIZE_RATIO)  # aplica a região de interesse
    
    
    if SHOW_ROI: # if True mostra a Região de Interesse
        t.region_of_interest(frame, RESIZE_RATIO)
    if SHOW_TRACKING_AREA:  # Desenha os Limites da Área de Tracking
        cv2.line(frame, (0, r(UPPER_LIMIT_TRACK)), (WIDTH, r(UPPER_LIMIT_TRACK)), t.WHITE, 2)
        cv2.line(frame, (0, r(BOTTOM_LIMIT_TRACK)), (WIDTH, r(BOTTOM_LIMIT_TRACK)), t.WHITE, 2)
````
#### 4.1 frame
![frame](https://github.com/Brockzera/SpeedTCC/blob/master/backup/imagens/1frame254.png)

#### 4.2 frameGray
![2framegray254](https://github.com/Brockzera/SpeedTCC/blob/master/backup/imagens/2framegray254.png)

#### 4.3 Roi
![3roi254](https://github.com/Brockzera/SpeedTCC/blob/master/backup/imagens/3roi254.png)

### 5. Equaliza o contraste
````python       
    # Equalizar Contraste
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(7, 7))
    hist = clahe.apply(frameGray)
    frameGray = hist
````
#### 5.1 hist
![4hist254](https://github.com/Brockzera/SpeedTCC/blob/master/backup/imagens/4hist254.png)

### 6. Operações morfológicas
````py    
    if ret is True:
        t.update_info_xml(frameCount, vehicle, dict_lane1, dict_lane2, dict_lane3)
        if SHOW_REAL_SPEEDS:
            t.print_xml_values(frame, RESIZE_RATIO, dict_lane1, dict_lane2, dict_lane3)

        fgmask = bgsMOG.apply(frameGray, None, 0.01) # Aplica o BackgroundSubtractor
        erodedmask = cv2.erode(fgmask, KERNEL_ERODE, iterations=1) # Aplica Erosão
        dilatedmask = cv2.dilate(erodedmask, KERNEL_DILATE, iterations=1) # Aplica Diltação

        # Encontra os contornos e salva os pontos em 'contours'
        _, contours, hierarchy = cv2.findContours(dilatedmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 ````
#### 6.1 fgmask
![5fgmask254](https://github.com/Brockzera/SpeedTCC/blob/master/backup/imagens/5fgmask254.png)

#### 6.2 erodedmask
![6erodedmask254](https://github.com/Brockzera/SpeedTCC/blob/master/backup/imagens/6erodedmask254.png)

#### 6.3 dilatedmask
![7dilatedmask254](https://github.com/Brockzera/SpeedTCC/blob/master/backup/imagens/7dilatedmask254.png)

### 7. Convex Hull
````py
        hull = []
        for i in range(len(contours)):  # calculate points for each contour
            # creating convex hull object for each contour
            hull.append(cv2.convexHull(contours[i], False))

        # create an empty black image
        drawing = np.zeros((dilatedmask.shape[0], dilatedmask.shape[1], 3), np.uint8) # Nessa imagem q será desenhado o ConvexHull
        area = []
        areahull = []
````
### 8. Condições para desenhar o retângulo
````py
        for i in range(len(contours)):
            if cv2.contourArea(contours[i]) > r(MIN_AREA_FOR_DETEC):
                # draw ith contour
                #cv2.drawContours(drawing, contours, i, t.GREEN, 0, 8, hierarchy)
                # draw ith convex hull object
                out = cv2.drawContours(drawing, hull, i, t.WHITE, -1, 8)
                area.append(cv2.contourArea(contours[i]))
                areahull.append(cv2.contourArea(hull[i]))
                (x, y, w, h) = cv2.boundingRect(hull[i]) # Faz o retangulo envolta do objeto           
                center = (int(x + w/2), int(y + h/2)) # Ponto central definido
                #out = cv2.rectangle(out, (x, y), (x + w, y + h), t.t.GREEN, 2) # printa na mask
                # CONDIÇÕES PARA CONTINUAR COM TRACKING
#                if h > r(HEIGHT)*.80 or w > r(WIDTH)*.40:
#                    continue

                if w < r(340) and h < r(340):  # ponto que da pra mudar
                    continue
                # Área de medição do Tracking
                if center[1] > r(BOTTOM_LIMIT_TRACK) or center[1] < r(UPPER_LIMIT_TRACK):
                    continue
                
                if SHOW_CAR_RECTANGLE:
                    if center[1] > r(UPPER_LIMIT_TRACK):
                        cv2.rectangle(frame, (x, y), (x+w, y+h), t.GREEN, 2)
                    else:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), t.PINK, 2)
````                
#### Resultado de `cv2.drawContours(drawing, hull, i, t.WHITE, -1, 8)`
![8out254](https://github.com/Brockzera/SpeedTCC/blob/master/backup/imagens/8out254.png)

### 9. _Tracking_
````py
                # ################## TRACKING #################################
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
                        if distance < r(BLOB_LOCKON_DIST_PX_MAX) and distance > r(BLOB_LOCKON_DIST_PX_MIN):
                            closest_blob = close_blob
                            continue # retirar depois
                            # If it's close enough, make sure the blob was moving in the expected direction
#                            if close_blob['trail'][0][1] < center[1]:  # verifica se esta na dir up
#                                continue
#                            else:
#                                closest_blob = close_blob
#                                continue  # defalut break

                    if closest_blob:
                        # If we found a blob to attach this blob to, we should
                        # do some math to help us with speed detection
                        prev_center = closest_blob['trail'][0]
                        if center[1] < prev_center[1]:  # It's moving up
                            closest_blob['trail'].insert(0, center)  # Add point
                            closest_blob['last_seen'] = frame_time
                            if len(closest_blob['trail']) > 10:
                                if closest_blob['trail'][0][0] < r(570):
                                    cf = CF_LANE1
                                    closest_blob['speed'].insert(0, calculate_speed(closest_blob['trail'], FPS))
                                elif closest_blob['trail'][0][0] >= r(570) and closest_blob['trail'][0][0] < r(1180):
                                    cf = CF_LANE2
                                    closest_blob['speed'].insert(0, calculate_speed(closest_blob['trail'], FPS))
                                elif closest_blob['trail'][0][0] >= r(1180):
                                    cf = CF_LANE3
                                    closest_blob['speed'].insert(0, calculate_speed(closest_blob['trail'], FPS))

                if not closest_blob: # Cria as variaves
                    # If we didn't find a blob, let's make a new one and add it to the list
                    b = dict(id=str(uuid.uuid4())[:8], first_seen=frame_time,
                             last_seen=frame_time, trail=[center], speed=[0],
                             size=[0, 0],)
                    tracked_blobs.append(b)  # Agora tracked_blobs não será False
                # ################# END TRACKING ##############################
````
### 10. Deleta os _trails_
````py
        if tracked_blobs:
            # Prune out the blobs that haven't been seen in some amount of time
            for i in range(len(tracked_blobs) - 1, -1, -1):
                if frame_time - tracked_blobs[i]['last_seen'] > BLOB_TRACK_TIMEOUT: # Deleta caso de timeout
                    print("Removing expired track {}".format(tracked_blobs[i]['id']))
#                    prev_speed = np.mean(tracked_blobs[i]['speed'])
                    prev_speed = ave_speed
                    final_ave_speed = 0.0
                    del tracked_blobs[i]
````

## 11. Printa os resultados
````py
        # ################ PRINTA OS BLOBS ####################################
        for blob in tracked_blobs:  # Desenha os pontos centrais
            for (a, b) in t.pairwise(blob['trail']):
                if SHOW_TRAIL:
                    cv2.circle(frame, a, 3, t.BLUE, -1)
                    cv2.line(frame, a, b, t.WHITE, 1)

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
#                print('========prev_final_ave_speed==', float("{0:.5f}".format(final_ave_speed)))
                if ave_speed == prev_speed and final_ave_speed != 1:
                    final_ave_speed = ave_speed
                    print('===== final_ave_speed =====', float("{0:.5f}".format(final_ave_speed)))
#                    cv2.imwrite('img/{}speed_{}.png'.format(frameCount,final_ave_speed), frame)

                # ############### FIM PRINTA OS BLOBS  ########################
                # MOSTRA RESULTADOS DAS VELOCIDADES CALCULADAS ################
                if blob['trail'][0][0] < r(571): # entao esta na faixa 1
                    lane = 1
                    t.show_results_on_screen(frame, frameCount, ave_speed, lane, blob, total_cars,
                                             RESIZE_RATIO, VIDEO, dict_lane1, dict_lane2,
                                             dict_lane3, SAVE_FRAME_F1, SAVE_FRAME_F2, SAVE_FRAME_F3)
                elif blob['trail'][0][0] >= r(571) and blob['trail'][0][0] < r(1143):  # faixa 2
                    lane = 2
                    t.show_results_on_screen(frame, frameCount, ave_speed, lane, blob, total_cars,
                                             RESIZE_RATIO, VIDEO, dict_lane1, dict_lane2,
                                             dict_lane3, SAVE_FRAME_F1, SAVE_FRAME_F2, SAVE_FRAME_F3)
                elif blob['trail'][0][0] >= r(1143):  # entao esta na faixa 3
                    lane = 3
                    t.show_results_on_screen(frame, frameCount, ave_speed, lane, blob, total_cars,
                                             RESIZE_RATIO, VIDEO, dict_lane1, dict_lane2,
                                             dict_lane3, SAVE_FRAME_F1, SAVE_FRAME_F2, SAVE_FRAME_F3)

                # SALVA VALORES DE VELOCIDADE EM ARQUIVOS CSV
                # Ta ruim, nao tá salvando certinho
````
### 12. Salva os resultados em CSV
- Não da pra confiar muito bem nesses resultados, tenho que rever
````py
                t.save_real_speed_in_csv(total_cars, dict_lane1, real_speed_lane1, dict_lane2,
                                         real_speed_lane2, dict_lane3, real_speed_lane3)
                # Ta ruim, nao tá salvando certinho
                t.save_mea_speed_in_csv(blob, total_cars, prev_len_speed, final_ave_speed, lane,
                                        dict_lane1, meas_speed_lane1, dict_lane2, meas_speed_lane2,
                                        dict_lane3, meas_speed_lane3)

        print('**************************************************************')
        if SHOW_FRAME_COUNT:
            PERCE = str(int((100*frameCount)/vehicle['videoframes']))
            cv2.putText(frame, f'frame: {frameCount} {PERCE}%', (r(14), r(1071)), 0, .65, t.WHITE, 2)
````

### 13. Mostra os resultados de cada operação
````py
##
        # ########## MOSTRA OS VIDEOS  ########################################
#        cv2.imshow('equ', equ)
#        cv2.imshow('res', res)
#        cv2.imshow('fgmask', fgmask)
#        cv2.imshow('erodedmask',erodedmask)
#        cv2.imshow('dilatedmask', dilatedmask)
#        cv2.imshow('contornos',contornos)
#        cv2.imshow('out',out)
        cv2.imshow('frame', frame)
#        final = np.hstack((erodedmask, dilatedmask))
#        cv2.imshow('final', final)
#        cv2.imshow('mask_eroded', np.concatenate((fgmask, dilatedmask),0))
#        crop_img = outputFrame[70:320, 0:640]
````
### 14. Salva os frames em imagens
- Uso quando preciso ver o que está acontecendo em um determinado intervalo
````py
       if frameCount > 40 and frameCount < 281:
           cv2.imwrite('img/{}.png'.format(frameCount), frame)
           cv2.imwrite('img/teste/{}.png'.format(frameCount), np.vstack((out,frame)))
````

### 15. Fim do código
````py
        frameCount += 1    # Atualiza a quantidade de Frames
        if frameCount == CLOSE_VIDEO:  # fecha o video de acordo com o 
        							   # frame escolhido no começo do código
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Tecla Q para fechar
            break
    else:  # sai do while: ret == False
        break
cap.release()
cv2.destroyAllWindows()
````

![frame](https://github.com/Brockzera/SpeedTCC/blob/master/backup/imagens/1frame254.png)
![2framegray254](https://github.com/Brockzera/SpeedTCC/blob/master/backup/imagens/2framegray254.png)
![3roi254](https://github.com/Brockzera/SpeedTCC/blob/master/backup/imagens/3roi254.png)
![4hist254](https://github.com/Brockzera/SpeedTCC/blob/master/backup/imagens/4hist254.png)

![5fgmask254](https://github.com/Brockzera/SpeedTCC/blob/master/backup/imagens/5fgmask254.png)
![6erodedmask254](https://github.com/Brockzera/SpeedTCC/blob/master/backup/imagens/6erodedmask254.png)
![7dilatedmask254](https://github.com/Brockzera/SpeedTCC/blob/master/backup/imagens/7dilatedmask254.png)
![8out254](https://github.com/Brockzera/SpeedTCC/blob/master/backup/imagens/8out254.png)
![final](https://github.com/Brockzera/SpeedTCC/blob/master/backup/imagens/final.png)
