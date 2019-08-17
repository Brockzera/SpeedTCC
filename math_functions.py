
import cv2


def resize(number, ratio):  
    return int(number*ratio)


def get_center_of_rectangle(pt1, pt2):
    x,y = pt1[0], pt1[1]
    w,h = pt2[0], pt2[1]
    return (int(x + w/2), int(y + h/2))


def calculate_speed(trails, fps, ratio, correction_factor):
    med_area_meter = 3.9  # metros (Valor estimado)
    med_area_pixel = resize(485,ratio)
    qntd_frames =  11 #len(trails)  # default 11
    dist_pixel = cv2.norm(trails[0], trails[10])  
    dist_meter = dist_pixel*(med_area_meter/med_area_pixel)
    speed = (dist_meter*3.6*correction_factor)/(qntd_frames*(1/fps))
    return speed


if __name__ == '__main__':
    print('arquivo math_functions executado. \n Esse arquivo possui apenas funcoes')