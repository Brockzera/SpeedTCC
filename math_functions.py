
def resize(number, ratio):  
    return int(number*ratio)


def get_center_of_rectangle(pt1, pt2):
    x,y = pt1[0], pt1[1]
    w,h = pt2[0], pt2[1]
    return (int(x + w/2), int(y + h/2))


if __name__ == '__main__':
    print('arquivo math_functions executado. \n Esse arquivo possui apenas funcoes')