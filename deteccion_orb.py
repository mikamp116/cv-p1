
"""Importamos las librerías a utilizar y comprobamos que usamos las versiones correctas de Python y OpenCV"""
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import sys
import math

print("Python version: " + sys.version)
print("OpenCV version: " + cv2.__version__)

"""
Carga de imágenes.
Todas las imágenes de entramiento están en el mismo directorio y tienen un nombre y una extensión similar. Almacenamos
las partes comunes de la ruta para que el proceso de carga sea más fácil. En el método load() se cargarán todas las
imágenes de entrenamiento, mientras que en el softLoad() se cargan únicamente 6 imágenes preseleccionadas
"""

def load():
    return [cv2.imread('train/frontal_' + str(i) + '.jpg', 0) for i in range(1,49)]

def ordenar(list):
    list.sort(key=len)
    ret = list[0:10]
    ret.sort()
    aux = list[10:]
    aux.sort()
    return ret + aux

def load2(dir):
    cur_dir = os.path.abspath(os.curdir)
    files = ordenar(os.listdir(cur_dir + '/' + dir))
    return [cv2.imread('train/' + file, 0) for file in files]

def softLoad():
    train = []
    train.append(cv2.imread('train/frontal_9.jpg', 0))
    train.append(cv2.imread('train/frontal_39.jpg', 0))
    train.append(cv2.imread('train/frontal_43.jpg', 0))
    train.append(cv2.imread('train/frontal_7.jpg', 0))
    train.append(cv2.imread('train/frontal_19.jpg', 0))
    train.append(cv2.imread('train/frontal_26.jpg', 0))
    return train

"""Todas las imágenes de entreamiento (almacenadas en el directorio train) tienen el mismo tamaño y el frontal
está cuadrado, por lo que el centro de todas las imágenes, y también el centro del frontal, se encontrará en el
punto(225, 110), almacenado en la variable centre"""

centre = (225, 110)

"""Definimos unos métodos para calcular el módulo del vector que une un punto p con el centro de la imágen y para
calcular el ángulo que forma un punto p con el centro respecto del eje X en el sentido de las agujas del reloj"""

def calculateModule(p):
    return np.sqrt((centre[0] - p[0]) ** 2 + (centre[1] - p[1]) ** 2)

def calculateAngleToCentre(p):
    return (math.atan2((p[1] - centre[1]), (centre[0] - p[0])) * 180 / math.pi) % 360

