"""Importamos las librerías a utilizar y comprobamos que usamos las versiones correctas de Python y OpenCV"""
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import sys
import math
import time

print("Python version: " + sys.version)
print("OpenCV version: " + cv2.__version__)

"""
Carga de imágenes.
Todas las imágenes de entramiento están en el mismo directorio y tienen un nombre y una extensión similar. Almacenamos
las partes comunes de la ruta para que el proceso de carga sea más fácil. En el método load() se cargarán todas las
imágenes de entrenamiento, mientras que en el softLoad() se cargan únicamente 6 imágenes preseleccionadas
"""


def load2():
    return [cv2.imread('train/frontal_' + str(i) + '.jpg', 0) for i in range(1, 49)]


def ordenar(lst):
    lst.sort(key=len)
    ret = lst[0:10]
    ret.sort()
    aux = lst[10:]
    aux.sort()
    return ret + aux


def load(directory='train'):
    cur_dir = os.path.abspath(os.curdir)
    files = ordenar(os.listdir(cur_dir + '/' + directory))
    return [cv2.imread(directory + '/' + file, 0) for file in files]


def soft_load():
    return [cv2.imread('train/frontal_9.jpg', 0), cv2.imread('train/frontal_39.jpg', 0),
            cv2.imread('train/frontal_43.jpg', 0), cv2.imread('train/frontal_7.jpg', 0),
            cv2.imread('train/frontal_19.jpg', 0), cv2.imread('train/frontal_26.jpg', 0)]


"""Todas las imágenes de entreamiento (almacenadas en el directorio train) tienen el mismo tamaño y el frontal
está cuadrado, por lo que el centro de todas las imágenes, y también el centro del frontal, se encontrará en el
punto(225, 110), almacenado en la variable centre"""

"""Definimos unos métodos para calcular el módulo del vector que une un punto p con el centro de la imágen y para
calcular el ángulo que forma un punto p con el centro respecto del eje X en el sentido de las agujas del reloj"""


def calculate_module(p):
    centre = (225, 110)
    return np.sqrt((centre[0] - p[0]) ** 2 + (centre[1] - p[1]) ** 2)


def calculate_angle_to_centre(p):
    centre = (225, 110)
    return (math.atan2((p[1] - centre[1]), (centre[0] - p[0])) * 180 / math.pi) % 360


"""Más tarde va a ser necesario almacenar información sobre los puntos de interés (key points o kp), por lo que
definimos una clase para ello"""


class Match:
    def __init__(self, module, kp_angle, scale, des_angle):
        self.module = module
        self.kp_angle = kp_angle
        self.scale = scale
        self.des_angle = des_angle

    def get_module(self):
        return self.module

    def get_kp_angle(self):
        return self.kp_angle

    def get_scale(self):
        return self.scale

    def get_des_angle(self):
        return self.des_angle


def train(images, detector):
    """Usaremos una estructura de datos tipo FLANN para almacenar los descriptores"""
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=3, multi_probe_level=1)
    search_params = dict(checks=-1)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    """Almacenamos la información de los puntos de interés del entrenamiento en match_table"""
    match_table = []

    # Bucle de entrenamiento
    for image in images:
        kps, des = detector.detectAndCompute(image, None)
        # image_det = cv2.drawKeypoints(image, kps, None, color=(255,0,255),
        #                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        image_match = [Match(calculate_module(k.pt), calculate_angle_to_centre(k.pt), k.size, k.angle) for k in kps]
        match_table.append(image_match)
        flann.add([des])

    return match_table, flann


def detect(images, detector, match_table, flann, KNN_MATCHES, GAUSSIAN_KERNEL_SIGMA, DEBUG):
    if DEBUG == 1:
        test_kps_table = []
        test_des_table = []
        matrices_votacion = []
    detected_points = []

    for test_image in images:
        kps, des = detector.detectAndCompute(test_image, None)
        if DEBUG == 1:
            test_des_table.append(des)
            test_kps_table.append(kps)
        # sh = cv2.drawKeypoints(test_image, kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        results = flann.knnMatch(des, k=KNN_MATCHES)

        matriz_votacion = np.zeros((int(test_image.shape[0] / 10), int(test_image.shape[1] / 10)), dtype=np.float32)

        for r in results:
            for m in r:
                match = match_table[m.imgIdx][m.trainIdx]
                m_test = kps[m.queryIdx]
                trns = (m_test.size / match.get_scale()) * match.get_module()
                angle = match.get_kp_angle() + match.get_des_angle() - m_test.angle
                x = int((m_test.pt[0] + (trns * math.cos(angle))) / 10)
                y = int((m_test.pt[1] - (trns * math.sin(angle))) / 10)
                if 0 < x < matriz_votacion.shape[1] and 0 < y < matriz_votacion.shape[0]:
                    matriz_votacion[y, x] += 1

        sigma = GAUSSIAN_KERNEL_SIGMA
        ksize = 6 * sigma + 1
        gaussian_kernel_y = cv2.getGaussianKernel(ksize, sigma)
        gaussian_kernel_x = gaussian_kernel_y.T
        gaussian_kernel = gaussian_kernel_y * gaussian_kernel_x
        matriz_filtrada = cv2.filter2D(matriz_votacion, -1, gaussian_kernel)

        if DEBUG == 1:
            matrices_votacion.append(matriz_filtrada)

        z = np.unravel_index(np.argmax(matriz_filtrada, axis=None), matriz_filtrada.shape)
        q = (int(z[1] * 10), int(z[0] * 10))
        detected_points.append(q)

    return detected_points


def draw_points(images, points):
    # cv2.circle(test, (int(z[0]*10), int(z[1]*10)), 15, (255,0,0), thickness=10, lineType=8, shift=0)
    for index in range(len(images)):
        cv2.circle(images[index], points[index], 15, (255, 0, 0), thickness=10, lineType=8, shift=0)
        plt.imshow(images[index])
        plt.title("Imagen " + str(index + 1))
        plt.show()


def main(NUM_KEYPOINTS, SCALE_FACTOR, PYRAMID_LEVELS, KNN_MATCHES, GAUSSIAN_KERNEL_SIGMA, DEBUG=0):
    train_images = load()
    orb = cv2.ORB_create(nfeatures=NUM_KEYPOINTS, scaleFactor=SCALE_FACTOR, nlevels=PYRAMID_LEVELS)
    match_table, flann = train(train_images, orb)
    test_images = load('test')
    # para hacer deteccion de una imagen en concretro, pasar esta imagen en una lista del siguiente modo
    # test_images = [test_images[i]], donde i es el indice de la imagen a testear
    detected_points = detect(test_images, orb, match_table, flann, KNN_MATCHES, GAUSSIAN_KERNEL_SIGMA, DEBUG)
    draw_points(test_images, detected_points)


if __name__ == "__main__":
    NUM_KEYPOINTS = 100
    SCALE_FACTOR = 1.3
    PYRAMID_LEVELS = 4
    KNN_MATCHES = 3
    GAUSSIAN_KERNEL_SIGMA = 2
    DEBUG = 1

    # para ver las matrices de votacion, introducir el parametro DEBUG
    main(NUM_KEYPOINTS, SCALE_FACTOR, PYRAMID_LEVELS, KNN_MATCHES, GAUSSIAN_KERNEL_SIGMA)
