import cv2 as cv
import numpy as np


def chromatic_coordinates_correction(image):
    # Divide la imagen en los canales B, G y R
    b, g, r = cv.split(image.astype(float))

    # Suma de los canales por cada píxel
    total_sum = b + g + r

    # Evita la división por cero
    total_sum[total_sum == 0] = 1.0

    # Transformación a cada canal
    corrected_b = np.clip(b / total_sum, 0, 1) * 255
    corrected_g = np.clip(g / total_sum, 0, 1) * 255
    corrected_r = np.clip(r / total_sum, 0, 1) * 255

    # Combina los canales corregidos
    corrected_image = cv.merge([corrected_b, corrected_g, corrected_r]).astype(np.uint8)

    return corrected_image


def white_patch_correction(image):
    # Divide la imagen en los canales B, G y R
    b, g, r = cv.split(image.astype(float))

    # Calcula los valores máximos para cada canal en la imagen
    max_b = np.max(b)
    max_g = np.max(g)
    max_r = np.max(r)

    # Evita la división por cero
    if max_b == 0:
        max_b = 1.0
    if max_g == 0:
        max_g = 1.0
    if max_r == 0:
        max_r = 1.0

    # Aplica la transformación a cada canal
    corrected_b = (255 / max_b) * b
    corrected_g = (255 / max_g) * g
    corrected_r = (255 / max_r) * r

    # Combina los canales corregidos
    corrected_image = cv.merge([corrected_b, corrected_g, corrected_r]).astype(np.uint8)

    return corrected_image

