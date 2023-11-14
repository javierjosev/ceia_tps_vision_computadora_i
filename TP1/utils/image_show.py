import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def cv_image_show(image):
    cv.imshow('Image', image)
    # El programa cierra la ventana al presionar cualquier tecla
    cv.waitKey(0) 
    cv.destroyAllWindows()


def pyplot_image_show(image):

    if len(image.shape) == 2:
        # Imagen en escala de grises
        plt.imshow(image, cmap='gray')
    else:
        # Imagen a color
        plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))

    plt.axis('off')  # Oculta los ejes
    plt.show()


def cv_image_compare(image1, image2, title1='Image 1', title2='Image 2'):
    # Concatena horizontalmente las dos imágenes
    concatenated_images = np.hstack((image1, image2))

    # Muestra las imágenes concatenadas
    cv.imshow(title1 + ' y ' + title2, concatenated_images)

    # Espera a que se presione una tecla y luego cierra la ventana
    cv.waitKey(0)
    cv.destroyAllWindows()


def pyplot_image_compare(image1, image2, title1='Image 1', title2='Image 2'):
    # Configura el diseño de la figura con dos subgráficos en una fila
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Muestra la primera imagen en el primer subgráfico
    if len(image1.shape) == 2:
        # Imagen en escala de grises
        axs[0].imshow(image1, cmap='gray')
    else:
        # Imagen a color
        axs[0].imshow(cv.cvtColor(image1, cv.COLOR_BGR2RGB))
    # axs[0].imshow(cv.cvtColor(image1, cv.COLOR_BGR2RGB))
    axs[0].axis('off')
    axs[0].set_title(title1)

    # Muestra la segunda imagen en el segundo subgráfico
    if len(image2.shape) == 2:
        # Imagen en escala de grises
        axs[1].imshow(image2, cmap='gray')
    else:
        # Imagen a color
        axs[1].imshow(cv.cvtColor(image2, cv.COLOR_BGR2RGB))
    #axs[1].imshow(cv.cvtColor(image2, cv.COLOR_BGR2RGB))
    axs[1].axis('off')
    axs[1].set_title(title2)

    # Ajusta el diseño y muestra la figura
    plt.tight_layout()
    plt.show()


def pyplot_hsv_image_channels(img_color):

    img_HSV = cv.cvtColor(img_color, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(img_HSV)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(h, cmap='gray')
    axs[0].set_title('Hue Channel')

    axs[1].imshow(s, cmap='gray')
    axs[1].set_title('Saturation Channel')

    axs[2].imshow(v, cmap='gray')
    axs[2].set_title('Value Channel')

    plt.tight_layout()
    plt.show()