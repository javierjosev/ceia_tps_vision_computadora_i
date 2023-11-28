from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def histogram_3D_hsv_plot(img_color):
    
    img_HSV = cv.cvtColor(img_color, cv.COLOR_BGR2HSV)

    #c1, c2, c3 = cv.split(imgRGB)
    h, s, v = cv.split(img_HSV)

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")

    imgRGB = cv.cvtColor(img_color, cv.COLOR_BGR2RGB)
    pixel_colors = imgRGB.reshape((np.shape(imgRGB)[0]*np.shape(imgRGB)[1], 3))
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")
    plt.show()

    return h, s, v


def histogram_bgr_plot(img_color):

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    b, g, r = cv.split(img_color)

    hist1,bins1 = np.histogram(b.ravel(),256,[0,256])
    hist2,bins2 = np.histogram(g.ravel(),256,[0,256])
    hist3,bins3 = np.histogram(r.ravel(),256,[0,256])

    axs[0].set_title("Blue")
    axs[0].plot(hist1, color='blue', label='Blue')
    axs[1].set_title("Green")
    axs[1].plot(hist2, color='green', label='Green')
    axs[2].set_title("Red")
    axs[2].plot(hist3, color='red', label='Red')

    plt.tight_layout()
    plt.show()


def histogram_hsv_plot(img_color):
    img_HSV = cv.cvtColor(img_color, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(img_HSV)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Colormap for channels
    #cmap_hue = plt.get_cmap('hsv')
    cmap_hue = plt.get_cmap('gray')
    cmap_saturation = plt.get_cmap('gray')
    cmap_value = plt.get_cmap('gray')

    hist1, bins1 = np.histogram(h.ravel(), 180, [0, 180])
    hist2, bins2 = np.histogram(s.ravel(), 256, [0, 256])
    hist3, bins3 = np.histogram(v.ravel(), 256, [0, 256])

    # Hue
    axs[0].bar(range(180), hist1, color=cmap_hue((np.arange(180) / 180.0)), alpha=0.7, label='Hue')
    axs[0].set_title("Hue")
    axs[0].set_facecolor('#db9f5a')  # Light orange background

    # Saturation
    axs[1].bar(range(256), hist2, color=cmap_saturation((np.arange(256) / 256.0)), alpha=0.7, label='Saturation')
    axs[1].set_title("Saturation")
    axs[1].set_facecolor('#db9f5a')  # Light orange background

    # Value
    axs[2].bar(range(256), hist3, color=cmap_value((np.arange(256) / 256.0)), alpha=0.7, label='Value')
    axs[2].set_title("Value")
    axs[2].set_facecolor('#db9f5a')  # Light orange background

    plt.tight_layout()
    plt.show()


