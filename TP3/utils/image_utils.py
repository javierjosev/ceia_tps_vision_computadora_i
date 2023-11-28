import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv



def pyplot_image_show(image):

    if len(image.shape) == 2:
        # Imagen en escala de grises
        plt.imshow(image, cmap='gray')
    else:
        # Imagen a color
        plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))

    plt.axis('off')  # Oculta los ejes
    plt.show()
    


def absolute_central_moment(image):
    # Step 1: Calculate the image histogram
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])

    # Step 2: Calculate the mean intensity value (u) of the histogram
    u = np.mean(hist)

    # Unique k gray-level values
    k_values = np.unique(hist)

    # Step 3: Get the number of gray levels (L)
    # L = len(k_values)

    # Step 4: Initialize ACMo to zero
    ACMo = 0

    # Step 5: Calculate ACMo using the formula
    for k in k_values:
        Pk = np.count_nonzero(hist == k) / np.sum(hist)  # Relative frequency of the k-th gray-level
        ACMo += abs(k - u) * Pk

    return ACMo


def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# if the bounding boxes are integers, convert them to floats -- this
	# is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]

	# compute the area of the bounding boxes and grab the indexes to sort
	# (in the case that no probabilities are provided, simply sort on the
	# bottom-left y-coordinate)
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = y2

	# if probabilities are provided, sort on them instead
	if probs is not None:
		idxs = probs

	# sort the indexes
	idxs = np.argsort(idxs)

	# keep looping while some indexes still remain in the indexes list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the index value
		# to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the largest (x, y) coordinates for the start of the bounding
		# box and the smallest (x, y) coordinates for the end of the bounding
		# box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]

		# delete all indexes from the index list that have overlap greater
		# than the provided overlap threshold
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))

	# return only the bounding boxes that were picked
	return boxes[pick].astype("int")