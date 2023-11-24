import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from utils import image_utils


def mult_template_matching_canning_with_temp_resizing(template_bgr, image_bgr, method, canny_threshold1, canny_threshold2, min_resolution_coeff, threshold, debug=False):

    # If image FM quality is less than 0.7, size is increased
    fm_image_quality = image_utils.fm_image_quality_measure(image_bgr)
    fm_scale = 1.5
    if fm_image_quality < 0.78:
        image_bgr = cv.resize(image_bgr, (int(image_bgr.shape[1] * fm_scale), int(image_bgr.shape[0] * fm_scale)), interpolation=cv.INTER_LINEAR)
        #cv.resize(coca_multi_img, (new_width, new_height), interpolation=cv.INTER_LINEAR)

    # Converting image to grayscale
    image = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
    ## Preprocessing image ###############
    image = cv.GaussianBlur(image, (5, 5), 0)
    image = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX)
    # detect edges in the grayscale image
    image_edged = cv.Canny(image, canny_threshold1, canny_threshold2)
    
    # Converting template to grayscale
    template = cv.cvtColor(template_bgr, cv.COLOR_BGR2GRAY)
    ## Preprocessing template ###############
    template = cv.GaussianBlur(template, (5, 5), 0)
    template = cv.normalize(template, None, 0, 255, cv.NORM_MINMAX)
    # detect edges in the grayscale template
    template_edged = cv.Canny(template, canny_threshold1, canny_threshold2)

    # found keeps track of the region and scale of the image with the best match.
    # found initialization
    found = (0, 0, 0, 0)
    best_result = None

    # image & template dimensions
    image_height, image_width = image.shape
    template_height, template_width = template_edged.shape

    # scale the template to the image size without losing aspect ratio
    scaling_factor = min(image_width / template_width, image_height / template_height)
    template_edged = cv.resize(template_edged, None, fx=scaling_factor, fy=scaling_factor)

    # loop over the scales of the image
    for scale in np.linspace(0.1, 1.0, 20)[::-1]:
        
        template_resized = cv.resize(template_edged, (int(template_edged.shape[1] * scale), int(template_edged.shape[0] * scale)))
        tH_temp, tW_temp = template_resized.shape

        # Breaking out the for loop because of insufficient template resolution. As the image is scaled without losing aspect ratio
        # we can use width or height to calculate the coefficient
        resolution_coeff = tW_temp/template_width
        if resolution_coeff > min_resolution_coeff:
            tH = tH_temp
            tW = tW_temp
        else:
            break

        # matching to find the template in the image
        result = cv.matchTemplate(image_edged, template_resized, eval(method))
        
        (_, maxVal, _, maxLoc) = cv.minMaxLoc(result)

        # check to see if the iteration should be visualized
        if debug:
            # draw a bounding box around the detected region
            clone = np.dstack([image_edged, image_edged, image_edged])
            cv.rectangle(clone, (maxLoc[0], maxLoc[1]), (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
            cv.putText(clone, f"maxVal: {maxVal}", (10,20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            maxVal_found, _, _, _ = found
            cv.putText(clone, f"maxVal found: {maxVal_found}", (10,40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv.putText(clone, f"Scaling factor: {scale}", (10,60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            template_resized_fm = image_utils.fm_image_quality_measure(template_resized)
            cv.putText(clone, f"FM: {template_resized_fm}", (10,80), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv.putText(clone, f"Resolution coeff: {resolution_coeff}", (10,100), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv.imshow(f"Visualizing iteration", clone)
            cv.imshow("Resized template", template_resized)
            cv.waitKey(0)
        
        # if we have found a new maximum correlation value, the found variable is updated
        if maxVal > found[0]:
            # found = (maxVal, maxLoc, r)
            found = (maxVal, maxLoc, tW, tH)
            best_result = result
    
    # gets the found values and draws the result bounding box
    clone_result = np.dstack(cv.split(image_bgr))
    loc = np.where( best_result >= threshold)
    for pt in zip(*loc[::-1]):
        # cv.rectangle(clone_result, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        # cv.rectangle(clone_result, (maxLoc[0], maxLoc[1]), (maxLoc[0] + tW, maxLoc[1] + tH), (255, 0, 255), 2)
        cv.rectangle(clone_result, pt, (pt[0] + tW, pt[1] + tH), (0,0,255), 2)

    cv.imshow(f"Visualizing Result", clone_result)
    cv.waitKey(0)
    cv.destroyAllWindows()



def template_matching_canning_with_temp_resizing(template_bgr, image_bgr, method, canny_threshold1, canny_threshold2, min_resolution_coeff, debug=False):

    # Converting image to grayscale
    image = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
    ## Preprocessing image ###############
    image = cv.GaussianBlur(image, (5, 5), 0)
    image = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX)
    # detect edges in the grayscale image
    image_edged = cv.Canny(image, canny_threshold1, canny_threshold2)
    
    # Converting template to grayscale
    template = cv.cvtColor(template_bgr, cv.COLOR_BGR2GRAY)
    ## Preprocessing template ###############
    template = cv.GaussianBlur(template, (5, 5), 0)
    template = cv.normalize(template, None, 0, 255, cv.NORM_MINMAX)
    # detect edges in the grayscale template
    template_edged = cv.Canny(template, canny_threshold1, canny_threshold2)

    # found keeps track of the region and scale of the image with the best match.
    # found initialization
    found = (0, 0, 0, 0)

    # image & template dimensions
    image_height, image_width = image.shape
    template_height, template_width = template_edged.shape

    # scale the template to the image size without losing aspect ratio
    scaling_factor = min(image_width / template_width, image_height / template_height)
    template_edged = cv.resize(template_edged, None, fx=scaling_factor, fy=scaling_factor)

    # loop over the scales of the image
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        
        template_resized = cv.resize(template_edged, (int(template_edged.shape[1] * scale), int(template_edged.shape[0] * scale)))
        tH, tW = template_resized.shape

        # Breaking out the for loop because of insufficient template resolution. As the image is scaled without losing aspect ratio
        # we can use width or height to calculate the coefficient
        resolution_coeff = tW/template_width
        if resolution_coeff < min_resolution_coeff:
            break

        # matching to find the template in the image
        result = cv.matchTemplate(image_edged, template_resized, eval(method))
        
        (_, maxVal, _, maxLoc) = cv.minMaxLoc(result)

        # check to see if the iteration should be visualized
        if debug:
            # draw a bounding box around the detected region
            clone = np.dstack([image_edged, image_edged, image_edged])
            cv.rectangle(clone, (maxLoc[0], maxLoc[1]), (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
            cv.putText(clone, f"maxVal: {maxVal}", (10,20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            maxVal_found, _, _, _ = found
            cv.putText(clone, f"maxVal found: {maxVal_found}", (10,40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv.putText(clone, f"Scaling factor: {scale}", (10,60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv.imshow(f"Visualizing iteration", clone)
            cv.imshow("Resized template", template_resized)
            cv.waitKey(0)
        
        # if we have found a new maximum correlation value, the found variable is updated
        if maxVal > found[0]:
            # found = (maxVal, maxLoc, r)
            found = (maxVal, maxLoc, tW, tH)
    
    # gets the found values and draws the result bounding box
    (maxVal, maxLoc, tW, tH) = found
    clone_result = np.dstack(cv.split(image_bgr))
    cv.rectangle(clone_result, (maxLoc[0], maxLoc[1]), (maxLoc[0] + tW, maxLoc[1] + tH), (255, 0, 255), 2)
    cv.putText(clone_result, f"maxVal found: {maxVal}", (maxLoc[0], maxLoc[1]+15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    cv.imshow(f"Visualizing Result", clone_result)

    cv.waitKey(0)
    cv.destroyAllWindows()
