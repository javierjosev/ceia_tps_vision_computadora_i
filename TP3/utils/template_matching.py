import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from utils import image_utils



def template_matching_canning_with_temp_resizing(template_bgr, image_bgr, method, canny_threshold1, canny_threshold2, min_resolution_coeff, debug=False):

    image_temp_scale = 0.5
    image_temp_max_ratio = 3

    brisque_scale = 2
    brisque_min_score = 25

    # Converting image to grayscale
    image = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
    # Converting template to grayscale
    template = cv.cvtColor(template_bgr, cv.COLOR_BGR2GRAY)

    # image & template dimensions
    image_height, image_width = image.shape
    template_height, template_width = template.shape

    # Scaling and quality corrections
    # If the relation between the image/template is more than image_temp_max_ratio times, the image is scaled by image_temp_scale
    if image_width/template_width > image_temp_max_ratio:
        image_bgr = cv.resize(image_bgr, (int(image_bgr.shape[1] * image_temp_scale), int(image_bgr.shape[0] * image_temp_scale)), interpolation=cv.INTER_LINEAR)
        image = cv.resize(image, (int(image.shape[1] * image_temp_scale), int(image.shape[0] * image_temp_scale)), interpolation=cv.INTER_LINEAR)
        image_height, image_width = image.shape
    # If the BRISQUE quality is less than brisque_min_score, the image size scaled by brisque_scale
    brisque_score, _, _, _ = cv.quality.QualityBRISQUE_compute(image, "utils/brisque_model_live.yml", "utils/brisque_range_live.yml")
    if brisque_score < brisque_min_score:
        image_bgr = cv.resize(image_bgr, (int(image_bgr.shape[1] * brisque_scale), int(image_bgr.shape[0] * brisque_scale)), interpolation=cv.INTER_LINEAR)
        image = cv.resize(image, (int(image.shape[1] * brisque_scale), int(image.shape[0] * brisque_scale)), interpolation=cv.INTER_LINEAR)
        image_height, image_width = image.shape

    ## Preprocessing image ###############
    image = cv.GaussianBlur(image, (5, 5), 0)
    image = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX)
    # detect edges in the grayscale image
    image_edged = cv.Canny(image, canny_threshold1, canny_threshold2)
    
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
            cv.putText(clone, f"Resolution coeff: {resolution_coeff}", (10,100), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
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
    cv.rectangle(clone_result, (maxLoc[0], maxLoc[1]), (maxLoc[0] + tW, maxLoc[1] + tH), (255, 0, 0), 3)
    cv.putText(clone_result, f"maxVal found: {round(maxVal,6)}", (maxLoc[0], maxLoc[1]+15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    image_utils.pyplot_image_show(clone_result)
    if debug:
        cv.imshow(f"Visualizing Result", clone_result)
        cv.waitKey(0)
        cv.destroyAllWindows()



def mult_template_matching_canning_with_temp_resizing(template_bgr, image_bgr, method, canny_threshold1, canny_threshold2, min_resolution_coeff, threshold, debug=False):

    image_temp_scale = 0.5
    image_temp_max_ratio = 3

    brisque_scale = 2
    brisque_min_score = 25

    # Converting image to grayscale
    image = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
    # Converting template to grayscale
    template = cv.cvtColor(template_bgr, cv.COLOR_BGR2GRAY)

    # image & template dimensions
    image_height, image_width = image.shape
    template_height, template_width = template.shape

    # Scaling and quality corrections
    # If the relation between the image/template is more than image_temp_max_ratio times, the image is scaled by image_temp_scale
    if image_width/template_width > image_temp_max_ratio:
        image_bgr = cv.resize(image_bgr, (int(image_bgr.shape[1] * image_temp_scale), int(image_bgr.shape[0] * image_temp_scale)), interpolation=cv.INTER_LINEAR)
        image = cv.resize(image, (int(image.shape[1] * image_temp_scale), int(image.shape[0] * image_temp_scale)), interpolation=cv.INTER_LINEAR)
        image_height, image_width = image.shape
    # If the BRISQUE quality is less than brisque_min_score, the image size scaled by brisque_scale
    brisque_score, _, _, _ = cv.quality.QualityBRISQUE_compute(image, "utils/brisque_model_live.yml", "utils/brisque_range_live.yml")
    if brisque_score < brisque_min_score:
        image_bgr = cv.resize(image_bgr, (int(image_bgr.shape[1] * brisque_scale), int(image_bgr.shape[0] * brisque_scale)), interpolation=cv.INTER_LINEAR)
        image = cv.resize(image, (int(image.shape[1] * brisque_scale), int(image.shape[0] * brisque_scale)), interpolation=cv.INTER_LINEAR)
        image_height, image_width = image.shape

    ## Preprocessing image ###############
    image = cv.GaussianBlur(image, (5, 5), 0)
    image = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX)
    # detect edges in the grayscale image
    image_edged = cv.Canny(image, canny_threshold1, canny_threshold2)
    
    ## Preprocessing template ###############
    template = cv.GaussianBlur(template, (5, 5), 0)
    template = cv.normalize(template, None, 0, 255, cv.NORM_MINMAX)
    # detect edges in the grayscale template
    template_edged = cv.Canny(template, canny_threshold1, canny_threshold2)

    template_height, template_width = template_edged.shape

    # found keeps track of the region and scale of the image with the best match.
    # found initialization
    found = (0, 0, 0, 0)
    best_result = None

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
    (maxVal, maxLoc, tW, tH) = found
    clone_result = np.dstack(cv.split(image_bgr))

    (y_points, x_points) = np.where(best_result >= threshold) 
    # initialize our list of rectangles 
    boxes = list() 
    # loop over the starting (x, y)-coordinates again 
    for (x, y) in zip(x_points, y_points): 
        # update our list of rectangles 
        boxes.append((x, y, x + tW, y + tH)) 
    # apply non-maxima suppression to the rectangles this will create a single bounding box 
    boxes = image_utils.non_max_suppression(np.array(boxes)) 
    # loop over the final bounding boxes 
    for (x1, y1, x2, y2) in boxes: 
        # draw the bounding box on the image
        correlation_value = round(float(best_result[y1, x1]),6)
        cv.rectangle(clone_result, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv.putText(clone_result, f"Value found: {correlation_value}", (x1, y1+15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    # if no rectangle is selected because of the threshold, it draws the maximum
    if len(boxes) == 0:
        cv.rectangle(clone_result, (maxLoc[0], maxLoc[1]), (maxLoc[0] + tW, maxLoc[1] + tH), (255, 0, 0), 3)
        cv.putText(clone_result, f"Value found: {round(maxVal,6)}", (maxLoc[0], maxLoc[1]+15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    image_utils.pyplot_image_show(clone_result)
    if debug:
        cv.imshow(f"Visualizing Result", clone_result)
        cv.waitKey(0)
        cv.destroyAllWindows()

