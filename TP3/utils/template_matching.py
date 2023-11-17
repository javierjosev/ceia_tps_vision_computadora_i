import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt



def mult_template_matching_canning_with_temp_resizing(template_bgr, image_bgr, method, threshold, debug=False):

    # Converting image & template to grayscale
    image = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
    # detect edges in the grayscale image        
    edged = cv.Canny(image, 50, 200)
    
    template = cv.cvtColor(template_bgr, cv.COLOR_BGR2GRAY)

    # found keeps track of the region and scale of the image with the best match.
    # found initialization
    found = []

    # image & template dimensions
    image_height, image_width = image.shape
    template_height, template_width = template.shape

    # scale the template to the image size without losing aspect ratio
    scaling_factor = min(image_width / template_width, image_height / template_height)
    template = cv.resize(template, None, fx=scaling_factor, fy=scaling_factor)

    # loop over the scales of the image
    for scale in np.linspace(0.1, 1.0, 20)[::-1]:
        template_resized = cv.resize(template, (int(template.shape[1] * scale), int(template.shape[0] * scale)))
        tH, tW = template_resized.shape

        # # improving template_resized contrast
        # template_resized = cv.equalizeHist(template_resized)

        template_resized = cv.Canny(template_resized, 50, 200)

        # matching to find the template in the image
        result = cv.matchTemplate(edged, template_resized, eval(method))
        
        (_, maxVal, _, maxLoc) = cv.minMaxLoc(result)

        # check to see if the iteration should be visualized
        if debug:
            # draw a bounding box around the detected region
            clone = np.dstack([edged, edged, edged])
            cv.rectangle(clone, (maxLoc[0], maxLoc[1]), (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
            cv.putText(clone, f"maxVal: {maxVal}", (10,20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            maxVal_found, _, _, _ = found
            cv.putText(clone, f"maxVal found: {maxVal_found}", (10,40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv.imshow(f"Visualizing iteration", clone)
            cv.imshow("Resized template", template_resized)
            cv.waitKey(0)
        
        # if we have found a new maximum correlation value, the found variable is updated
        if maxVal > threshold:
            # found = (maxVal, maxLoc, r)
            found.append([maxVal, maxLoc, tW, tH])
    
    # gets the found values and draws the result bounding box
    
    clone_result = np.dstack(cv.split(image_bgr))

    for finding in found:
        (maxVal, maxLoc, tW, tH) = finding
        cv.rectangle(clone_result, (maxLoc[0], maxLoc[1]), (maxLoc[0] + tW, maxLoc[1] + tH), (255, 0, 255), 2)
        cv.putText(clone_result, f"maxVal found: {'%.3f'%(maxVal)}", (maxLoc[0], maxLoc[1]+15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    cv.imshow(f"Visualizing Result", clone_result)

    cv.waitKey(0)
    cv.destroyAllWindows()




def template_matching_canning_with_temp_resizing(template_bgr, image_bgr, method, debug=False):

    # Converting image & template to grayscale
    image = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
    # detect edges in the grayscale image        
    edged = cv.Canny(image, 50, 200)
    
    template = cv.cvtColor(template_bgr, cv.COLOR_BGR2GRAY)

    # found keeps track of the region and scale of the image with the best match.
    # found initialization
    found = (0, 0, 0, 0)

    # image & template dimensions
    image_height, image_width = image.shape
    template_height, template_width = template.shape

    # scale the template to the image size without losing aspect ratio
    scaling_factor = min(image_width / template_width, image_height / template_height)
    template = cv.resize(template, None, fx=scaling_factor, fy=scaling_factor)

    # loop over the scales of the image
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        template_resized = cv.resize(template, (int(template.shape[1] * scale), int(template.shape[0] * scale)))
        tH, tW = template_resized.shape

        # # improving template_resized contrast
        # template_resized = cv.equalizeHist(template_resized)

        template_resized = cv.Canny(template_resized, 50, 200)

        # matching to find the template in the image
        result = cv.matchTemplate(edged, template_resized, eval(method))
        
        (_, maxVal, _, maxLoc) = cv.minMaxLoc(result)

        # check to see if the iteration should be visualized
        if debug:
            # draw a bounding box around the detected region
            clone = np.dstack([edged, edged, edged])
            cv.rectangle(clone, (maxLoc[0], maxLoc[1]), (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
            cv.putText(clone, f"maxVal: {maxVal}", (10,20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            maxVal_found, _, _, _ = found
            cv.putText(clone, f"maxVal found: {maxVal_found}", (10,40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
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




###### AUXILIARY & TEST Methods #################


def template_matching_canning_with_resizing(template, image, visualize=True):

    # Converting image & template to grayscale
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # volver atrás
    template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

    template = cv.Canny(template, 50, 150)
    if visualize:
        cv.imshow("Template", template)

    #(tH, tW) = template.shape[:2]
    tW, tH = template.shape[::-1]

    # found keeps track of the region and scale of the image with the best match.
    found = None

    # verifying sizes and redimensioning if necessary..
    if template.shape[0] > image.shape[0] or template.shape[1] > image.shape[1]:
        # Redimensionar el template para que se ajuste a la imagen
        # template = cv.resize(template, (image.shape[1], image.shape[0]))
        # TODO implement!!!
        print("El tamaño del template es mayor que el tamaño de la imagen.") 
        return     
    
    # loop over the scales of the image
    for scale in np.linspace(0.2, 1.0, 40)[::-1]:
        # resize the image according to the scale, and keep track of the ratio of the resizing
        # resized = imutils.resize(image, width = int(image.shape[1] * scale))
        resized = cv.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))

        r = image.shape[1] / float(resized.shape[1])

        # if the resized image is smaller than the template, then break from the loop
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break
            
        # detect edges in the resized, grayscale image and apply template
        # matching to find the template in the image
        edged = cv.Canny(resized, 50, 150)
        result = cv.matchTemplate(edged, template, cv.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv.minMaxLoc(result)

        # check to see if the iteration should be visualized
        if visualize:
            # draw a bounding box around the detected region
            clone = np.dstack([edged, edged, edged])
            cv.rectangle(clone, (maxLoc[0], maxLoc[1]),
                (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
            cv.imshow("Visualize", clone)
            cv.waitKey(0)
        
        # if we have found a new maximum correlation value, then update
        # the found variable
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)
    
    # unpack the bookkeeping variable and compute the (x, y) coordinates
    # of the bounding box based on the resized ratio
    (_, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
    # draw a bounding box around the detected result and display the image
    cv.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv.imshow("Image", image)
    cv.waitKey(0)
    cv.destroyAllWindows()



def single_match_template(template, img, methods):

    img_rgb= cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_gray= cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    w, h = template.shape[::-1] 

    for meth in methods:
        # Hago una copia de la imagen porque ciclo a ciclo le dibujo rectángulos
        img_salida = img_rgb.copy()
        
        method = eval(meth)
        
        # Aplicamos la coincidencia de patrones
        #--------------------------------------
        res = cv.matchTemplate(img_gray, template, method)
        
        # Encontramos los valores máximos y mínimos
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        
        # Si el método es TM_SQDIFF o TM_SQDIFF_NORMED, tomamos el mínimo
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        
        # Marcamos el lugar donde lo haya encontrado
        #----------------------------------------
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv.rectangle(img_salida,top_left, bottom_right, 255, 2)
        
        # Graficamos el procesamiento y la salida
        #----------------------------------------
        plt.figure()
        
        # Resultado de coincidencia
        plt.subplot(121),plt.imshow(res,cmap = 'gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        
        # Imagen original con recuadros
        plt.subplot(122),plt.imshow(img_salida)
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        
        plt.suptitle(meth)
        plt.show()

