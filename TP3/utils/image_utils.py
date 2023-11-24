import numpy as np


def fm_image_quality_measure(image):
    # Step 1: Compute the Fourier Transform (F) of the image
    F = np.fft.fft2(image)
    # Step 2: Shift the origin of F to the center (Fc)
    Fc = np.fft.fftshift(F)
    # Step 3: Calculate the absolute value of the centered Fourier transform (AF)
    AF = np.abs(Fc)
    # Step 4: Calculate the maximum value of the frequency component in F (M)
    M = np.max(AF)
    # Step 5: Calculate the threshold value (thres)
    threshold = M / 1000
    # Step 6: Count the number of pixels in F whose value is greater than thres (TH)
    Th = np.sum(AF > threshold)
    # Calculate the Image Quality measure (FM)
    FM = Th / (image.shape[0] * image.shape[1])
    return FM


