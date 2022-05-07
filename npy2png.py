import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import os
import cv2 as cv

con_arr = np.load("./brats_npy/npy_pred/pred_t1ce.npy")
for i in range(0,10685):
    arr = con_arr[i,:,:]
    mmin = np.min(arr)
    mmax = np.max(arr)
    for m in range(arr.shape[0]):
        for n in range(arr.shape[1]):
            arr[m,n] = (arr[m,n] - mmin) *255/ (mmax-mmin)
    disp_to_img = cv.resize(arr,(240,240))
    # cv.imshow(disp_to_img)
    cv.imwrite("./brats_npy/t1ce_png1/%d.png"%i,disp_to_img)
    print(i)