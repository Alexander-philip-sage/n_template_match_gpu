import pandas as pd
import time
from PIL import Image, ImageOps
from test_cases import crop_template_search_window
import numpy as np
from test_cases import TestCase
import pickle
import cupy
import cupyx.scipy.fftpack as cufp

def do_cupy(F,im_shape,im_tm):
    '''F fourier transformed image
    im_shape shape of image
    im_tm template image in numpy
    returns matrix of result'''
    ST = time.time()
    im_tm = cupy.array(im_tm)
    print(f"Time to copy template image to GPU: ", time.time()-ST)
    F_tm = cufp.fftn(im_tm, shape=im_shape)

    # compute the best match location
    F_cc = F * np.conj(F_tm)
    c = (cufp.ifftn(F_cc/np.abs(F_cc))).real
    return c

def timing_test_cases():
    timing_data = []
    verbose=False
    print("\ntiming scipy implementation")
    image_fname = "search8000x8000.png"
    with open("test_cases.pickle", 'rb') as fileobj:
        test_cases = pickle.load(fileobj)    
    image_path=image_fname
    #image_path = os.path.join("/eagle/BrainImagingML/apsage/n_template_match_gpu",image_fname)
    image = np.asarray(ImageOps.grayscale(Image.open(image_path)), dtype=np.float32)
    ##indexing a numpy array passes a reference not a copy
    for test_case in test_cases:
        template, search_window = crop_template_search_window(test_case, image)
        start = time.time()
        search_window_gpu = cupy.array(search_window)
        im_tm = cupy.array(im_tm)
        search_window_f = cufp.fftn(search_window_gpu)
        fft_time = time.time()-start
        do_cupy(search_window_f,search_window.shape,template)
