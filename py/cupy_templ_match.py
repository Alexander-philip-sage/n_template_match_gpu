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
    N=10
    for test_case in test_cases:
        template, search_window = crop_template_search_window(test_case, image)
        start = time.time()
        search_window_gpu = cupy.array(search_window)
        im_tm = cupy.array(im_tm)
        gpu_memalloc_time = time.time()-start
        search_window_f = cufp.fftn(search_window_gpu)
        fft_time = time.time()-start
        res = do_cupy(search_window_f,search_window.shape,template)
        max_loc = np.unravel_index(res.argmax(), res.shape)
        max_loc = (max_loc[0]+test_case.image_loc[0],max_loc[1]+test_case.image_loc[1])
        correct=True
        if max_loc!=(test_case.template_loc[0], test_case.template_loc[1]):
            print("cupy incorrect location")
            correct=False        
        start = time.time()
        for i in range(N):
            res = do_cupy(search_window_f,search_window.shape,template)
        match_time = (time.time()-start)/N
        timing_data.append(['cupy',test_case.template_size, test_case.image_size, correct, (gpu_memalloc_time+fft_time+match_time),fft_time, gpu_memalloc_time,match_time])
    time_df = pd.DataFrame(timing_data, columns=['algorithm', 'template_size', 'search_window_size', 'accuracy', 'time', 'fft_time_image', 'gpu_memalloc_time', match_time])
    time_df.to_csv("tm_timing_cupy.csv", index=False)

    test_case = test_cases[2]
    pair_scaling=[]
    for j in range(1,N*2):
        template, search_window = crop_template_search_window(test_case, image)
        start = time.time()
        search_window_gpu = cupy.array(search_window)
        im_tm = cupy.array(im_tm)
        gpu_memalloc_time = time.time()-start
        search_window_f = cufp.fftn(search_window_gpu)
        fft_time = time.time()-start
        start = time.time()
        for i in range(j):
            res = do_cupy(search_window_f,search_window.shape,template)
        match_time = (time.time()-start)/j
        pair_scaling.append(['cupy',test_case.template_size, test_case.image_size,j, (gpu_memalloc_time+fft_time+match_time),fft_time, gpu_memalloc_time,match_time])
    pair_scaling_df = pd.DataFrame(pair_scaling, columns=['algorithm', 'template_size', 'search_window_size','N-pairs', 'time', 'fft_time_image', 'gpu_memalloc_time', 'match_time'])
    pair_scaling_df.to_csv("tm_timing_N_cupy.csv", index=False)

