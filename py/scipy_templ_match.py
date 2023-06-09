import scipy.fftpack as scfp
import pandas as pd
import time
from PIL import Image, ImageOps
from test_cases import crop_template_search_window
import numpy as np
from test_cases import TestCase
import pickle

def do_scipy(F,im,im_tm):
    '''F fourier transformed image
    im image in numpy
    im_tm template image in numpy
    returns matrix of result'''
    F_tm = scfp.fftn(im_tm, shape=im.shape)
    # compute the best match location
    F_cc = F * np.conj(F_tm)
    c = (scfp.ifftn(F_cc/np.abs(F_cc))).real
    return c
def timing_scaling():
    timing_data = []
    verbose=False
    print("\ntiming scipy implementation")
    image_fname = "search8000x8000.png"
    with open("test_cases.pickle", 'rb') as fileobj:
        test_cases = pickle.load(fileobj)    
    image_path=image_fname
    #image_path = os.path.join("/eagle/BrainImagingML/apsage/n_template_match_gpu",image_fname)
    image = np.asarray(ImageOps.grayscale(Image.open(image_path)), dtype=np.float32)
    for test_case in test_cases:
        template, search_window = crop_template_search_window(test_case, image)
        start = time.time()
        search_window_f = scfp.fftn(search_window)
        fft_time = time.time()-start
        res = do_scipy(search_window_f,search_window,template)
        max_loc = np.unravel_index(res.argmax(), res.shape)
        max_loc = (max_loc[0]+test_case.image_loc[0],max_loc[1]+test_case.image_loc[1])
        correct=True
        if max_loc!=(test_case.template_loc[0], test_case.template_loc[1]):
            print("scipy incorrect location")
            correct=False
        N = 10
        start = time.time()
        for i in range(N):
            res = do_scipy(search_window_f,search_window,template)
        match_time = (time.time()-start)/N
        print("scipy single image-template pair: ",test_case.template_size, test_case.image_size, correct, match_time+fft_time, 's' ) 
        
        timing_data.append(['scipy',test_case.template_size, test_case.image_size, correct, match_time+fft_time,fft_time , match_time])
    time_df = pd.DataFrame(timing_data, columns=['algorithm', 'template_size', 'search_window_size', 'accuracy', 'time', 'fft_time_image', "match_time"])
    time_df.to_csv("tm_timing_scipy.csv", index=False)

def time_N_pairs():
    print("\ntiming scipy N-pairs implementation")
    image_fname = "search8000x8000.png"
    with open("test_cases_400_1000.pickle", 'rb') as fileobj:
        test_cases = pickle.load(fileobj)    
    image_path=image_fname
    #image_path = os.path.join("/eagle/BrainImagingML/apsage/n_template_match_gpu",image_fname)
    image = np.asarray(ImageOps.grayscale(Image.open(image_path)), dtype=np.float32)
    pair_scaling=[]
    for i, test_case in enumerate(test_cases):
        j = i+1
        template, search_window = crop_template_search_window(test_case, image)
        start = time.time()
        search_window_f = scfp.fftn(search_window)
        fft_time = time.time()-start
        start = time.time()
        for i in range(j):
            res = do_scipy(search_window_f,search_window,template)
        match_time = (time.time()-start)/j
 
        pair_scaling.append(['scipy',test_case.template_size, test_case.image_size,j, match_time+fft_time,fft_time , match_time ])
    pair_scaling_df = pd.DataFrame(pair_scaling, columns=['algorithm', 'template_size', 'search_window_size','N-pairs', 'time', 'fft_time_image', "match_time"])
    pair_scaling_df.to_csv("tm_timing_N_scipy.csv", index=False)


if __name__=='__main__':
    timing_scaling()
    time_N_pairs()