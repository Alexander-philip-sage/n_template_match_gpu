import sys
import cv2
import numpy as np
import pickle
import os
from typing import Tuple
from test_cases import TestCase
import glob
import pandas as pd
import time
from test_cases import crop_template_search_window
def timing_scaling():
    timing_data = []
    verbose=False
    print("\ntiming opencv-cpu implementation")
    image_fname = "search8000x8000.png"
    with open("test_cases.pickle", 'rb') as fileobj:
        test_cases = pickle.load(fileobj)    
    image_path=image_fname
    #image_path = os.path.join("/eagle/BrainImagingML/apsage/n_template_match_gpu",image_fname)
    image = cv2.imread(image_path, 0)
    ##indexing a numpy array passes a reference not a copy
    for test_case in test_cases:
        template, search_window = crop_template_search_window(test_case, image)
        res = cv2.matchTemplate(search_window,template,cv2.TM_CCOEFF)
        max_loc = np.unravel_index(res.argmax(), res.shape)
        max_loc = (max_loc[0]+test_case.image_loc[0],max_loc[1]+test_case.image_loc[1])
        correct=True
        if max_loc!=(test_case.template_loc[0], test_case.template_loc[1]):
            print("opencv incorrect location")
            correct=False
        N = 10
        start = time.time()
        for i in range(N):
            res = cv2.matchTemplate(search_window,template,cv2.TM_CCOEFF)
        pair_time = (time.time()-start)/N
        print("opencv single image-template pair: ",test_case.template_size, test_case.image_size, correct, pair_time, 's' ) 
        
        timing_data.append(['opencv-cpu',test_case.template_size, test_case.image_size, correct, pair_time])
    time_df = pd.DataFrame(timing_data, columns=['algorithm', 'template_size', 'search_window_size', 'accuracy', 'time'])
    time_df.to_csv("tm_timing_opencv_cpu.csv", index=False)

def time_N_pairs():
    print("\ntiming opencv-cpu  N-pairs implementation")
    image_fname = "search8000x8000.png"
    with open("test_cases.pickle", 'rb') as fileobj:
        test_cases = pickle.load(fileobj)    
    image_path=image_fname
    #image_path = os.path.join("/eagle/BrainImagingML/apsage/n_template_match_gpu",image_fname)
    image = cv2.imread(image_path, 0)
    pair_scaling=[]
    for i , test_case in enumerate(test_cases):
        j = i+1
        template, search_window = crop_template_search_window(test_case, image)
        start = time.time()
        for i in range(j):
            res = cv2.matchTemplate(search_window,template,cv2.TM_CCOEFF)
        match_time = (time.time()-start)/j
        pair_scaling.append(['opencv',test_case.template_size, test_case.image_size,j, match_time ])
    pair_scaling_df = pd.DataFrame(pair_scaling, columns=['algorithm', 'template_size', 'search_window_size','N-pairs', 'time'])
    pair_scaling_df.to_csv("tm_timing_N_opencv.csv", index=False)

if __name__=='__main__':
    timing_scaling()
    time_N_pairs()