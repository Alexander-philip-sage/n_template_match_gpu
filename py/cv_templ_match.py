import sys
import cv2
import numpy as np
import os
from typing import Tuple
import glob
import time
from test_cases import BEADS_TEST_CASES_CCOEFF, STUFF_TEST_CASES_CCOEFF, METHODS, get_test_data, find_match_location

if __name__=='__main__':
    verbose=False
    print("timing opencv-cpu implementation")
    image_fname, method_name, start_dim1, start_dim2, templ_width =get_test_data(STUFF_TEST_CASES_CCOEFF, 0)
    image_path = os.path.join("/eagle/BrainImagingML/apsage/n_template_match_gpu",image_fname)
    image = cv2.imread(image_path, 0)
    ##indexing a numpy array passes a reference not a copy
    template = image[start_dim1:start_dim1+templ_width, start_dim2:start_dim2+templ_width].copy()
    assert (image.shape[0]>template.shape[0])
    assert (image.shape[1]>template.shape[1])  
    assert method_name in ['cv2.TM_CCOEFF', 'cv2.TM_CCORR'], 'other methods not implemented'
    start = time.time()
    method = eval(method_name)
    res = cv2.matchTemplate(image,template,method)
    print("total seconds: ", time.time()-start) 
    location = find_match_location(res, method_name)
    if location==(start_dim1, start_dim2):
        print("found correct location")
    else:
        verbose=True
    if verbose:
        print("image type",type(image))
        print("image path", image_fname)
        print(f"image shape {image.shape}")
        print("method", method_name) 
        print(f"point on template, template size ({start_dim1}, {start_dim2}, {templ_width})")
        print("found max at", location)
