import sys
import cv2
import numpy as np
import os
from test_cases import TestCase
import pickle
def crop_region(image, size, point):
    '''returns the point on the image for the search window'''
    assert size <= image.shape[0]
    assert size <= image.shape[1]
    offset = size//2
    if (point[0]-offset)>=0:
        i_start_dim1 = (point[0]-offset) 
    else:
        i_start_dim1 = 0
        offset =offset+(offset-point[0])
    if (point[1]-offset)>=0:
        i_start_dim2 = (point[1]-offset) 
    else:
        i_start_dim2 = 0
        offset =offset+(offset-point[1])
    i_start = (i_start_dim1, i_start_dim2)
    return i_start

def find_scaling_test_cases():
    test_cases = []
    image_fname = "search8000x8000.png"
    image_path = image_fname
    #image_path = os.path.join("/eagle/BrainImagingML/apsage/n_template_match_gpu",image_fname)
    image = cv2.imread(image_path, 0)    
    print("creating test cases based on opencv correct results")
    image_sizes = [500*(i+1) for i in range(8)]
    template_sizes = [200*(i+1) for i in range(20)]
    for img_size in image_sizes:
        for tmpl_size in template_sizes:
            
            if img_size > tmpl_size:
                save=False
                ct=0
                while (not save) and (ct < 10):
                    verbose=False
                    ct+=1
                    #pick points somewhat in the center
                    dim1 = np.random.randint(low=1000, high=image.shape[0]-1000)
                    dim2 = np.random.randint(low=1000, high=image.shape[0]-1000)
                    loc = (dim1, dim2)
                    tmpl = image[dim1:dim1+tmpl_size, dim2:dim2+tmpl_size].copy()
                    search_window_loc = crop_region(image, img_size, loc)
                    search_window = image[search_window_loc[0]:search_window_loc[0]+img_size,search_window_loc[1]:search_window_loc[1]+img_size].copy()
                    res = cv2.matchTemplate(search_window,tmpl,cv2.TM_CCOEFF)
                    max_loc = np.unravel_index(res.argmax(), res.shape)
                    max_loc = (max_loc[0]+search_window_loc[0],max_loc[1]+search_window_loc[1])
                    if max_loc==(dim1, dim2):
                        save = True
                        test_cases.append(TestCase(loc,search_window_loc, tmpl_size, img_size, image_fname))
    print("test cases generated", len(test_cases))
    with open("test_cases.pickle", 'wb') as fileobj:
        pickle.dump(test_cases,fileobj)

def find_same_sizes(template_size:int, image_size:int, ct_test_cases:int):

    test_cases = []
    image_fname = "search8000x8000.png"
    image_path = image_fname
    #image_path = os.path.join("/eagle/BrainImagingML/apsage/n_template_match_gpu",image_fname)
    image = cv2.imread(image_path, 0)    
    print("creating test cases based on opencv correct results")
    for i in range(ct_test_cases):
        if image_size > template_size:
            save=False
            ct=0
            while (not save) and (ct < 10):
                verbose=False
                ct+=1
                #pick points somewhat in the center
                dim1 = np.random.randint(low=1000, high=image.shape[0]-1000)
                dim2 = np.random.randint(low=1000, high=image.shape[0]-1000)
                loc = (dim1, dim2)
                tmpl = image[dim1:dim1+template_size, dim2:dim2+template_size].copy()
                search_window_loc = crop_region(image, image_size, loc)
                search_window = image[search_window_loc[0]:search_window_loc[0]+image_size,search_window_loc[1]:search_window_loc[1]+image_size].copy()
                res = cv2.matchTemplate(search_window,tmpl,cv2.TM_CCOEFF)
                max_loc = np.unravel_index(res.argmax(), res.shape)
                max_loc = (max_loc[0]+search_window_loc[0],max_loc[1]+search_window_loc[1])
                if max_loc==(dim1, dim2):
                    save = True
                    test_cases.append(TestCase(loc,search_window_loc, template_size, image_size, image_fname))
    print("test cases generated", len(test_cases))
    with open(f"test_cases_{template_size}_{image_size}.pickle", 'wb') as fileobj:
        pickle.dump(test_cases,fileobj)

if __name__=='__main__':
    find_same_sizes(400, 1000, 20)
    #find_scaling_test_cases()
