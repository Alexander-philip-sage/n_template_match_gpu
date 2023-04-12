#module load conda/2022-09-08
#conda activate mb_aligner
import numpy as np
from test_cases import BEADS_TEST_CASES_CCOEFF, STUFF_TEST_CASES_CCOEFF, METHODS, get_test_data, find_match_location
import glob
from PIL import Image, ImageOps
import os
import time

from numba import jit, njit

@njit
def template_match(image:np.ndarray, template:np.ndarray, method:str)->np.ndarray:
  w = template.shape[0]
  h = template.shape[1]
  diff_dim1 = image.shape[0]-template.shape[0]+1
  diff_dim2 = image.shape[1]-template.shape[1]+1
  ret = np.zeros((diff_dim1,diff_dim2), dtype=np.float64)
  total_steps = diff_dim1*diff_dim2
  if method =='cv2.TM_CCOEFF':
    template = np.subtract(template,(np.sum(template)/(template.shape[0]*template.shape[1])))
  for i in range(diff_dim1):
    for j in range(diff_dim2):
      #step = i*diff_dim2 +j+1
      #if (step)%15000==0:
      #  print("step", step)
      ret[i,j]=np.sum(np.multiply(image[i:i+w,j:j+h], template))
  return ret

if __name__=='__main__':
  #test_data= glob.glob(os.path.join('*.jpg'))
  image_fname, method_name, start_dim1, start_dim2, templ_width =get_test_data(STUFF_TEST_CASES_CCOEFF, 0)
  image = np.asarray(ImageOps.grayscale(Image.open(image_fname)), dtype=np.float64)
  ##indexing a numpy array passes a reference not a copy
  template = image[start_dim1:start_dim1+templ_width, start_dim2:start_dim2+templ_width].copy()
  print("image type",type(image))
  print("image path", image_fname)
  print(f"image shape {image.shape}")
  print("method", method_name) 
  print(f"point of template, template size ({start_dim1}, {start_dim2}, {templ_width})")
  print("running the function twice since the first time it runs, it compiles")
  print("only timing the second run")
  assert (image.shape[0]>template.shape[0])
  assert (image.shape[1]>template.shape[1])  
  assert method_name in ['cv2.TM_CCOEFF', 'cv2.TM_CCORR'], 'other methods not implemented'
  template_match(image, template, method_name)
  start = time.time()
  res = template_match(image, template, method_name)
  print("total seconds: ", time.time()-start) 
  location = find_match_location(res, method_name)
  print("found max at", location)
