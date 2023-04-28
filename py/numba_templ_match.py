#qsub -A BrainImagingML -q debug -l select=1 -l walltime=00:60:00 -k doe -l filesystems=home:eagle -N numba_templ numba_templ_match.sh
import numpy as np
from test_cases import BEADS_TEST_CASES_CCOEFF, STUFF_TEST_CASES_CCOEFF, METHODS, get_test_data, find_match_location
from test_cases import COORD_START_DIM1, COORD_START_DIM2, COORD_HEIGHT, COORD_WIDTH, generate_coords
import glob
from PIL import Image, ImageOps
import os
import time

from numba import  jit,njit, prange

@njit(parallel=True)
def np_template_match(image:np.ndarray, template:np.ndarray, res:np.ndarray, method:str)->None:
  w = template.shape[0]
  h = template.shape[1]
  diff_dim1 = image.shape[0]-template.shape[0]+1
  diff_dim2 = image.shape[1]-template.shape[1]+1
  template = np.subtract(template,(np.sum(template)/(template.shape[0]*template.shape[1])))
  for i in prange(diff_dim1):
    for j in prange(diff_dim2):
      res[i,j]=np.sum(np.multiply(image[i:i+w,j:j+h], template))

@njit(parallel=True)
def einsum_template_match(image:np.ndarray, template:np.ndarray, res:np.ndarray, method:str)->None:
  w = template.shape[0]
  h = template.shape[1]
  diff_dim1 = image.shape[0]-template.shape[0]+1
  diff_dim2 = image.shape[1]-template.shape[1]+1
  template = np.subtract(template,(np.sum(template)/(template.shape[0]*template.shape[1])))
  for i in prange(diff_dim1):
    for j in prange(diff_dim2):
      image_window = image[i:i+w,j:j+h]
      for ti in range(template.shape[0]):
        for tj in range(template.shape[1]):
          res[i,j] +=  image_window[ti,tj]*template[ti,tj]

def time_single_run(image, image_fname, template, res, method_name, start_dim1, start_dim2, templ_width, verbose=False, implementation='einsum'):
  if implementation=='einsum':
    start = time.time()
    einsum_template_match(image, template, res, method_name)
    ein_time = time.time()-start
    print("einsum_implementation")
    print("total seconds - single pair: ",ein_time ) 
  elif implementation=='np':
    start = time.time()
    np_template_match(image, template, res, method_name)
    np_time = time.time()-start
    print("np_multiply_implementation")
    print("total seconds - single pair: ",np_time ) 
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

def time_single_run_adjusted(image:np.ndarray, image_fname:str,im_coord:np.ndarray, template,templ_coord, res, method_name, verbose=False):
  start = time.time()
  einsum_template_match(image, template, res, method_name)
  end = time.time()
  location = find_match_location(res, method_name)
  location = (location[0]+im_coord[COORD_START_DIM1], location[1]+im_coord[COORD_START_DIM2])
  failed = False
  if (location[0]!=templ_coord[COORD_START_DIM1]) or (location[1]!=templ_coord[COORD_START_DIM2] ):
    verbose=True
    failed = True
  if verbose:
    print("image type",type(image))
    print("image path", image_fname)
    print(f"image shape {image.shape}")
    print("method", method_name) 
    print(f"image coordinates {im_coord}")
    print(f"point of template on image, template size ({templ_coord[COORD_START_DIM1]},",
          f"{templ_coord[COORD_START_DIM2]}, {template.shape})")
    print("found max at", location)
  return failed , end-start

def main():
  print("\ntiming numba implementation")
  #test_data= glob.glob(os.path.join('*.jpg'))
  image_fname, method_name, start_dim1, start_dim2, templ_width =get_test_data(STUFF_TEST_CASES_CCOEFF, 6)
  #image_path = os.path.join("/Users/apsage/Documents/n_template_match_gpu/",image_fname)
  image_path = os.path.join("/eagle/BrainImagingML/apsage/n_template_match_gpu",image_fname)
  #image_path = image_fname
  image = np.asarray(ImageOps.grayscale(Image.open(image_path)), dtype=np.float32)
  ##indexing a numpy array passes a reference not a copy
  template = image[start_dim1:start_dim1+templ_width, start_dim2:start_dim2+templ_width].copy()
  diff_dim1 = image.shape[0]-template.shape[0]+1
  diff_dim2 = image.shape[1]-template.shape[1]+1
  res = np.zeros((diff_dim1,diff_dim2), dtype=np.float32)
  assert (image.shape[0]>template.shape[0])
  assert (image.shape[1]>template.shape[1])  
  assert method_name in ['cv2.TM_CCOEFF'], 'other methods not implemented'
  print("running the function twice since the first time it runs, it compiles")
  print("only timing the second run")
  einsum_template_match(image, template, res, method_name)
  np_template_match(image, template, res, method_name)
  time_single_run(image, image_fname, template, res, method_name, start_dim1, start_dim2, templ_width,  implementation='einsum')
  time_single_run(image, image_fname, template, res, method_name, start_dim1, start_dim2, templ_width, implementation='np')
  print(start_dim1, start_dim2)
class Coordinate():
  def __init__(self, start_dim1, start_dim2,height, width, offset=None):
    self.start_dim1 = start_dim1
    self.start_dim2 = start_dim2
    self.height = height
    self.width = width
    if offset:
      self.offset = offset
  def __repr__(self):
    return f"start dim1 {self.start_dim1}, start dim2 {self.start_dim2}, height {self.height}, width {self.width}"

def n_templates():
  print("\ntiming multi-image-templ-pair numba implementation")
  image_fname, method_name, start_dim1, start_dim2, templ_width =get_test_data(STUFF_TEST_CASES_CCOEFF, 0)
  image_path = os.path.join("/eagle/BrainImagingML/apsage/n_template_match_gpu/",image_fname)
  image_path = os.path.join("/Users/apsage/Documents/n_template_match_gpu/",image_fname)
  #image_path = image_fname
  ##setup image, template and coordinates
  image = np.asarray(ImageOps.grayscale(Image.open(image_path)), dtype=np.float32)
  print("size of base image", image.shape)
  ##indexing a numpy array passes a reference not a copy
  template = image[start_dim1:start_dim1+templ_width, start_dim2:start_dim2+templ_width].copy()
  ct_test_cases = len(STUFF_TEST_CASES_CCOEFF[1])
  template_coords = np.zeros((ct_test_cases,4),dtype=np.int16)
  image_coords= np.zeros((ct_test_cases,5),dtype=np.int16)
  generate_coords(ct_test_cases,STUFF_TEST_CASES_CCOEFF, image, template_coords, image_coords)
  #print("templ coordinates", Coordinate(start_dim1, start_dim2, templ_width, templ_width))
  #print("image coords", Coordinate(i_start_dim1, i_start_dim2, i_height, i_width))
  print("number of test cases", len(template_coords))
  diff_dim1 = image.shape[0]-template.shape[0]+1
  diff_dim2 = image.shape[1]-template.shape[1]+1
  res = np.zeros((diff_dim1,diff_dim2), dtype=np.float32)
  assert (image.shape[0]>template.shape[0])
  assert (image.shape[1]>template.shape[1])  
  assert method_name in ['cv2.TM_CCOEFF'], 'other methods not implemented'
  print("running the function twice since the first time it runs, it compiles")
  print("only timing the second run")  
  einsum_template_match(image, template, res, method_name)
  timing =0 
  incorrect = 0 
  for i in range(len(template_coords)):
    tm_start_dim1 = template_coords[i][COORD_START_DIM1]
    tm_height = template_coords[i][COORD_HEIGHT]
    tm_start_dim2 = template_coords[i][COORD_START_DIM2]
    tm_width = template_coords[i][COORD_WIDTH]
    tm = image[tm_start_dim1:tm_start_dim1+tm_height,
               tm_start_dim2:tm_start_dim2+tm_width].copy()
    im_start_dim1 = image_coords[i][COORD_START_DIM1]
    im_height = image_coords[i][COORD_HEIGHT]
    im_start_dim2 = image_coords[i][COORD_START_DIM2]
    im_width = image_coords[i][COORD_WIDTH]
    im = image[im_start_dim1:im_start_dim1+im_height,
               im_start_dim2:im_start_dim2+im_width]
    res= np.zeros((diff_dim1,diff_dim2), dtype=np.float32)
    ret_bl, time_passed = time_single_run_adjusted(im, image_fname,image_coords[i], tm, template_coords[i], res, method_name)
    timing += time_passed
    if ret_bl:
      incorrect+=1
  if incorrect:
    print(incorrect, " incorrect results")
  print("seconds/template-image pair: ", (timing)/ct_test_cases) 

if __name__=='__main__':
  ##cannot figure out why this is incorrect
  n_templates()
  main()