#qsub -A BrainImagingML -q debug -l select=1 -l walltime=00:60:00 -k doe -l filesystems=home:eagle -N numba_templ numba_templ_match.sh
import numpy as np
from test_cases import BEADS_TEST_CASES_CCOEFF, STUFF_TEST_CASES_CCOEFF, METHODS, get_test_data, find_match_location
from test_cases import COORD_START_DIM1, COORD_START_DIM2, COORD_HEIGHT, COORD_WIDTH, generate_coords
import glob
from PIL import Image, ImageOps
import os
import time

from numba import  njit, prange

@njit(parallel=True)
def template_match(image:np.ndarray, template:np.ndarray, res:np.ndarray, method:str)->None:
  w = template.shape[0]
  h = template.shape[1]
  diff_dim1 = image.shape[0]-template.shape[0]+1
  diff_dim2 = image.shape[1]-template.shape[1]+1
  template = np.subtract(template,(np.sum(template)/(template.shape[0]*template.shape[1])))
  for i in prange(diff_dim1):
    for j in prange(diff_dim2):
      res[i,j]=np.sum(np.multiply(image[i:i+w,j:j+h], template))

def time_single_run(image, image_fname, template, res, method_name, start_dim1, start_dim2, templ_width, verbose=False):
  start = time.time()
  template_match(image, template, res, method_name)
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

def time_single_run_adjusted(image:np.ndarray, image_fname:str,im_coord:np.ndarray, template,templ_coord, res, method_name, verbose=False):
  template_match(image, template, res, method_name)
  location = find_match_location(res, method_name)
  location = (location[0]+im_coord[COORD_START_DIM1], location[1]+im_coord[COORD_START_DIM2])
  if location!=(templ_coord[COORD_START_DIM1],templ_coord[COORD_START_DIM2] ):
    verbose=True
  if verbose:
    print("image type",type(image))
    print("image path", image_fname)
    print(f"image shape {image.shape}")
    print("method", method_name) 
    print(f"image coordinates {im_coord}")
    print(f"point of template on image, template size ({templ_coord[COORD_START_DIM1]}, {templ_coord[COORD_START_DIM2]}, {template.shape})")
    print("found max at", location)
  return True

def main():
  print("timing numba implementation")
  #test_data= glob.glob(os.path.join('*.jpg'))
  image_fname, method_name, start_dim1, start_dim2, templ_width =get_test_data(STUFF_TEST_CASES_CCOEFF, 7)
  #image_path = os.path.join("/Users/apsage/Documents/n_template_match_gpu/",image_fname)
  image_path = os.path.join("/eagle/BrainImagingML/apsage/n_template_match_gpu",image_fname)
  image = np.asarray(ImageOps.grayscale(Image.open(image_path)), dtype=np.float64)
  ##indexing a numpy array passes a reference not a copy
  template = image[start_dim1:start_dim1+templ_width, start_dim2:start_dim2+templ_width].copy()
  diff_dim1 = image.shape[0]-template.shape[0]+1
  diff_dim2 = image.shape[1]-template.shape[1]+1
  res = np.zeros((diff_dim1,diff_dim2), dtype=np.float64)
  assert (image.shape[0]>template.shape[0])
  assert (image.shape[1]>template.shape[1])  
  assert method_name in ['cv2.TM_CCOEFF'], 'other methods not implemented'
  print("running the function twice since the first time it runs, it compiles")
  print("only timing the second run")
  template_match(image, template, res, method_name)
  time_single_run(image, image_fname, template, res, method_name, start_dim1, start_dim2, templ_width)
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
  print("timing multi-image-templ-pair numba implementation")
  print("iteration over image-templ paris is not parallelized with numba")
  image_fname, method_name, start_dim1, start_dim2, templ_width =get_test_data(STUFF_TEST_CASES_CCOEFF, 0)
  image_path = os.path.join("/eagle/BrainImagingML/apsage/n_template_match_gpu/",image_fname)
  #image_path = os.path.join("/Users/apsage/Documents/n_template_match_gpu/",image_fname)
  #image_path = image_fname
  ##setup image, template and coordinates
  image = np.asarray(ImageOps.grayscale(Image.open(image_path)), dtype=np.float64)
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
  res = np.zeros((diff_dim1,diff_dim2), dtype=np.float64)
  assert (image.shape[0]>template.shape[0])
  assert (image.shape[1]>template.shape[1])  
  assert method_name in ['cv2.TM_CCOEFF'], 'other methods not implemented'
  print("running the function twice since the first time it runs, it compiles")
  print("only timing the second run")  
  template_match(image, template, res, method_name)
  start = time.time()
  for i in range(len(template_coords)):
    tm = image[template_coords[i][COORD_START_DIM1]:template_coords[i][COORD_START_DIM1]+template_coords[i][COORD_HEIGHT],
               template_coords[i][COORD_START_DIM2]:template_coords[i][COORD_START_DIM2]+template_coords[i][COORD_WIDTH]]
    im = image[image_coords[i][COORD_START_DIM1]:image_coords[i][COORD_START_DIM1]+image_coords[i][COORD_HEIGHT],
               image_coords[i][COORD_START_DIM2]:image_coords[i][COORD_START_DIM2]+image_coords[i][COORD_WIDTH]]
    ret_bl = time_single_run_adjusted(im, image_fname,image_coords[i], tm, template_coords[i], res, method_name)
  print("seconds/template-image pair: ", (time.time()-start)/ct_test_cases) 

if __name__=='__main__':
  n_templates()
