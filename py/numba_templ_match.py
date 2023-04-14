#qsub -A BrainImagingML -q debug -l select=1 -l walltime=00:60:00 -k doe -l filesystems=home:eagle -N numba_templ numba_templ_match.sh
import numpy as np
from test_cases import BEADS_TEST_CASES_CCOEFF, STUFF_TEST_CASES_CCOEFF, METHODS, get_test_data, find_match_location
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

def main():
  verbose=False
  print("timing numba implementation")
  #test_data= glob.glob(os.path.join('*.jpg'))
  image_fname, method_name, start_dim1, start_dim2, templ_width =get_test_data(STUFF_TEST_CASES_CCOEFF, 0)
  image_path = os.path.join("/Users/apsage/Documents/n_template_match_gpu/py",image_fname)
  #image_path = os.path.join("/eagle/BrainImagingML/apsage/n_template_match_gpu",image_fname)
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

class Coordinate():
  def __init__(self, start_dim1, start_dim2,height, width):
    self.start_dim1 = start_dim1
    self.start_dim2 = start_dim2
    self.height = height
    self.width = width

def n_templates():
  verbose=False
  print("timing multi-image-templ-pair numba implementation")
  image_fname, method_name, start_dim1, start_dim2, templ_width =get_test_data(STUFF_TEST_CASES_CCOEFF, 0)
  image_path = os.path.join("/Users/apsage/Documents/n_template_match_gpu/py",image_fname)

  ##setup image, template and coordinates
  image = np.asarray(ImageOps.grayscale(Image.open(image_path)), dtype=np.float64)
  ##indexing a numpy array passes a reference not a copy
  template = image[start_dim1:start_dim1+templ_width, start_dim2:start_dim2+templ_width].copy()
  template_coords = []
  image_coords=[]
  for ti in enumerate(len(STUFF_TEST_CASES_CCOEFF)):
    image_fname, method_name, start_dim1, start_dim2, templ_width =get_test_data(STUFF_TEST_CASES_CCOEFF, 0)
    template_coords.append(Coordinate(start_dim1, start_dim2, templ_width, templ_width))
    i_start_dim1 = (start_dim1-(templ_width//2)) if (start_dim1-(templ_width//2))>=0 else 0
    i_start_dim2 = (start_dim2-(templ_width//2)) if (start_dim2-(templ_width//2))>=0 else 0
    i_height = (templ_width+(templ_width//2)) if (templ_width+(templ_width//2))<image.shape[0] else (image.shape[0]-1)
    i_width = (templ_width+(templ_width//2)) if (templ_width+(templ_width//2))<image.shape[1] else (image.shape[1]-1)
    image_coords.append(Coordinate(i_start_dim1, i_start_dim2, i_height, i_width))

  diff_dim1 = image.shape[0]-template.shape[0]+1
  diff_dim2 = image.shape[1]-template.shape[1]+1
  res = np.zeros((diff_dim1,diff_dim2), dtype=np.float64)
  assert (image.shape[0]>template.shape[0])
  assert (image.shape[1]>template.shape[1])  
  assert method_name in ['cv2.TM_CCOEFF'], 'other methods not implemented'
  print("running the function twice since the first time it runs, it compiles")
  print("only timing the second run")  
  template_match(image, template, res, method_name)
  for i in range(len(template_coords)):
    im = 
    templ_coord = template_coords[i]
    tm = image[templ_coord.start_dim1:templ_coord.start_dim1+templ_coord.height,templ_coord.start_dim2:templ_coord.start_dim2+templ_coord.width]
    template_match(im, tm, res, method_name)
if __name__=='__main__':
  main()