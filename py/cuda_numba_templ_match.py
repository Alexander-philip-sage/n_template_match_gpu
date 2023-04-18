#qsub -A BrainImagingML -q debug -l select=1 -l walltime=00:60:00 -k doe -l filesystems=home:eagle -N cuda_templ_match cuda_templ_match.sh
#supported numpy numba cuda functions http://numba.pydata.org/numba-doc/latest/cuda/cudapysupported.html
import numpy as np
from test_cases import BEADS_TEST_CASES_CCOEFF, STUFF_TEST_CASES_CCOEFF, METHODS, get_test_data, find_match_location
from test_cases import COORD_START_DIM1, COORD_START_DIM2, COORD_HEIGHT, COORD_WIDTH, generate_coords
import glob
from PIL import Image, ImageOps
import os
import time
import math

from numba import  cuda, jit

#@cuda.jit(device=True)


@cuda.jit
def convolv_2d(image:np.ndarray, template:np.ndarray, res:np.ndarray)->None:
  '''it will run a matrix multiplication on 
  each thread instead of a single scalar multiplication, but if we did all the matrix multiplication on individual threads, 
  then saved the results to a lookup data structure, 
  we would then need the same size for loop to loop through the data structure 
  summing the results into the result array - not saving any time'''
  cell_sum = 0
  # Thread id in a 1D block
  tx = cuda.threadIdx.x
  # Block id in a 1D grid
  ty = cuda.blockIdx.x
  # Block width, i.e. number of threads per block
  bw = cuda.blockDim.x
  ##global index
  gi= tx + ty * bw
  ri = gi//res.shape[1]
  rj = gi%res.shape[1]
  if ri < res.shape[0] and rj < res.shape[1]:
    for ti in range(template.shape[0]):
      for tj in range(template.shape[1]):
        cell_sum+=image[ri+ti,rj+tj]*template[ti,tj]
    res[ri,rj] = cell_sum
  
def template_match_host(image:np.ndarray, template:np.ndarray, res:np.ndarray, method:str)->None:
  '''each thread will get one template image pair and comput the matrix
  multiplication on it. The convolution is made up of the same number of 
  threads as there are cells in the final matrix. A single convolution on a 
  single template image pair is performed here'''
  ##do on cpu
  template = np.subtract(template,(np.sum(template)/(template.shape[0]*template.shape[1])))
  ##copy to device
  d_image = cuda.to_device(image)
  d_template = cuda.to_device(template)
  d_res = cuda.device_array(res.shape, dtype=res.dtype)
  threadsperblock = 32
  blockspergrid = math.ceil(res.shape[0]*res.shape[1]/threadsperblock)
  convolv_2d[blockspergrid, threadsperblock](d_image, d_template, d_res)
  cuda.synchronize()
  ##copy res back to host
  d_res.copy_to_host(res)

def template_match():
  verbose=False
  print("timing numba implementation")
  #test_data= glob.glob(os.path.join('*.jpg'))
  image_fname, method_name, start_dim1, start_dim2, templ_width =get_test_data(STUFF_TEST_CASES_CCOEFF, 0)
  image_path = image_fname
  #image_path = os.path.join("/eagle/BrainImagingML/apsage/n_template_match_gpu",image_fname)
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
  template_match_host(image, template, res, method_name)
  start = time.time()
  N = 10
  for i in range(N):
    template_match_host(image, template, res, method_name)
  end = time.time()
  print("total seconds - single image-template pair: ", (end-start)/N) 
  FLOPS_thread = template.shape[0]*template.shape[1]
  thread_ct = res.shape[0]*res.shape[1]
  print("thread count", thread_ct)
  print("GFLOPS/s", (FLOPS_thread*thread_ct*1e-9)*N/(end-start))
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

@cuda.jit
def n_convolv_2d(image:np.ndarray, template:np.ndarray, res:np.ndarray, image_coords:np.ndarray, template_coords:np.ndarray)->None:
  '''it will run a matrix multiplication on 
  each thread instead of a single scalar multiplication, but if we did all the matrix multiplication on individual threads, 
  then saved the results to a lookup data structure, 
  we would then need the same size for loop to loop through the data structure 
  summing the results into the result array - not saving any time'''
  cell_sum = 0
  # Thread id in a 1D block
  tx = cuda.threadIdx.x
  # Block id in a 1D grid
  ty = cuda.blockIdx.x
  # Block width, i.e. number of threads per block
  bw = cuda.blockDim.x
  ##global thread index
  gi= tx + ty * bw
  threads_p_pair = res.shape[1]*res.shape[2]
  pair_index = gi//threads_p_pair
  convol_index = gi%threads_p_pair
  ri = convol_index//res.shape[2]
  rj = convol_index%res.shape[2]
  if (ri < res.shape[1] and rj < res.shape[2]) and (pair_index < res.shape[0]):
    im_coord = image_coords[pair_index]
    sub_im = image[im_coord[COORD_START_DIM1]:im_coord[COORD_START_DIM1]+im_coord[COORD_HEIGHT], im_coord[COORD_START_DIM2]:im_coord[COORD_START_DIM2]+im_coord[COORD_WIDTH]]
    tm_coord = template_coords[pair_index]
    sub_tm = template[tm_coord[COORD_START_DIM1]:tm_coord[COORD_START_DIM1]+tm_coord[COORD_HEIGHT], tm_coord[COORD_START_DIM2]:tm_coord[COORD_START_DIM2]+tm_coord[COORD_WIDTH]]
    for ti in range(template.shape[0]):
      for tj in range(template.shape[1]):
        cell_sum+= sub_im[ri+ti,rj+tj]*sub_tm[ti,tj]
    res[pair_index,ri,rj] = cell_sum

def n_template_match_host(image, template, res, template_coords, image_coords):
  '''each thread will get one template image pair and comput the matrix
  multiplication on it. The convolution is made up of the same number of 
  threads as there are cells in the final matrix. N image-template paris will be
  processed in parallel across various threads'''
  ##do on cpu
  template = np.subtract(template,(np.sum(template)/(template.shape[0]*template.shape[1])))
  ##copy to device
  d_image    = cuda.to_device(image)
  d_template = cuda.to_device(template)
  d_res      = cuda.device_array(res.shape, dtype=res.dtype)
  d_image_coords    = cuda.to_device(image_coords)
  d_template_coords      = cuda.to_device(template_coords)
  ct_total_threads = res.shape[0]*res.shape[1]*res.shape[2]
  if res.shape[1]*res.shape[2]<1024:
    threadsperblock = res.shape[1]*res.shape[2]
  else:
    threadsperblock=1024
  #blockspergrid = (ct_total_threads+threadsperblock-1)//threadsperblock
  blockspergrid = math.ceil(ct_total_threads/threadsperblock)
  n_convolv_2d[blockspergrid, threadsperblock](d_image, d_template, d_res, d_image_coords, d_template_coords)
  cuda.synchronize()
  ##copy res back to host
  d_res.copy_to_host(res)

def n_template_match():
  print("timing multi-image-templ-pair cuda-numba implementation")
  image_fname, method_name, start_dim1, start_dim2, templ_width =get_test_data(STUFF_TEST_CASES_CCOEFF, 0)
  image_path = image_fname
  image = np.asarray(ImageOps.grayscale(Image.open(image_path)), dtype=np.float64)
  template = image[start_dim1:start_dim1+templ_width, start_dim2:start_dim2+templ_width].copy()
  ct_test_cases = len(STUFF_TEST_CASES_CCOEFF[1])
  template_coords = np.zeros((ct_test_cases,4),dtype=np.int16)
  image_coords= np.zeros((ct_test_cases,5),dtype=np.int16)
  generate_coords(ct_test_cases,STUFF_TEST_CASES_CCOEFF, image, template_coords, image_coords)  
  diff_dim1 = image.shape[0]-template.shape[0]+1
  diff_dim2 = image.shape[1]-template.shape[1]+1
  res = np.zeros((ct_test_cases,diff_dim1,diff_dim2), dtype=np.float64)
  assert (image.shape[0]>template.shape[0])
  assert (image.shape[1]>template.shape[1])  
  assert method_name in ['cv2.TM_CCOEFF'], 'other methods not implemented'
  image_copy = image.copy()
  n_template_match_host(image, image_copy, res, template_coords, image_coords)
  start = time.time()
  n_template_match_host(image, image_copy, res, template_coords, image_coords)
  print("thread count", res.shape[0]*res.shape[1]*res.shape[2])
  print("thread per convolution", res.shape[1]*res.shape[2])
  print("number of template-image pairs", ct_test_cases)
  print("seconds/template-image pair: ", (time.time()-start)/ct_test_cases) 
  for ri in range(res.shape[0]):
    im_coord = image_coords[ri]
    templ_coord = template_coords[ri]
    location = find_match_location(res[ri], method_name)
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

if __name__=='__main__':
  template_match()
