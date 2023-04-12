#module load conda/2022-09-08
#conda activate mb_aligner
#qsub -A BrainImagingML -q debug -l select=1 -l walltime=00:60:00 -k doe -l filesystems=home:eagle -N numba_templ numba_templ_match.sh
#supported numpy numba cuda functions http://numba.pydata.org/numba-doc/latest/cuda/cudapysupported.html
import numpy as np
from test_cases import BEADS_TEST_CASES_CCOEFF, STUFF_TEST_CASES_CCOEFF, METHODS, get_test_data, find_match_location
import glob
from PIL import Image, ImageOps
import os
import time

from numba import  cuda, njit
#@cuda.jit(device=True)

#threadsperblock = (16, 16)
#blockspergrid_x = np.ceil(an_array.shape[0] / threadsperblock[0])
#blockspergrid_y = np.ceil(an_array.shape[1] / threadsperblock[1])
#blockspergrid = (blockspergrid_x, blockspergrid_y)
#increment_a_2D_array[blockspergrid, threadsperblock](an_array)
@cuda.jit
def increment_a_2D_array(an_array):
    x, y = cuda.grid(2)
    if x < an_array.shape[0] and y < an_array.shape[1]:
       an_array[x, y] += 1
@cuda.jit
def increment_by_one(an_array):
  # Thread id in a 1D block
  tx = cuda.threadIdx.x
  # Block id in a 1D grid
  ty = cuda.blockIdx.x
  # Block width, i.e. number of threads per block
  bw = cuda.blockDim.x
  # Compute flattened index inside the array
  pos = tx + ty * bw
  if pos < an_array.size:  # Check array boundaries
      an_array[pos] += 1

@cuda.jit
def convolv_2d(image:np.ndarray, template:np.ndarray, res:np.ndarray)->None:
  '''this will not be efficient. it will run a matrix multiplication on 
  each thread instead of a single scalar multiplication'''
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
  
@njit
def template_match_host(image:np.ndarray, template:np.ndarray, res:np.ndarray, method:str)->None:
  '''each thread will get one template image pair and comput the matrix
  multiplication on it'''
  template = np.subtract(template,(np.sum(template)/(template.shape[0]*template.shape[1])))
  threadsperblock = 32
  blockspergrid = np.ceil(res.shape[0]*res.shape[1]/threadsperblock)
  convolv_2d[blockspergrid, threadsperblock](image, template, res)

if __name__=='__main__':
  verbose=False
  print("timing numba implementation")
  #test_data= glob.glob(os.path.join('*.jpg'))
  image_fname, method_name, start_dim1, start_dim2, templ_width =get_test_data(STUFF_TEST_CASES_CCOEFF, 0)
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
  start = time.time()
  threadsperblock = 32
  blockspergrid = (an_array.size + (threadsperblock - 1)) // threadsperblock
  increment_by_one[blockspergrid, threadsperblock](an_array)  
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
