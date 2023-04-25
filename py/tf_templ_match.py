import tensorflow as tf
import tensorflow.math as math
import time
import numpy as np
from test_cases import BEADS_TEST_CASES_CCOEFF, STUFF_TEST_CASES_CCOEFF, METHODS, get_test_data, find_match_location
import glob
from PIL import Image, ImageOps
import os

def avg_pixel_val(template:np.ndarray):
  return tf.cast(math.reduce_sum(template), tf.float32)/tf.cast(math.reduce_prod(template.shape),tf.float32)
def one_avg_pixel_val_test():
  assert  7.0==avg_pixel_val(np.ones((3,3))*7).numpy()
one_avg_pixel_val_test()

def tf_template_match(image:np.ndarray, template:np.ndarray, method:str, verbose=False):

  tf_template = tf.convert_to_tensor(
                template, dtype=tf.float32, name='template')
  if method=='cv2.TM_CCOEFF':
    tf_template = math.subtract(tf_template,avg_pixel_val(tf_template))
  tf_image = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(
              image, dtype=tf.float32, name='image'),2), 0) 
  tf_e_template =tf.expand_dims(tf.expand_dims(tf_template, 2),3) 
  res=tf.nn.conv2d(tf_image, tf_e_template,
            strides=[1,1,1,1],
            padding="VALID")
  if verbose:
    print("before squeeze",res.shape)
  res = tf.squeeze(
          tf.squeeze(
          res, axis=0, name=None
          ), axis=2, name=None
  )
  if verbose:
    print("after squeeze", res.shape)
  return res

def templ_match():
  verbose=False
  print("timing tf conv2d implementation")
  image_fname, method_name, start_dim1, start_dim2, templ_width =get_test_data(STUFF_TEST_CASES_CCOEFF, 0)
  #image_path = os.path.join("/eagle/BrainImagingML/apsage/n_template_match_gpu",image_fname)
  image_path = image_fname
  image = np.asarray(ImageOps.grayscale(Image.open(image_path)), dtype=np.float32)
  ##indexing a numpy array passes a reference not a copy
  template = image[start_dim1:start_dim1+templ_width, start_dim2:start_dim2+templ_width].copy()
  assert (image.shape[0]>template.shape[0])
  assert (image.shape[1]>template.shape[1])  
  assert method_name in ['cv2.TM_CCOEFF', 'cv2.TM_CCORR'], 'other methods not implemented'
  N = 10
  start = time.time()
  for i in range(N):
    res=tf_template_match(image, template, method_name).numpy()
  end = time.time()
  print("total seconds - tf single image-template pair: ", (end-start)/N) 
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

##
##batch versions
##
def prep_batch_template(repeat = 0):
  start_template = time.time()

  image_fname = STUFF_TEST_CASES_CCOEFF[0]
  image_path = os.path.join("/eagle/BrainImagingML/apsage/n_template_match_gpu",image_fname)
  image_path = image_fname
  image = np.asarray(ImageOps.grayscale(Image.open(image_path)), dtype=np.float32)
  test_cases = STUFF_TEST_CASES_CCOEFF[1]
  cases = []
  for tc in test_cases:
    if tc[0]=='cv2.TM_CCOEFF' and tc[3]==200:
      for i in range(repeat+1):
        cases.append(tc)
  batch = np.zeros((cases[0][3],cases[0][3],1,len(cases)), dtype=np.float32)
  for i in range(len(cases)):
    tc = cases[i]
    batch[:,:,0,i] = image[tc[1]:, tc[2]:][:tc[3], :tc[3]].copy()
  time.time()
  template_time = time.time() - start_template
  print("time to prepare template per template", template_time)
  print(" per template", template_time/len(cases))
  return image, batch, cases 
def tf_batch_conv(image:np.ndarray, batch_templates:np.ndarray, verbose: bool = False):
  start_variable_seup=time.time()
  tf_image = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(
              image, dtype=tf.float32, name='image'),2), 0) 
  memory_time = time.time() - start_variable_seup
  start_compute = time.time()
  ccoeff_coef = (math.reduce_sum(batch_templates,axis=(0,1), keepdims=True)/(batch_templates.shape[0]*batch_templates.shape[1]))
  print("ccoeff shape",ccoeff_coef.shape)
  batch_templates = math.subtract(batch_templates, ccoeff_coef)
  #print("batch_templates", type(batch_templates.numpy()), batch_templates.shape)
  #print("tf_image", type(tf_image.numpy()), tf_image.shape)
  res=tf.nn.conv2d(tf_image, batch_templates,
            strides=[1,1,1,1],
            padding="VALID")
  end_compute = time.time()
  compute_time = end_compute - start_compute
  if verbose:
    print("batch template shape", batch_templates.shape)
    print('memory allocation time:', memory_time)
    print("image shape", tf_image.shape)
    print("batch template shape", batch_templates.shape)
    print("compute time:", compute_time)
    print("compute time per frame:", compute_time/batch_templates.shape[-1])
    print("total time per frame", (end_compute - start_variable_seup)/batch_templates.shape[-1])
    print("result shape", res.shape)
  return res, batch_templates.shape[-1], compute_time, memory_time
def batch_main():
  image, batch_template, cases=prep_batch_template(repeat=2)
  method_name = cases[0][0]
  res, ct_frames, compute_time, memory_time = tf_batch_conv(image, batch_template, verbose=False)
  time.sleep(2)
  start=time.time()
  res, ct_frames, compute_time, memory_time = tf_batch_conv(image, batch_template, verbose=True)
  #plt.imshow(res)
  end = time.time()
  print("total seconds: ", end-start)
  result =res.numpy()
  all_correct = True
  for i in range(result.shape[3]):
    x,y=find_match_location(result[0,:,:,i], method_name)
    start_dim1 , start_dim2 = cases[i][1:3]
    if start_dim1!= x or start_dim2!=y:
      print('correct',start_dim1, start_dim2 )
      print("found", x,y)
      all_correct=False
  if all_correct:
    print("correct location")  

if __name__=='__main__':
  batch_main()