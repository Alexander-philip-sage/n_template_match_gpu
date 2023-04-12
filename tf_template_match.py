import tensorflow as tf
import tensorflow.math as math
import time
import numpy as np
from test_cases import BEADS_TEST_CASES_CCOEFF, STUFF_TEST_CASES_CCOEFF, METHODS, get_test_data, find_match_location
import glob
from PIL import Image, ImageOps

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

if __name__=='__main__':
  image_fname, method_name, start_dim1, start_dim2, templ_width =get_test_data(STUFF_TEST_CASES_CCOEFF, 0)
  image = np.asarray(ImageOps.grayscale(Image.open(image_fname)))
  ##indexing a numpy array passes a reference not a copy
  template = image[start_dim1:start_dim1+templ_width, start_dim2:start_dim2+templ_width].copy()
  assert (image.shape[0]>template.shape[0])
  assert (image.shape[1]>template.shape[1])  
  print("image type",type(image))
  print("image path", image_fname)
  print(f"image shape {image.shape}")
  print("method", method_name) 
  print(f"point of template, template size ({start_dim1}, {start_dim2}, {templ_width})")
  assert method_name in ['cv2.TM_CCOEFF', 'cv2.TM_CCORR'], 'other methods not implemented'
  start = time.time()
  res=tf_template_match(image, template, method_name)
  print("total seconds: ", time.time()-start) 
  location = find_match_location(res, method_name)
  print("found max at", location)