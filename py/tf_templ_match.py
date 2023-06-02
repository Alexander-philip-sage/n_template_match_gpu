import tensorflow as tf
import tensorflow.math as math
import pandas as pd
import time
from test_cases import crop_template_search_window
import numpy as np
from test_cases import TestCase
from test_cases import  STUFF_TEST_CASES_CCOEFF,  get_test_data, find_match_location

from PIL import Image, ImageOps
import os
import pickle

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
  #image_path = image_fname
  image_path = os.path.join("/eagle/BrainImagingML/apsage/n_template_match_gpu",image_fname)
  image = np.asarray(ImageOps.grayscale(Image.open(image_path)), dtype=np.float32)
  ##indexing a numpy array passes a reference not a copy
  template = image[start_dim1:start_dim1+templ_width, start_dim2:start_dim2+templ_width].copy()
  assert (image.shape[0]>template.shape[0])
  assert (image.shape[1]>template.shape[1])  
  assert method_name in ['cv2.TM_CCOEFF', 'cv2.TM_CCORR'], 'other methods not implemented'
  N = 10
  res=tf_template_match(image, template, method_name).numpy()
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
  #image_path = image_fname
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
  return image, batch, cases , template_time
def tf_batch_conv(image:np.ndarray, batch_templates:np.ndarray, verbose: bool = False):
  start_variable_seup=time.time()
  tf_image = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(
              image, dtype=tf.float32, name='image'),2), 0) 
  memory_time = time.time() - start_variable_seup
  start_compute = time.time()
  ccoeff_coef = (math.reduce_sum(batch_templates,axis=(0,1), keepdims=True)/(batch_templates.shape[0]*batch_templates.shape[1]))
  batch_templates = math.subtract(batch_templates, ccoeff_coef)
  #print("batch_templates", type(batch_templates.numpy()), batch_templates.shape)
  #print("tf_image", type(tf_image.numpy()), tf_image.shape)
  res=tf.nn.conv2d(tf_image, batch_templates,
            strides=[1,1,1,1],
            padding="VALID")
  end_compute = time.time()
  compute_time = end_compute - start_compute
  if verbose:
    print("ccoeff shape",ccoeff_coef.shape)
    print("image shape", tf_image.shape)
    print("batch template shape", batch_templates.shape)
    print('tf_tensor conversion of image:', memory_time)
    print("compute time:", compute_time)
    print("compute time per frame:", compute_time/batch_templates.shape[-1])
    print("result shape", res.shape)
  return res, batch_templates.shape[-1], compute_time, memory_time

def batch_main():
  image, batch_template, cases, template_time=prep_batch_template(repeat=2)
  method_name = cases[0][0]
  res, ct_frames, compute_time, memory_time = tf_batch_conv(image, batch_template, verbose=False)
  time.sleep(2)
  start=time.time()
  res, ct_frames, compute_time, memory_time = tf_batch_conv(image, batch_template, verbose=True)
  #plt.imshow(res)
  end = time.time()
  print("total time per frame", ((end-start)+template_time)/ct_frames)
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

def timing_test_cases():
    timing_data = []
    verbose=False
    print("\ntiming tf-batch implementation")
    with open("test_cases.pickle", 'rb') as fileobj:
        test_cases = pickle.load(fileobj)    
    image_fname = "search8000x8000.png"
    image_path=image_fname
    #image_path = os.path.join("/eagle/BrainImagingML/apsage/n_template_match_gpu",image_fname)
    #image = cv2.imread(image_path, 0)
    image = np.asarray(ImageOps.grayscale(Image.open(image_path)), dtype=np.float32)
    for test_i,test_case in enumerate(test_cases):
        N = 10
        print("test", test_i)
        template, search_window = crop_template_search_window(test_case, image)
        start = time.time()
        batch_template = np.zeros((template.shape[0],template.shape[1],1,N), dtype=np.float32)
        for i in range(N):
          batch_template[:,:,0,i] = template[:,:].copy()
        mem_templ_time = time.time()-start
        print("memory allocated",mem_templ_time ,'s')
        res_batch, ct_frames, compute_time, memory_time = tf_batch_conv(search_window, batch_template.copy(), verbose=False)
        res = res_batch[0,:,:,0].numpy()
        max_loc = np.unravel_index(res.argmax(), res.shape)
        max_loc = (max_loc[0]+test_case.image_loc[0],max_loc[1]+test_case.image_loc[1])
        correct=True
        if max_loc!=(test_case.template_loc[0], test_case.template_loc[1]):
            print("tf batch incorrect location")
            correct=False
        start = time.time()
        res_batch, ct_frames, compute_time, memory_time = tf_batch_conv(search_window, batch_template.copy(), verbose=False)
        match_time = (time.time()-start)/N
        print("tf-batch image-template pair: ",test_case.template_size, test_case.image_size, correct, match_time, 's' ) 
        
        timing_data.append(['tf-batch',test_case.template_size, test_case.image_size, correct, match_time, mem_templ_time])
    time_df = pd.DataFrame(timing_data, columns=['algorithm', 'template_size', 'search_window_size', 'accuracy', 'time', 'mem_templ_time'])
    time_df.to_csv("tm_timing_tf_batch.csv", index=False)

    test_case = test_cases[2]
    pair_scaling=[]
    for j in range(1,N*2):
        template, search_window = crop_template_search_window(test_case, image)
        start = time.time()
        batch_template = np.zeros((template.shape[0],template.shape[1],1,N), dtype=np.float32)
        for i in range(j):
          batch_template[:,:,0,i] = template[:,:].copy()
        mem_templ_time = time.time()-start
        start = time.time()
        res_batch, ct_frames, compute_time, memory_time = tf_batch_conv(search_window, batch_template.copy(), verbose=False)
        match_time = (time.time()-start)/j
        
        pair_scaling.append(['tf-batch',test_case.template_size, test_case.image_size,j,match_time ,mem_templ_time  ])
    pair_scaling_df = pd.DataFrame(pair_scaling, columns=['algorithm', 'template_size', 'search_window_size','N-pairs', 'time', 'cpu_mem_templ_time'])
    #pair_scaling_df.to_csv("tm_timing_N_tf_batch.csv", index=False)

def test_batch_shape():
    print("\ntf-batch implementation")
    with open("test_cases_400_1000.pickle", 'rb') as fileobj:
        test_cases = pickle.load(fileobj)    
    image_fname = "search8000x8000.png"
    image_path=image_fname
    #image_path = os.path.join("/eagle/BrainImagingML/apsage/n_template_match_gpu",image_fname)
    #image = cv2.imread(image_path, 0)
    image = np.asarray(ImageOps.grayscale(Image.open(image_path)), dtype=np.float32)
    test_case = test_cases[2]
    N=len(test_cases)
    template, search_window = crop_template_search_window(test_case, image)
    batch_template = np.zeros((template.shape[0],template.shape[1],1,N), dtype=np.float32)
    for i in range(N):
      batch_template[:,:,0,i] = template[:,:].copy()
    res_batch, ct_frames, compute_time, memory_time = tf_batch_conv(search_window, batch_template.copy(), verbose=False)
    print("single image. multi-template shapes")
    print("template", batch_template.shape, "search_window",np.expand_dims(np.expand_dims(search_window,2),0).shape, "result", res_batch.shape)
    res = res_batch[0,:,:,0].numpy()
    max_loc = np.unravel_index(res.argmax(), res.shape)
    max_loc = (max_loc[0]+test_case.image_loc[0],max_loc[1]+test_case.image_loc[1])
    correct=True
    if max_loc!=(test_case.template_loc[0], test_case.template_loc[1]):
      print("tf batch incorrect location")
      correct=False
    print("tf-batch image-template pair: ",test_case.template_size, test_case.image_size, correct, 's' ) 

    batch_templates = np.zeros((template.shape[0],template.shape[1],1,N), dtype=np.float32)
    image_batch = np.zeros((N,search_window.shape[0],search_window.shape[1],1))
    for i in range(N):
      test_case = test_cases[i]
      template, search_window = crop_template_search_window(test_case, image)
      batch_templates[:,:,0,i] = template.copy()
      image_batch[i,:,:,0] = search_window.copy()
    print("multi- image. multi-template shapes")
    ccoeff_coef = (math.reduce_sum(batch_templates,axis=(0,1), keepdims=True)/(batch_templates.shape[0]*batch_templates.shape[1]))
    batch_templates = math.subtract(batch_templates, ccoeff_coef)
    #print("batch_templates", type(batch_templates.numpy()), batch_templates.shape)
    #print("tf_image", type(tf_image.numpy()), tf_image.shape)
    res_batch=tf.nn.conv2d(image_batch, batch_templates,
              strides=[1,1,1,1],
              padding="VALID")
    print("template", batch_template.shape, "search_window",image_batch.shape, "result", res_batch.shape)
    res = res_batch[0,:,:,N-1].numpy()
    max_loc = np.unravel_index(res.argmax(), res.shape)
    max_loc = (max_loc[0]+test_case.image_loc[0],max_loc[1]+test_case.image_loc[1])
    correct=True
    if max_loc!=(test_case.template_loc[0], test_case.template_loc[1]):
      print("tf batch incorrect location")
      correct=False
    print("tf-batch image-template pair: ",test_case.template_size, test_case.image_size, correct, 's' ) 

    batch_templates = np.zeros((template.shape[0],template.shape[1],1,N), dtype=np.float32)
    image_batch = np.zeros((1,search_window.shape[0],search_window.shape[1],N))
    for i in range(N):
      test_case = test_cases[i]
      template, search_window = crop_template_search_window(test_case, image)      
      batch_templates[:,:,0,i] = template.copy()
      image_batch[0,:,:,i] = search_window.copy()
    print("multi- image. multi-template shapes")
    ccoeff_coef = (math.reduce_sum(batch_templates,axis=(0,1), keepdims=True)/(batch_templates.shape[0]*batch_templates.shape[1]))
    batch_templates = math.subtract(batch_templates, ccoeff_coef)
    #print("batch_templates", type(batch_templates.numpy()), batch_templates.shape)
    #print("tf_image", type(tf_image.numpy()), tf_image.shape)
    res_batch=tf.nn.conv2d(image_batch, batch_templates,
              strides=[1,1,1,1],
              padding="VALID")
    print("template", batch_template.shape, "search_window",image_batch.shape, "result", res_batch.shape)
    res = res_batch[0,:,:,N-1].numpy()
    max_loc = np.unravel_index(res.argmax(), res.shape)
    max_loc = (max_loc[0]+test_case.image_loc[0],max_loc[1]+test_case.image_loc[1])
    correct=True
    if max_loc!=(test_case.template_loc[0], test_case.template_loc[1]):
      print("tf batch incorrect location")
      correct=False
    print("tf-batch image-template pair: ",test_case.template_size, test_case.image_size, correct, 's' ) 

if __name__=='__main__':
  timing_test_cases()
