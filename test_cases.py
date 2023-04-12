import numpy as np
STUFF_TEST_CASES=['stuff.jpg',[['cv2.TM_CCOEFF', 270, 244, 200], ['cv2.TM_CCOEFF', 228, 156, 200], 
                                ['cv2.TM_CCOEFF', 64, 45, 200], ['cv2.TM_CCOEFF', 39, 158, 200], 
                                ['cv2.TM_CCOEFF', 252, 169, 200], 
                                ['cv2.TM_CCOEFF', 139, 54, 200], ['cv2.TM_CCOEFF', 146, 247, 200], 
                                ['cv2.TM_CCOEFF', 187, 236, 125], ['cv2.TM_CCOEFF_NORMED', 270, 244, 200], ['cv2.TM_CCOEFF_NORMED', 228, 156, 200], 
                                ['cv2.TM_CCOEFF_NORMED', 64, 45, 200], ['cv2.TM_CCOEFF_NORMED', 39, 158, 200], 
                                ['cv2.TM_CCOEFF_NORMED', 252, 169, 200], ['cv2.TM_CCOEFF_NORMED', 29, 412, 200],
                                ['cv2.TM_CCOEFF_NORMED', 139, 54, 200], ['cv2.TM_CCOEFF_NORMED', 146, 247, 200], 
                                ['cv2.TM_CCOEFF_NORMED', 187, 236, 125], ['cv2.TM_CCOEFF_NORMED', 15, 23, 140], 
                                ['cv2.TM_CCORR_NORMED', 270, 244, 200], ['cv2.TM_CCORR_NORMED', 228, 156, 200], 
                                ['cv2.TM_CCORR_NORMED', 64, 45, 200], ['cv2.TM_CCORR_NORMED', 39, 158, 200], 
                                ['cv2.TM_CCORR_NORMED', 252, 169, 200], ['cv2.TM_CCORR_NORMED', 29, 412, 200], 
                                ['cv2.TM_CCORR_NORMED', 139, 54, 200],
                                ['cv2.TM_CCORR_NORMED', 146, 247, 200], ['cv2.TM_CCORR_NORMED', 187, 236, 125], 
                                ['cv2.TM_CCORR_NORMED', 15, 23, 140], ['cv2.TM_SQDIFF', 270, 244, 200], ['cv2.TM_SQDIFF', 228, 156, 200], 
                                ['cv2.TM_SQDIFF', 64, 45, 200], ['cv2.TM_SQDIFF', 39, 158, 200], ['cv2.TM_SQDIFF', 252, 169, 200],
                                ['cv2.TM_SQDIFF', 29, 412, 200], ['cv2.TM_SQDIFF', 139, 54, 200], ['cv2.TM_SQDIFF', 146, 247, 200], 
                                ['cv2.TM_SQDIFF', 187, 236, 125], ['cv2.TM_SQDIFF', 15, 23, 140], ['cv2.TM_SQDIFF_NORMED', 270, 244, 200],
                                ['cv2.TM_SQDIFF_NORMED', 228, 156, 200], ['cv2.TM_SQDIFF_NORMED', 64, 45, 200], ['cv2.TM_SQDIFF_NORMED', 39, 158, 200], 
                                ['cv2.TM_SQDIFF_NORMED', 252, 169, 200], ['cv2.TM_SQDIFF_NORMED', 29, 412, 200], 
                                ['cv2.TM_SQDIFF_NORMED', 139, 54, 200], ['cv2.TM_SQDIFF_NORMED', 146, 247, 200], 
                                ['cv2.TM_SQDIFF_NORMED', 187, 236, 125], ['cv2.TM_SQDIFF_NORMED', 15, 23, 140]]]
BEADS_TEST_CASES=['beads.jpg',[['cv2.TM_CCOEFF', 547, 1523, 200], ['cv2.TM_CCOEFF', 547, 1523, 200], 
                                ['cv2.TM_CCOEFF', 36, 115, 200], ['cv2.TM_CCOEFF', 186, 1200, 200], 
                                ['cv2.TM_CCOEFF', 183, 1334, 200], ['cv2.TM_CCOEFF', 241, 1037, 100], 
                                ['cv2.TM_CCOEFF', 516, 705, 300], ['cv2.TM_CCOEFF_NORMED', 547, 1523, 200],
                                ['cv2.TM_CCOEFF_NORMED', 547, 1523, 200], ['cv2.TM_CCOEFF_NORMED', 36, 115, 200],
                                ['cv2.TM_CCOEFF_NORMED', 186, 1200, 200], ['cv2.TM_CCOEFF_NORMED', 183, 1334, 200],
                                ['cv2.TM_CCOEFF_NORMED', 241, 1037, 100], ['cv2.TM_CCOEFF_NORMED', 516, 705, 300], 
                                ['cv2.TM_CCORR', 547, 1523, 200], ['cv2.TM_CCORR', 547, 1523, 200], ['cv2.TM_CCORR', 186, 1200, 200], 
                                ['cv2.TM_CCORR', 516, 705, 300], ['cv2.TM_CCORR_NORMED', 547, 1523, 200], ['cv2.TM_CCORR_NORMED', 547, 1523, 200],
                                ['cv2.TM_CCORR_NORMED', 36, 115, 200], ['cv2.TM_CCORR_NORMED', 186, 1200, 200], ['cv2.TM_CCORR_NORMED', 183, 1334, 200], 
                                ['cv2.TM_CCORR_NORMED', 241, 1037, 100], ['cv2.TM_CCORR_NORMED', 516, 705, 300], ['cv2.TM_SQDIFF', 547, 1523, 200], 
                                ['cv2.TM_SQDIFF', 547, 1523, 200], ['cv2.TM_SQDIFF', 36, 115, 200], ['cv2.TM_SQDIFF', 186, 1200, 200], 
                                ['cv2.TM_SQDIFF', 183, 1334, 200], ['cv2.TM_SQDIFF', 241, 1037, 100], ['cv2.TM_SQDIFF', 516, 705, 300],
                                ['cv2.TM_SQDIFF_NORMED', 547, 1523, 200], ['cv2.TM_SQDIFF_NORMED', 547, 1523, 200], ['cv2.TM_SQDIFF_NORMED', 36, 115, 200], 
                                ['cv2.TM_SQDIFF_NORMED', 186, 1200, 200], ['cv2.TM_SQDIFF_NORMED', 183, 1334, 200], 
                                ['cv2.TM_SQDIFF_NORMED', 241, 1037, 100], ['cv2.TM_SQDIFF_NORMED', 516, 705, 300]]]
METHODS = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
STUFF_TEST_CASES_CCOEFF = [STUFF_TEST_CASES[0], [x for x in STUFF_TEST_CASES[1] if x[0]=='cv2.TM_CCOEFF']]
BEADS_TEST_CASES_CCOEFF = [STUFF_TEST_CASES[0], [x for x in BEADS_TEST_CASES[1] if x[0]=='cv2.TM_CCOEFF']]

def get_test_data(test_list, index):
    image_fname=test_list[0]
    method_name, dim1, dim2, templ_width = test_list[1][index]
    #method_name, dim1, dim2, templ_width = test_list[1][index][0], test_list[1][index][1], test_list[1][index][2], test_list[1][index][3]
    return image_fname, method_name, dim1, dim2, templ_width

def find_match_location(res:np.ndarray, method_name:str, verbose=False):
  max_loc = np.unravel_index(res.argmax(), res.shape)
  min_loc = np.unravel_index(res.argmin(), res.shape)
  if verbose:
    max_val =res.max()
    min_val = res.min()
    print("min value", min_val, "at", min_loc, "max value", max_val, "at", max_loc)
    all_vals = res.flatten()
    mean , std = all_vals.mean(), all_vals.std()
  if verbose:
    print("norm", mean, "std", std)
  #plt.xscale('log')
  # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
  if verbose:
    print("distance of the found location in units of std")
  if method_name in ['cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']:
    top_left = min_loc
    if verbose:
      print('dist', (mean-min_val)/std)
  else:
    if verbose:
      print('dist', (max_val-mean)/std)
    top_left = max_loc
  return top_left