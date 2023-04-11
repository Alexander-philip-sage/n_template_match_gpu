#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include<opencv2/opencv.hpp>

struct TemplateMatchTestCase{
	std::string method;
	int dim1;
	int dim2;
	int width;
};
int main() {

  TemplateMatchTestCase beads_test_cases[39] = { {"cv2.TM_CCOEFF", 547, 1523, 200},
  {"cv2.TM_CCOEFF", 547, 1523, 200},  {"cv2.TM_CCOEFF", 36, 115, 200},  {"cv2.TM_CCOEFF", 186, 1200, 200},  {"cv2.TM_CCOEFF", 183, 1334, 200},
  {"cv2.TM_CCOEFF", 241, 1037, 100},  {"cv2.TM_CCOEFF", 516, 705, 300},  {"cv2.TM_CCOEFF_NORMED", 547, 1523, 200},  {"cv2.TM_CCOEFF_NORMED", 547, 1523, 200},
  {"cv2.TM_CCOEFF_NORMED", 36, 115, 200},  {"cv2.TM_CCOEFF_NORMED", 186, 1200, 200},  {"cv2.TM_CCOEFF_NORMED", 183, 1334, 200},  {"cv2.TM_CCOEFF_NORMED", 241, 1037, 100},
  {"cv2.TM_CCOEFF_NORMED", 516, 705, 300},  {"cv2.TM_CCORR", 547, 1523, 200},  {"cv2.TM_CCORR", 547, 1523, 200},  {"cv2.TM_CCORR", 186, 1200, 200},
  {"cv2.TM_CCORR", 516, 705, 300},  {"cv2.TM_CCORR_NORMED", 547, 1523, 200},  {"cv2.TM_CCORR_NORMED", 547, 1523, 200},  {"cv2.TM_CCORR_NORMED", 36, 115, 200},
  {"cv2.TM_CCORR_NORMED", 186, 1200, 200},  {"cv2.TM_CCORR_NORMED", 183, 1334, 200},  {"cv2.TM_CCORR_NORMED", 241, 1037, 100},  {"cv2.TM_CCORR_NORMED", 516, 705, 300},
  {"cv2.TM_SQDIFF", 547, 1523, 200},  {"cv2.TM_SQDIFF", 547, 1523, 200},  {"cv2.TM_SQDIFF", 36, 115, 200},  {"cv2.TM_SQDIFF", 186, 1200, 200},
  {"cv2.TM_SQDIFF", 183, 1334, 200},  {"cv2.TM_SQDIFF", 241, 1037, 100},  {"cv2.TM_SQDIFF", 516, 705, 300},  {"cv2.TM_SQDIFF_NORMED", 547, 1523, 200},
  {"cv2.TM_SQDIFF_NORMED", 547, 1523, 200},  {"cv2.TM_SQDIFF_NORMED", 36, 115, 200},  {"cv2.TM_SQDIFF_NORMED", 186, 1200, 200},  {"cv2.TM_SQDIFF_NORMED", 183, 1334, 200},
  {"cv2.TM_SQDIFF_NORMED", 241, 1037, 100},  {"cv2.TM_SQDIFF_NORMED", 516, 705, 300}};
  TemplateMatchTestCase stuff_test_cases[49] = { {"cv2.TM_CCOEFF", 270, 244, 200},
  {"cv2.TM_CCOEFF", 228, 156, 200},  {"cv2.TM_CCOEFF", 64, 45, 200},  {"cv2.TM_CCOEFF", 39, 158, 200},  {"cv2.TM_CCOEFF", 252, 169, 200},
  {"cv2.TM_CCOEFF", 29, 412, 200},  {"cv2.TM_CCOEFF", 139, 54, 200},  {"cv2.TM_CCOEFF", 146, 247, 200},  {"cv2.TM_CCOEFF", 187, 236, 125},
  {"cv2.TM_CCOEFF_NORMED", 270, 244, 200},  {"cv2.TM_CCOEFF_NORMED", 228, 156, 200},  {"cv2.TM_CCOEFF_NORMED", 64, 45, 200},  {"cv2.TM_CCOEFF_NORMED", 39, 158, 200},
  {"cv2.TM_CCOEFF_NORMED", 252, 169, 200},  {"cv2.TM_CCOEFF_NORMED", 29, 412, 200},  {"cv2.TM_CCOEFF_NORMED", 139, 54, 200},  {"cv2.TM_CCOEFF_NORMED", 146, 247, 200},
  {"cv2.TM_CCOEFF_NORMED", 187, 236, 125},  {"cv2.TM_CCOEFF_NORMED", 15, 23, 140},  {"cv2.TM_CCORR_NORMED", 270, 244, 200},  {"cv2.TM_CCORR_NORMED", 228, 156, 200},
  {"cv2.TM_CCORR_NORMED", 64, 45, 200},  {"cv2.TM_CCORR_NORMED", 39, 158, 200},  {"cv2.TM_CCORR_NORMED", 252, 169, 200},  {"cv2.TM_CCORR_NORMED", 29, 412, 200},
  {"cv2.TM_CCORR_NORMED", 139, 54, 200},  {"cv2.TM_CCORR_NORMED", 146, 247, 200},  {"cv2.TM_CCORR_NORMED", 187, 236, 125},  {"cv2.TM_CCORR_NORMED", 15, 23, 140},
  {"cv2.TM_SQDIFF", 270, 244, 200},  {"cv2.TM_SQDIFF", 228, 156, 200},  {"cv2.TM_SQDIFF", 64, 45, 200},  {"cv2.TM_SQDIFF", 39, 158, 200},
  {"cv2.TM_SQDIFF", 252, 169, 200},  {"cv2.TM_SQDIFF", 29, 412, 200},  {"cv2.TM_SQDIFF", 139, 54, 200},  {"cv2.TM_SQDIFF", 146, 247, 200},
  {"cv2.TM_SQDIFF", 187, 236, 125},  {"cv2.TM_SQDIFF", 15, 23, 140},  {"cv2.TM_SQDIFF_NORMED", 270, 244, 200},  {"cv2.TM_SQDIFF_NORMED", 228, 156, 200},
  {"cv2.TM_SQDIFF_NORMED", 64, 45, 200},  {"cv2.TM_SQDIFF_NORMED", 39, 158, 200},  {"cv2.TM_SQDIFF_NORMED", 252, 169, 200},  {"cv2.TM_SQDIFF_NORMED", 29, 412, 200},
  {"cv2.TM_SQDIFF_NORMED", 139, 54, 200},  {"cv2.TM_SQDIFF_NORMED", 146, 247, 200},  {"cv2.TM_SQDIFF_NORMED", 187, 236, 125},  {"cv2.TM_SQDIFF_NORMED", 15, 23, 140}
  };

  std::cout << "running test_template_match.cpp"<< std::endl;
  std::cout <<"test case method "<<stuff_test_cases[0].method<< " dim1 " <<stuff_test_cases[0].dim1 << " dim2 " <<stuff_test_cases[0].dim2 << " width " << stuff_test_cases[0].width<<std::endl;

  int method_int;
  cv::Mat image; cv::Mat templt; cv::Mat res;
  image = cv::imread("stuff.jpg", cv::IMREAD_GRAYSCALE);
  if(! image.data ) {
      std::cout <<  "Could not open or find the image" << std::endl ;
      return -1;
  }

  int i = 0;
  cv::Rect crop_region(stuff_test_cases[i].dim1, stuff_test_cases[i].dim2,stuff_test_cases[i].width, stuff_test_cases[i].width);
  templt =image(crop_region); 
  
  int result_cols = image.cols - templt.cols + 1;
  int result_rows = image.rows - templt.rows + 1;
  res.create( result_rows, result_cols, CV_32FC1 );

  //"switch" for defining the method as an int
  if(stuff_test_cases[i].method.compare("cv2.TM_CCOEFF")){
    method_int=cv::TM_CCOEFF;
  }else if(stuff_test_cases[i].method.compare("cv2.TM_CCOEFF_NORMED")){
    method_int=cv::TM_CCOEFF_NORMED;
  }else if(stuff_test_cases[i].method.compare("cv2.TM_CCORR")){
    method_int=cv::TM_CCORR;
  }else if(stuff_test_cases[i].method.compare("cv2.TM_CCORR_NORMED")){
    method_int=cv::TM_CCORR_NORMED;
  }else if(stuff_test_cases[i].method.compare("cv2.TM_SQDIFF")){
    method_int=cv::TM_SQDIFF;
  }else if(stuff_test_cases[i].method.compare("cv2.TM_SQDIFF_NORMED")){
    method_int=cv::TM_SQDIFF_NORMED;
  }

  matchTemplate( image, templt, res, method_int);

  /* cv::normalize( res, res, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );
  double minVal; double maxVal; cv::Point minLoc; cv::Point maxLoc;
  cv::Point matchLoc;
  cv::minMaxLoc( res, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat() );
  if( method_int  == cv::TM_SQDIFF || method_int == cv::TM_SQDIFF_NORMED )
    { matchLoc = minLoc; }
  else
    { matchLoc = maxLoc; }  

  return 0;*/
}
