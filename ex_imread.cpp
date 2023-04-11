#include <opencv2/highgui.hpp>
#include <iostream>

int main( int argc, char** argv ) {
  
  cv::Mat image;
  image = cv::imread("../mb_aligner/polaris_shells/OUT_DIR/png/0004_W05_Sec004/0004_W05_Sec004_tr1-tc1.png_tr1-tc1.png");
  
  if(! image.data ) {
      std::cout <<  "Could not open or find the image" << std::endl ;
      return -1;
    }
  
  
  return 0;
}
