#include<iostream>
#include "get_data.h"
#include "get_image.h"

#include<cstring>
#include<string>

#include <vector>
#include <assert.h>
#include <cmath>
#include<sstream>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"


typedef vector<double> Array;
typedef vector<Array> Matrix;
typedef vector<Matrix> Image;
typedef vector<Image> Container;

using namespace std;
using namespace cv;


int main()
{
  Get_Image obj1;

  Fetch_Data obj2;

  cv::String path("/home/atif/image_classification_c++/multi_filter_cpp/test_image/*.ppm"); //select only jpg
      vector<cv::String> fn;
      vector<cv::Mat> data;
      cv::glob(path,fn,true); // recurse
      cout<<"\n Loaded number of image: "<<fn.size()<<endl;
      for (size_t k=0; k<fn.size(); ++k)
      {
       cv::Mat im = cv::imread(fn[k]);
  //     const string c=fn[k];
       if (im.empty()) continue; //only proceed if sucsessful



       Image preprocessed_image = obj1.loadImage(fn[k]);

     Image convolution_filter_1 = obj2.convolution_kernal();

     Matrix conv_bias= obj2.conv_bias_value();

     Matrix dense_kernel = obj2.dense_value();

     Matrix dense_bias = obj2.dense_bias_value();



}
     return 0;




}
