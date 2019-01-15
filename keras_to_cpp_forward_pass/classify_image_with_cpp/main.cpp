#include<iostream>
#include "get_data.h"
#include "get_image.h"
#include "get_class.h"
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;
using namespace std::chrono;

int main()
{
  Get_Image obj1;
  Fetch_Data obj2;
  do_calculation obj3;
  double total_time=0.000000;
  double time_t;
  //  float average_time;


  Image convolution_filter_1 = obj2.convolution_kernal();

  Matrix conv_bias= obj2.conv_bias_value();

  Matrix dense_kernel = obj2.dense_value();

  Matrix dense_bias = obj2.dense_bias_value();

  cv::String path("/home/atif/image_classification_c++/multi_filter_cpp/test_image/*.ppm"); //select only ppm
  vector<cv::String> fn;
  vector<cv::Mat> data;
  cv::glob(path,fn,true); // recurse
  cout<<"\nNumber of image in the directory is: "<<fn.size()<<endl;


  for (size_t k=0; k<fn.size(); ++k)
  {

    //    int newheight;
    //    int newwidth;

    cv::Mat im = cv::imread(fn[k]);
    if (im.empty()) continue; //only proceed if sucsessful

    Image preprocessed_image = obj1.loadImage(fn[k]);

    auto start = high_resolution_clock::now();

    Image convImage = obj3.applyFilter(preprocessed_image, convolution_filter_1, conv_bias);

    Matrix resized_conv_relu_image_value = obj3.resized_conv_relu_image(convImage);

    Matrix matmul_dense_resized_relu = obj3.matmul_dense_resized_conv_relu(resized_conv_relu_image_value,dense_kernel,dense_bias);

    Matrix softmax_calculation = obj3.softmax(matmul_dense_resized_relu);

    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(stop - start);

    time_t = duration.count();

    total_time =total_time+ time_t;

    cout <<"\nTime taken for classification: "<< duration.count() <<" microseconds and total time is: "<<total_time<<endl;

    cout<<"\n-------------------------------------------------------------"<<endl;

  }
  double average_time = (total_time/fn.size());
  cout<<"\nAverage time to classify per image is: "<<average_time<<" microseconds"<<endl;
  return 0;
}
