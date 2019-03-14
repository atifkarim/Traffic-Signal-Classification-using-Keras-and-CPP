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

    auto start = high_resolution_clock::now(); //this line for starting the time calculation for whole classification process

    auto start_convolution = high_resolution_clock::now(); //time calculation start for convolution
    Image convImage = obj3.applyFilter(preprocessed_image, convolution_filter_1, conv_bias);
    auto stop_convolution = high_resolution_clock::now(); //time calculation stop for convolution
    auto duration_convolution = duration_cast<microseconds>(stop_convolution - start_convolution); //time calculated for convolution
    cout<<"\nTime taken for convolution is: "<<duration_convolution.count()/1000.000000<<" millisecond"<<endl;//print out the time


    auto start_resizing = high_resolution_clock::now();
    Matrix resized_conv_relu_image_value = obj3.resized_conv_relu_image(convImage);
    auto stop_resizing = high_resolution_clock::now();
    auto duration_resizing = duration_cast<microseconds>(stop_resizing - start_resizing);
    cout<<"\nTime taken for resizing is: "<<duration_resizing.count()/1000.000000<<" millisecond"<<endl;

    auto start_matmul = high_resolution_clock::now();
    Matrix matmul_dense_resized_relu = obj3.matmul_dense_resized_conv_relu(resized_conv_relu_image_value,dense_kernel,dense_bias);
    auto stop_matmul = high_resolution_clock::now();
    auto duration_matmul = duration_cast<microseconds>(stop_matmul - start_matmul);
    cout<<"Time taken for matrix_multiplication is: "<<duration_matmul.count()/1000.000000<<" millisecond"<<endl;

    auto start_softmax = high_resolution_clock::now();
    Matrix softmax_calculation = obj3.softmax(matmul_dense_resized_relu);
    auto stop_softmax = high_resolution_clock::now();
    auto duration_softmax = duration_cast<microseconds>(stop_softmax - start_softmax);
    cout<<"\nTime taken for softmax is: "<<duration_softmax.count()/1000.000000<<" millisecond"<<endl;



    auto stop = high_resolution_clock::now(); //Here time calculation for whole classification process stop

    auto duration = duration_cast<microseconds>(stop - start);

    time_t = duration.count()/1000.000;

    total_time =total_time+ time_t;

    // below instead of time_t you can write duration.count() if you want to see the direct output from chrono
    cout <<"\nTime taken for classification: "<< time_t <<" milliseconds and total time is: "<<total_time<<" milliseconds"<<endl;

    cout<<"\n-------------------------------------------------------------"<<endl;

  }
  double average_time = (total_time/fn.size());
  cout<<"\nAverage time to classify per image is: "<<average_time<<" milliseconds"<<endl;
  return 0;
}
