#include<iostream>
#include "get_data.h"
#include "get_image.h"
#include "get_class.h"

#include<cstring>
#include<string>
#include <vector>
#include <assert.h>
#include <cmath>
#include<sstream>
#include <fstream>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"


//using  ns = chrono::nanoseconds;
//using get_time = chrono::steady_clock ;

using namespace std::chrono;


//typedef vector<double> Array;
//typedef vector<Array> Matrix;
//typedef vector<Matrix> Image;
//typedef vector<Image> Container;



using namespace std;
using namespace cv;


int main()
{
  Get_Image obj1;

  Fetch_Data obj2;

  do_calculation obj3;

  cv::String path("/home/atif/image_classification_c++/multi_filter_cpp/test_image/*.ppm"); //select only jpg
  vector<cv::String> fn;
  vector<cv::Mat> data;
  cv::glob(path,fn,true); // recurse
  cout<<"\n Loaded number of image: "<<fn.size()<<endl;
  for (size_t k=0; k<fn.size(); ++k)
  {/*
        const int kNewWidth =48;
        const int kNewHeight =48;*/

    int newheight;
    int newwidth;

    cv::Mat im = cv::imread(fn[k]);
    //     const string c=fn[k];
    if (im.empty()) continue; //only proceed if sucsessful



    Image preprocessed_image = obj1.loadImage(fn[k]);

    Image convolution_filter_1 = obj2.convolution_kernal();

    Matrix conv_bias= obj2.conv_bias_value();

    Matrix dense_kernel = obj2.dense_value();

    Matrix dense_bias = obj2.dense_bias_value();

//    auto start = get_time::now();

    auto start = high_resolution_clock::now();

    Image convImage = obj3.applyFilter(preprocessed_image, convolution_filter_1, conv_bias);

    Matrix resized_conv_relu_image_value = obj3.resized_conv_relu_image(convImage);

    Matrix matmul_dense_resized_relu = obj3.matmul_dense_resized_conv_relu(resized_conv_relu_image_value,dense_kernel,dense_bias);
    Matrix softmax_calculation = obj3.softmax(matmul_dense_resized_relu);

//    auto end = get_time::now();
    auto stop = high_resolution_clock::now();
//    auto diff = end - start;
    auto duration = duration_cast<microseconds>(stop - start);
//    auto duration = duration_cast<seconds>(stop - start);

//    cout<<"Elapsed time is :  "<< chrono::duration_cast<ns>(diff).count()<<" ns "<<endl;

    cout <<"time taken: "<< duration.count() << endl;


  }
  return 0;




}
