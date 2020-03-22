#include<iostream>
#include "get_image.h"

#include<cstring>
#include<string>
#include <vector>
#include <assert.h>
#include <stdlib.h>
#include <typeinfo>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"


using namespace std;
using namespace cv;


const int kNewWidth =8;
const int kNewHeight =8;


Image Get_Image :: loadImage(const string &filename)
{
  int image_depth=1;
  //  cout<<"\nGive the depth of image: "<<endl;
  //  cin>>image_depth;
  Mat I = imread(filename, 1);
  cout<<"\nImage name: "<<filename;
    cout<<"\nLoaded Image Size: "<<I.size()<<endl;

  Mat E;
  cv::cvtColor(I, E, CV_RGB2GRAY);
//  cout<<"\nGRAY Image Matrix: "<<E<<endl;
//  cv::cvtColor(I, E, CV_BGR2GRAY);
  //  cout<<"\nGrayscaled Image Size: "<<E.size()<<endl;
  //      Mat Norm_img;
//  E.convertTo(E, CV_64F, 1.0 / 255, 0);
  E.convertTo(E, CV_16U, 255,0); // here image has assigned to give 16 bit unsigned value. multiply with 255
  //  cout<<endl;
//    cout<<"\nNormalized Image size: "<<E.size()<<endl;
//        cout<<"\nNormalized Image Matrix: "<<E<<endl;

  if (I.empty())
  {
    std::cout << "!!! Failed imread(): image not found" << std::endl;
    // don't let the execution continue, else imshow() will crash.
  }

  Mat new_image;
  resize( E,new_image, Size(kNewWidth, kNewHeight));
//  new_image.convertTo(new_image,CV_64F);
  new_image.convertTo(new_image,CV_16U); //here, image is converted to our desired size
//  cout << "Size of char: " << sizeof(new_image) << " byte" << endl;
//  cout <<"\nType is: "<< typeid(new_image).name() << endl;

  //  cout<<endl;
  //  cout<<"\nResized normalized image's size: "<<new_image.size()<<endl;
//    cout<<"\nResized nomalized image's matrix: \n"<<new_image<<endl; //48*48

  Image image_1(Image(image_depth,Matrix(kNewHeight,Array())));

  //    image_1.resize(kNewHeight*kNewWidth);
  //    image_1.resize(kNewHeight);
  //    image_1[0].resize(kNewWidth);
  //  cout<<"\n OOOKKK"<<endl;
//  double *ptrDst[new_image.rows]; //uncomment this line. correct line for floating point
  uint16_t *ptrDst[new_image.rows];
  //Array val;
  for(int k=0;k<image_depth;k++)
  {
    for(int i = 0; i < new_image.rows; ++i)
    {
//      ptrDst[i] = new_image.ptr<double>(i); //uncomment this line. correct line for floating point
      ptrDst[i] = new_image.ptr<uint16_t>(i);
      //       cout<<ptrDst[i];
      for(int j = 0; j < new_image.cols; ++j)
      {
//        double value = ptrDst[i][j]; //uncomment this line. correct line for floating point
        uint16_t value = ptrDst[i][j];
        image_1[k][i].push_back(value);
        //        cout<<"OK"<<endl;
        //        image_1[i][j]=value;
//cout<<value<<endl;
      }
      //cout<<"first row";
    }
//        cout<<image_1[k][0].size()<<endl;
  }
//    cout<<"\nNow the image is ready to be CONVOLVED!!!!!"<<endl;
//    cout<<"\nProcessed Image Depth: "<<image_1.size()<<endl;
//    cout<<"\nProcessed Image Row: "<<image_1[0].size()<<endl;
//    cout<<"\nProcessed Image Column: "<<image_1[0][0].size()<<endl;

  //  cout<<endl;

    for(int a=0;a<image_1.size();a++)
    {
      for(int b= 0; b<image_1[0].size();b++)
      {
        for(int c= 0; c<image_1[0][0].size();c++)
        {

                          cout<<image_1[a][b][c]<<" "; //uncomment it if you want to see the vector output where the convolution kernel weights are stored

        }
                    cout<<endl; //uncomment it if uncomment previous line
      }
    }




  return image_1;
}
