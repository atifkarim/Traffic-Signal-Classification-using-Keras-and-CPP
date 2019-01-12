#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include<cstring>
#include<string>
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"

#include <vector>
#include <assert.h>
#include <cmath>
#include<sstream>
#include <fstream>

using namespace cv;
using namespace std;


typedef vector<double> Array;
typedef vector<Array> Matrix; // 2D Matrix
typedef vector<Matrix> Image; // 3D Matrix
typedef vector<Image> Container; // 4D Matrix

const int kNewWidth =48;
const int kNewHeight =48;

int newheight;
int newwidth;


Image loadImage(const string &filename)
{
    cout<<"\n\n-----------------------Load Image Function has started to work-----------------------"<<endl;

    int image_depth;
//    cout<<"\nGive the depth of image: "<<endl;
    cin>>image_depth;

    Mat I = imread(filename, 1);
    cout<<"\nLoaded Image Size: "<<I.size()<<endl;

    Mat E;
    cv::cvtColor(I, E, CV_RGB2GRAY);
    cout<<"\nGrayscaled Image Size: "<<E.size()<<endl;
//    Mat Norm_img;
    E.convertTo(E, CV_64F, 1.0 / 255, 0);
//
    cout<<endl;
    cout<<"\nNormalized Image size: "<<E.size()<<endl;
//    cout<<"\nNormalized Image Matrix: "<<E<<endl;

    if (I.empty())
    {
        std::cout << "!!! Failed imread(): image not found" << std::endl;
    // don't let the execution continue, else imshow() will crash.
    }

    Mat mat_image;
    resize( E,mat_image, Size(kNewWidth, kNewHeight));
    mat_image.convertTo(mat_image,CV_64F);

    cout<<endl;
    cout<<"\nResized normalized image's size: "<<mat_image.size()<<endl;
//    cout<<"\nResized nomalized image's matrix: \n"<<mat_image<<endl; //48*48

    Image processed_image(Image(image_depth,Matrix(kNewHeight,Array())));

//    processed_image.resize(kNewHeight*kNewWidth);
//    processed_image.resize(kNewHeight);
//    processed_image[0].resize(kNewWidth);
    cout<<"\n OOOKKK"<<endl;
    double *ptrDst[mat_image.rows];
    //Array val;

    for(int k=0;k<image_depth;k++)
    {

        for(int i = 0; i < mat_image.rows; ++i)
        {


            ptrDst[i] = mat_image.ptr<double>(i);
       // cout<<ptrDst[i];
            for(int j = 0; j < mat_image.cols; ++j)
            {

                double value = ptrDst[i][j];
                processed_image[k][i].push_back(value);
//              cout<<"OK"<<endl;
//              processed_image[i][j]=value;

            }

        }
//    cout<<processed_image[k][0].size()<<endl;
    }

    cout<<"\nNow the image is ready to be CONVOLVED!!!!!"<<endl;
    cout<<"\nProcessed Image Depth: "<<processed_image.size()<<endl;
    cout<<"\nProcessed Image Row: "<<processed_image[0].size()<<endl;
    cout<<"\nProcessed Image Column: "<<processed_image[0][0].size()<<endl;

    cout<<endl;


    return processed_image;


}















int main()
{


    cv::String path("/home/atif/image_classification_c++/multi_filter_cpp/test_image/*.ppm"); //select only ppm. change the extension with respect to your saved image extension
    vector<cv::String> fn;
    vector<cv::Mat> data;
    cv::glob(path,fn,true); // recurse
    cout<<"\n Loaded number of image: "<<fn.size()<<endl;
    for (size_t k=0; k<fn.size(); ++k)
    {
     cv::Mat im = cv::imread(fn[k]);
//     const string c=fn[k];
     if (im.empty()) continue; //only proceed if sucsessful


     Image preprocessed_image = loadImage(fn[k]);






     return 0;

}

}
