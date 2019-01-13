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
//#include <png++/png.hpp>

using namespace cv;
using namespace std;


typedef vector<double> Array;
typedef vector<Array> Matrix;
typedef vector<Matrix> Image;
typedef vector<Image> Container;

const int kNewWidth =48;
const int kNewHeight =48;

int newheight;
int newwidth;

class Get_Image

{
public:

    Image loadImage(const string &filename);
};

class Fetch_Data
{

public:

    bool getFileContent(std::string fileName, std::vector<std::string> & vecOfStrs);
    Image convolution_kernal ();
    Matrix conv_bias_value ();
    Matrix dense_value ();
    Matrix dense_bias_value ();

};



Image Get_Image :: loadImage(const string &filename)
{
    int image_depth;
    cout<<"\nGive the depth of image: "<<endl;
    cin>>image_depth;
    cout<<"\n\n-----------------------Load Image Function has started to work-----------------------"<<endl;
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

    Mat new_mat, conv,A;
    resize( E,new_mat, Size(kNewWidth, kNewHeight));
    new_mat.convertTo(new_mat,CV_64F);

    cout<<endl;
    cout<<"\nResized normalized image's size: "<<new_mat.size()<<endl;
//    cout<<"\nResized nomalized image's matrix: \n"<<new_mat<<endl; //48*48

    Image image_1(Image(image_depth,Matrix(kNewHeight,Array())));

//    image_1.resize(kNewHeight*kNewWidth);
//    image_1.resize(kNewHeight);
//    image_1[0].resize(kNewWidth);
    cout<<"\n OOOKKK"<<endl;
    double *ptrDst[new_mat.rows];
    //Array val;
for(int k=0;k<image_depth;k++)
{

    for(int i = 0; i < new_mat.rows; ++i)
    {


        ptrDst[i] = new_mat.ptr<double>(i);
       // cout<<ptrDst[i];
       for(int j = 0; j < new_mat.cols; ++j)
       {

        double value = ptrDst[i][j];
        image_1[k][i].push_back(value);
//        cout<<"OK"<<endl;
//        image_1[i][j]=value;

        }
        //cout<<"first row";
    }
//    cout<<image_1[k][0].size()<<endl;
    }
    cout<<"\nNow the image is ready to be CONVOLVED!!!!!"<<endl;
    cout<<"\nProcessed Image Depth: "<<image_1.size()<<endl;
    cout<<"\nProcessed Image Row: "<<image_1[0].size()<<endl;
    cout<<"\nProcessed Image Column: "<<image_1[0][0].size()<<endl;

    cout<<endl;


    return image_1;


}






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


     }

     return 0;




}
