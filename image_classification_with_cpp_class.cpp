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

//    bool getFileContent(std::string fileName, std::vector<std::string> & vecOfStrs);
    Image convolution_kernal ();
    Matrix conv_bias_value ();
    Matrix dense_value ();
    Matrix dense_bias_value ();


};

bool getFileContent(std::string fileName, std::vector<std::string> & vecOfStrs)

{
    // Open the File
	std::ifstream in(fileName.c_str());

	// Check if object is valid
	if(!in)
	{
		std::cerr << "Cannot open the File : "<<fileName<<std::endl;
		return false;
	}

	std::string str;
	// Read the next line from File untill it reaches the end.
	while (std::getline(in, str))
	{
		// Line contains string of length > 0 then save it in vector
		if(str.size() > 0)
			vecOfStrs.push_back(str);
	}
	//Close The File
	in.close();
	return true;


}


Image Fetch_Data :: convolution_kernal()

{
    cout<<"\n\n-------------Convolution kernel make function has started to work----------------"<<endl;

std::vector<std::string> vecOfStr;



    string temp;

    bool result = getFileContent("/home/atif/image_classification_c++/multi_filter_cpp/conv_kernel_traffic_2_filter_no_pad_gray_ep_100_for_cpp.txt", vecOfStr);

    int num_ber=0;
    long double found;
    int filter_length;
    cout<<"\nPlease mention the filter length you have used: \nThe filter length is: "; //this info you have to give here must.
    cin>>filter_length;

    int filter_number;
    if(vecOfStr.size()%filter_length!=0)
    {
      cout<<"\nRemainder after dividing line number with filter length is: "<<vecOfStr.size()%filter_length<<endl;
        cout<<"\nRemainder must be zero. Program terminate.\nChoose, suitable Filter length to make remainder ZERO"<<endl;
        abort();
    }
//

    filter_number=vecOfStr.size()/filter_length;
    cout<<"\nNumber of line: "<<vecOfStr.size()<<endl;
    cout<<"\nNumber of filter will be: "<<filter_number<<endl;


    Image conv_kernal_1(Image(filter_number,Matrix(filter_length,Array())));
	//std::vector<std::vector<long double> > numbers;
    int v=0;
	if(result)
	{
		// Print the vector contents

    for(int i=0;i<vecOfStr.size();i++)

		{

        if(num_ber<filter_length)

           { std::string line;
            line= vecOfStr[i];
//            cout<<line<<endl;
			//std::cout<<line<<std::endl;
            stringstream ss;
			ss<<line;
//            cout<<ss<<endl;

            //cout<<vecOfStr.size();
            while(!ss.eof())
            {



            ss>>temp;
            if(stringstream(temp)>>found)
            {

                long double f;
                f=found;
                cout<<"f: "<<f<<endl;

                conv_kernal_1[v][num_ber].push_back(f);


		//cout<<"\nROWW: "<<conv_kernal_1.size()<<" and COLLL: "<<conv_kernal_1[0].size()<<endl;
                //std::cout<<f<<std::endl;
			}


			}


        cout<<"number: "<<num_ber<<endl;
		num_ber++;
//		cout<<"number: "<<num_ber<<endl;
		cout<<"num i: "<<i<<endl;
		}
//i=i;
        cout<<"now i: "<<i<<endl;
		if(num_ber==filter_length && 1<filter_number){
//
        cout<<"EXECUTED & getting new filter value"<<endl;

		num_ber=0;
		filter_number=filter_number-1;
		v++;
		cout<<"\nRemaining filter number is: "<<filter_number<<endl;
		}

		}
	}

//	cout<<"\n\nStored convolution kernel value:\n";
//	cout<<endl<<"\n\nNumber of filter in Convolution kernal is: "<<conv_kernal_1.size()<<"\n\nRow of convolution kernal is: "<<conv_kernal_1[0].size()<<endl;
//    cout<<"\n\nColumn of Convolution kernal is: "<<conv_kernal_1[0][0].size()<<endl;

    cout<<endl;


for(int a=0;a<conv_kernal_1.size();a++)

{	for(int b= 0; b<conv_kernal_1[0].size();b++)
	{
        for(int c= 0; c<conv_kernal_1[0][0].size();c++)
        {

           cout<<conv_kernal_1[a][b][c]<<" "; //uncomment it if you want to see the vector output where the convolution kernel weights are stored

        }
       //cout<<endl; //uncomment it if uncomment previous line
	}
}
    cout<<endl;
    cout<<endl<<"\n\nNumber of filter in Convolution kernal is: "<<conv_kernal_1.size()<<"\n\nRow of convolution kernal is: "<<conv_kernal_1[0].size()<<endl;
    cout<<"\n\nColumn of Convolution kernal is: "<<conv_kernal_1[0][0].size()<<endl;
    cout<<endl;

    return conv_kernal_1;

}


Matrix Fetch_Data :: conv_bias_value()

{

    cout<<"\n\n------------- CONV bias value make function has started to work-------------"<<endl;
std::vector<std::string> vecOfStr;



    string temp;

    long double found;

    bool result = getFileContent("/home/atif/image_classification_c++/multi_filter_cpp/conv_bias_update.txt", vecOfStr);


    Matrix conv_bias_val(Matrix(vecOfStr.size(),Array()));
	//std::vector<std::vector<long double> > numbers;

	if(result)
	{
		// Print the vector contents
		for(int i=0;i<vecOfStr.size();i++)
		{
            std::string line;
            line= vecOfStr[i];
			//std::cout<<line<<std::endl;
            stringstream ss;
			ss<<line;

            //cout<<vecOfStr.size();
            while(!ss.eof())
            {



            ss>>temp;
            if(stringstream(temp)>>found)
            {

                long double f;
                f=found;
                conv_bias_val[i].push_back(f);
                //std::cout<<f<<std::endl;
			}


			}
		}



	}


	for(int i= 0; i<conv_bias_val.size();i++)
	{
        for(int j= 0; j<conv_bias_val[0].size();j++)
        {

           //cout<<numbers[i][j]<<" "; //uncomment it if you want to see the vector output where the dense kernel weights are stored

        }
       //cout<<endl; //uncomment it if uncomment previous line
	}
    cout<<endl<<"\n\nRow of conv bias value: "<<conv_bias_val.size()<<"\n\nColumn of conv bias value:"<<conv_bias_val[0].size()<<endl<<endl;

    return conv_bias_val;



}



Matrix Fetch_Data :: dense_value()

{
  cout<<"\n\n-------------Dense Kernal make function has started to work-------------"<<endl;
std::vector<std::string> vecOfStr;



    string temp;

    long double found;

    bool result = getFileContent("/home/atif/image_classification_c++/multi_filter_cpp/dense_kernel_traffic_2_filter_no_pad_gray_ep_100_for_cpp.txt", vecOfStr);


    Matrix numbers(Matrix(vecOfStr.size(),Array()));
	//std::vector<std::vector<long double> > numbers;

	if(result)
	{
		// Print the vector contents
		for(int i=0;i<vecOfStr.size();i++)
		{
            std::string line;
            line= vecOfStr[i];
			//std::cout<<line<<std::endl;
            stringstream ss;
			ss<<line;

            //cout<<vecOfStr.size();
            while(!ss.eof())
            {



            ss>>temp;
            if(stringstream(temp)>>found)
            {

                long double f;
                f=found;
                numbers[i].push_back(f);
                //std::cout<<f<<std::endl;
			}


			}
		}



	}


	for(int i= 0; i<numbers.size();i++)
	{
        for(int j= 0; j<numbers[0].size();j++)
        {

           //cout<<numbers[i][j]<<" "; //uncomment it if you want to see the vector output where the dense kernel weights are stored

        }
       //cout<<endl; //uncomment it if uncomment previous line
	}
    cout<<endl<<"\n\nRow of dense kernel: "<<numbers.size()<<"\n\nColumn of dense kernel:"<<numbers[0].size()<<endl<<endl;

    return numbers;



}

Matrix Fetch_Data :: dense_bias_value()

{
    cout<<"\n\n-------------Dense bias value make function has started to work-------------"<<endl;
std::vector<std::string> vecOfStr;



    string temp;

    long double found;

    bool result = getFileContent("/home/atif/image_classification_c++/multi_filter_cpp/dense_bias_update.txt", vecOfStr);


    Matrix dense_bias_val(Matrix(vecOfStr.size(),Array()));
	//std::vector<std::vector<long double> > numbers;

	if(result)
	{
		// Print the vector contents
		for(int i=0;i<vecOfStr.size();i++)
		{
            std::string line;
            line= vecOfStr[i];
			//std::cout<<line<<std::endl;
            stringstream ss;
			ss<<line;

            //cout<<vecOfStr.size();
            while(!ss.eof())
            {



            ss>>temp;
            if(stringstream(temp)>>found)
            {

                long double f;
                f=found;
                dense_bias_val[i].push_back(f);
                //std::cout<<f<<std::endl;
			}


			}
		}



	}


	for(int i= 0; i<dense_bias_val.size();i++)
	{
        for(int j= 0; j<dense_bias_val[0].size();j++)
        {

           //cout<<numbers[i][j]<<" "; //uncomment it if you want to see the vector output where the dense kernel weights are stored

        }
       //cout<<endl; //uncomment it if uncomment previous line
	}
    cout<<endl<<"\n\nRow of dense bias value: "<<dense_bias_val.size()<<"\n\nColumn of dense bias value:"<<dense_bias_val[0].size()<<endl<<endl;

    return dense_bias_val;




}



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
//     bool Open_file = obj2.getFileContent();
     Image convolution_filter_1 = obj2.convolution_kernal();

     Matrix dense_kernel = obj2.dense_value();

     Matrix dense_bias = obj2.dense_bias_value();

     Matrix conv_bias= obj2.conv_bias_value();


     }

     return 0;




}
