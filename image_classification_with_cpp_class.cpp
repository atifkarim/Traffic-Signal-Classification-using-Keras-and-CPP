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

class Fetch_Data
{

public:

//    bool getFileContent(std::string fileName, std::vector<std::string> & vecOfStrs);
    Image convolution_kernal ();
    Matrix conv_bias_value ();
    Matrix dense_value ();
    Matrix dense_bias_value ();


};

double relu( double & a)
{
double modified_pix;
 if(a<= 0.0)
{
modified_pix=0.0;
}
else
{
modified_pix=a;
}
return modified_pix;

}


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

class do_calculation
{
public:
    Image applyFilter(Image &image, Image &filter, Matrix &conv_bias_weight);
    Matrix resized_conv_relu_image(Image &new_im);
    Matrix matmul_dense_resized_conv_relu(Matrix &resized_relu, Matrix &dense_kernel_weight, Matrix &dense_bias_weight);
    Matrix softmax(Matrix &softmax_value);
};


Image do_calculation :: applyFilter(Image &image, Image &filter, Matrix &conv_bias_weight)
{
cout<<"\n----------Apply Filter Function has started to work to do the convolution---------"<<endl;


    for(int row_c;row_c<conv_bias_weight.size();row_c++){
    cout<<"myyyyyyyy conv bias:   ["<<row_c<<"][0]: "<<conv_bias_weight[row_c][0]<<endl;
    }

    //assert(image.size()==1 && filter.size()!=0);
    cout<<"\n\nBelow information stands for preprocessed image & filter"<<endl;

    int height = image[0].size();
    cout<<"\npreprocessed image height: "<<height<<endl;
    int width = image[0][0].size();
    cout<<"\npreprocessed image width: "<<width<<endl;
    int filterHeight = filter[0].size();
    cout<<"\nApplied convolution filter height or ROW: "<<filterHeight<<endl;

    int filterWidth = filter[0][0].size();
    cout<<"\nApplied convolution filter width or COLUMN: "<<filterWidth<<endl;

    int feature_map=filter.size();
    cout<<"feat: "<<feature_map<<endl;

    cout<<"\nRow conv bias: "<<conv_bias_weight.size()<<endl;
    cout<<"\nColumn conv bias: "<<conv_bias_weight[0].size()<<endl;


    cout<<"\nBelow we will see after applying convolution image height and width.\nHere we have not used padding to the input image"<<endl;

    int newImageHeight = height-filterHeight+1;
    newheight= newImageHeight;
    cout<<"\nAfter convolution image height: "<<newImageHeight<<endl;

    int newImageWidth = width-filterWidth+1;
    newwidth= newImageWidth;
    cout<<"\nAfter convolution image width: "<<newImageWidth<<endl;

    int i,j,k,h,w,relu_applied,p;
    double d;
    Image newImage(Image(feature_map,Matrix(newheight,Array(newwidth))));
    cout<<"OK";

	//cout<<"RROW: "<<newImage.size()<<"COLL: "<<newImage[0].size()<<endl;
    //Matrix newImage;
    for(int row_c; row_c<conv_bias_weight.size(); row_c++){

    cout<<"\n\n !!!!myyyyyyyy conv bias:   ["<<row_c<<"][0]: "<<conv_bias_weight[row_c][0]<<endl;
for(k=0;k<feature_map;k++)
{
//        switch(k){
//
//        case 0:
//        d=2.01443624;
//        break;
//
//
//        case 1:
//        d=1.3674825;
//        break;
//
//        default:
//        break;
//        }
        for (i=0 ; i<newImageHeight ; i++)
        {
            for (j=0 ; j<newImageWidth ; j++)
            {
                for (h=i ; h<i+filterHeight ; h++)
                {
                    for (w=j ; w<j+filterWidth ; w++)
                    {





                        newImage[k][i][j] += (filter[k][h-i][w-j])*(image[0][h][w])+conv_bias_weight[row_c][0];
//                        cout<<"here conv bias in appy func: "<<conv_bias_weight[row_c][0]<<endl;




//                        newImage[k][i][j] += (filter[k][h-i][w-j])*(image[0][h][w]); //here zero for grayscale. If RGB then depth will come. Convolution algo will change
                      //  cout<<newImage[i][j];
                    }
                    p =newImage[k][i][j];
//                    cout<<"d: "<<d<<endl;
//                    d=d+(0.121489);
                    double rel= p; //here this value is conv_kernel_bias
                    relu_applied=relu(rel);
                    newImage[k][i][j]= relu_applied;
                    //(-2.121489);

                   // newImage[i][j]= d;

                }
                //cout<<"new_image\n\n"<<newImage[i][j];


            }

        }



        }}

        cout<<"\n !!!--------------Convolution Finished-------------------- !!!\n";
        cout<<"\nConvolved image Depth: "<<newImage.size()<<endl;
        cout<<"\nConvolved image Row: "<<newImage[0].size()<<endl;
        cout<<"\nConvolved image Column: "<<newImage[0][0].size()<<endl;

//        cout<<"\nDisplaying Convolved image's matrix"<<endl;
for(int k=0;k<newImage.size();k++){
        for(int x=0;x<newImage[0].size();x++)
        {   for(int y=0;y<newImage[0][0].size();y++)

                {
                cout<<newImage[k][x][y]<<" ";

                }
                cout<<endl;


        }}

    cout<<newImage.size();
    cout<<endl;
    return newImage;

}

Matrix do_calculation :: resized_conv_relu_image(Image &new_im)

{

    cout<<"\n---------Resizing of Convolved_Relued Image has started-------"<<endl;

Matrix sized_image(Matrix(1,Array(new_im.size()*new_im[0].size()*new_im[0][0].size())));
//sized_image=new_im;

//sized_image.resize(1);  // It means we have resized it for one row only //i can do also in this way
//sized_image[0].resize(newheight*newwidth); //It means we have made 46*46 column.
//Why we have made it?? for the matrix multiplication with dense kernel weight

//cout<<"\n\nrow of container for resized_conv_relu image: "<<sized_image.size();
//cout<<"\n\ncolumn of container for resized_conv_relu image: "<<sized_image[0].size();

int i=0;
int j=0;
//i=sized_image.size();

for(int m=0;m<new_im.size();m++)
{
for (int k=0 ; k<new_im[0].size() ; k++)

{
    for(int l=0; l<new_im[0][0].size();l++)
    {

        int value = new_im[m][k][l];
        sized_image[i][j]=value;
        j++;

            //cout<<"hii"<<sized_image[i][j]<<endl;

    }
}

}

for (int i=0 ; i<sized_image.size() ; i++)

{
    for(int j=0; j<sized_image[0].size();j++)
    {
//        cout<<sized_image[i][j];

    }
    //cout<<"I ran only once:"<<endl;
    cout<<"\n\nrow of resized_conv_relu image: "<<sized_image.size();
    cout<<"\n\ncolumn of resized_conv_relu image: "<<sized_image[0].size();

    cout<<"\n\nHere you can see that COLUMN of resized_conv_relu Matrix and ROW of dense_kernel Matrix is same\nSo, we can do here Matrix Multiplication.";
    cout<<"\nMatrix multiplication will be held in this way >>> \nresized_conv_relu X dense kernel\nHere, 'X' sign used for indicating multiplication.";

                //cout<<endl;
}

return sized_image;


}

Matrix do_calculation :: matmul_dense_resized_conv_relu(Matrix &resized_relu, Matrix &dense_kernel_weight, Matrix &dense_bias_weight)
{

        cout<<"\n\n---------------Matrix Multiplication between resized conv_relued & Dense kernal function has started to work-------------"<<endl;

Matrix multiply_dense_relu;
multiply_dense_relu.resize(1); //resized for 1 ROW
//multiply_dense_relu[0].resize(5); //resized for 5 COLUMN

multiply_dense_relu[0].resize(9); // class 9 so I need 9 column
int i, j, k;

	// Initializing elements of matrix mult to 0.
	for(i = 0; i < resized_relu.size(); ++i)
	{
		for(j = 0; j < dense_kernel_weight[0].size(); ++j)
		{
			multiply_dense_relu[i][j] = 0;
		}
	}

	// Multiplying matrix firstMatrix and secondMatrix and storing in array mult.
	int matmul_at=0;
	for(i = 0; i < resized_relu.size() ; ++i)
	{
		for(j = 0; j < dense_kernel_weight[0].size(); ++j)
		{
			for(k=0; k<resized_relu[0].size(); ++k)
			{
				multiply_dense_relu[i][j] += resized_relu[i][k] * dense_kernel_weight[k][j];
//				cout<<matmul_at<<"matmul"<<endl;
				matmul_at++;
			}
		}
	}



	int a,b;

	cout << "\n\nOutput Matrix:" << endl;
	for(a = 0; a < resized_relu.size(); ++a)
	{
		for(b = 0; b < dense_kernel_weight[0].size(); ++b)
		{
			cout << multiply_dense_relu[a][b] << " ";
			if(b == dense_kernel_weight[0].size() - 1)
				cout << endl << endl;
//				cout<<"hey man"<<endl;
		}
	}



	//Matrix dense_bias(1, Array(5));
	/*Matrix dense_bias(Matrix(1,Array(5))); //create matrix to store dense_kernel_bias value
	dense_bias[0][0]=-0.10792767;
	dense_bias[0][1]= 0.18778336;
	dense_bias[0][2]= -0.09822812;
	dense_bias[0][3] = -0.0668834;
	dense_bias[0][4] = 0.08525585;*/

//	Matrix dense_bias(Matrix(1,Array(9))); //class 9 ; so 9
//	dense_bias[0][0]= -1.3402408;
//	dense_bias[0][1]= -0.57729775;
//	dense_bias[0][2]= -0.3844931;
//	dense_bias[0][3] = 1.2625753;
//	dense_bias[0][4] = 1.2135713;
//
//	dense_bias[0][5]=1.3687891;
//	dense_bias[0][6]=-0.5227518;
//	dense_bias[0][7]=-0.9303516;
//	dense_bias[0][8]=-0.08981144;

    cout << "\n\nResultant matrix after adding dense_bias with matrix multiplied resized relu and dense_kernel matrix" << endl;
	for(int x=0;x<multiply_dense_relu.size();x++)
	{ for(int y=0;y<multiply_dense_relu[0].size();y++)

        {
            //int temp=dense_bias[x][y] + multiply_dense_relu[x][y];
           multiply_dense_relu[x][y]= dense_bias_weight[y][x] + multiply_dense_relu[x][y];
           // multiply_dense_relu[x][y]=temp;
            cout<<multiply_dense_relu[x][y]<<" ";
            cout<<"d_val: ["<<y<<"]["<<x<<"] :"<<dense_bias_weight[y][x]<<endl;

//            cout<<dense_bias[0][0]<<endl;

        }
    }
    cout<<endl;

return multiply_dense_relu;

}

Matrix do_calculation :: softmax(Matrix &softmax_value)

{

        cout<<"\n-------Softmax function has started to work.Here you will also see the final result of Classification.------"<<endl;

cout<<"\nROW of input array for softmax : "<<softmax_value.size();
cout<<"\nCOLUMN of input array for softmax : "<<softmax_value[0].size()<<endl;


Matrix softmax_output;
softmax_output.resize(1);
//softmax_output[0].resize(5);

softmax_output[0].resize(9); //class 9; so 9

//
//cout<<"beginning ROW softmax output: "<<softmax_output.size()<<endl;
//cout<<"beginning COL softmax output: "<<softmax_output[0].size()<<endl;

//cout<<"exp_value: ";
//for(int a=0;a<softmax_output.size();a++)
//{for(int b=0;b<softmax_output[0].size();b++)
//    {
//    cout<<"\nb: "<<b;
////    cout<<softmax_output[0][b]<<" ";
//    }
//
//cout<<"\none time run"<<endl;
//}

//std::vector< double > exp_soft;

double exp_value_softmax;
double exp_value_sum=0.0000000;

cout<<"\nArray after doing exponential operation on the input array of Softmax Function"<<endl;

for ( int i = 0; i < softmax_value.size(); i++ )
{
  for ( int j = 0; j < softmax_value[0].size(); j++ )
  {
//      cout<<softmax_value[i][j]<<" ";
      exp_value_softmax=softmax_value[i][j];
      exp_value_softmax=exp(exp_value_softmax);
//      cout<<"exp of: "<<softmax_value[i][j]<<" is: "<<exp_value_softmax<<"\n";
//      softmax_output[0].push_back(exp_value_softmax);
        softmax_output[i][j]=exp_value_softmax;
        cout<<softmax_output[i][j]<<"  ";
        exp_value_sum+=softmax_output[i][j];

      //softmax_value[i][j]=exp_value_softmax;
//      cout<<softmax_value[i][j]<<" ";

  }
  cout<<"\nOne time run"<<endl;
    cout<<"Sum of all exp_value: "<<exp_value_sum<<endl;

}
//double x=0.00000;
cout<<"\nOutput of Softmax algorithm"<<endl;
for(int a=0;a<softmax_output.size();a++)
{for(int b=0;b<softmax_output[0].size();b++)
    {
        softmax_output[a][b]= (softmax_output[a][b])/exp_value_sum;
        cout<<softmax_output[a][b]<<" ";
//        x+=softmax_output[a][b];

    }

cout<<"\nOne time run"<<endl;
//cout<<"x: "<<x;
//cout<<"exp_value_sum: "<<exp_value_sum<<endl;
}
double x,y,z;
x=0.000000;
y=0.000000;
z=0.000000;

cout<<"----------Ready to see the class of Input Image---------"<<endl;
for(int s=0;s<softmax_output.size();s++)
{
    for(int t=0;t<softmax_output[0].size();t++)
    {
        x=softmax_output[s][t];
        if(x>z){

        cout<<"\npresent val: "<<x;
        z=x;
        y=t;
        }
        else{
        }
    }
    cout<<"\nClass is: "<<y;
}




cout<<endl;

//cout<<"AAAAAA: "<<sum<<endl;
return softmax_output;

}






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
    {
     cv::Mat im = cv::imread(fn[k]);
//     const string c=fn[k];
     if (im.empty()) continue; //only proceed if sucsessful


     Image preprocessed_image = obj1.loadImage(fn[k]);
//     bool Open_file = obj2.getFileContent();
     Image convolution_filter_1 = obj2.convolution_kernal();

     Matrix conv_bias= obj2.conv_bias_value();

     Image convImage = obj3.applyFilter(preprocessed_image, convolution_filter_1, conv_bias);

     Matrix resized_conv_relu_image_value = obj3.resized_conv_relu_image(convImage);

     Matrix dense_kernel = obj2.dense_value();

     Matrix dense_bias = obj2.dense_bias_value();

     Matrix matmul_dense_resized_relu = obj3.matmul_dense_resized_conv_relu(resized_conv_relu_image_value,dense_kernel,dense_bias);

     Matrix softmax_calculation = obj3.softmax(matmul_dense_resized_relu);


     }

     return 0;




}
