
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

const int kNewWidth =48;
const int kNewHeight =48;

int newheight;
int newwidth;

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


Matrix dense_value (){

std::vector<std::string> vecOfStr;



    string temp;

    long double found;

    bool result = getFileContent("/home/atif/traffic_cl_9_ep_500_gray_for_cpp_dense_new_exp_val.txt", vecOfStr);


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
    cout<<endl;
    cout<<endl<<"\n\nRow of dense kernel: "<<numbers.size()<<"\n\nColumn of dense kernel:"<<numbers[0].size()<<endl;

    return numbers;

}




int relu( int & a)
{
int modified_pix;
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




Matrix convkernal(int height, int width)
{
   Matrix kernal(height, Array(width));
   /* kernal[0][0]=-1.2638985;
    kernal[0][1]=-0.4732446;
    kernal[0][2]=-0.6323778;
    kernal[1][0]=-0.4970976;
    kernal[1][1]=-1.2289356;
    kernal[1][2]=-0.6131264;
    kernal[2][0]=-1.1365112;
    kernal[2][1]=-0.47544488;
    kernal[2][2]=-0.5749136;*/

    kernal[0][0]= -2.1352224;
    kernal[0][1]= -0.6845135;
    kernal[0][2]= 0.19564846;
    kernal[1][0]= 0.06378726;
    kernal[1][1]= 1.8066058;
    kernal[1][2]= 3.5368507;
    kernal[2][0]= -1.8619282;
    kernal[2][1]=  -0.2793754;
    kernal[2][2]=  2.145677;


  /*  kernal[0][0]=1;
    kernal[0][1]=0;
    kernal[0][2]=-1;
    kernal[1][0]=0;
    kernal[1][1]=0;
    kernal[1][2]=0;
    kernal[2][0]=-1;
    kernal[2][1]=0;
    kernal[2][2]=1;*/

    return kernal;
}

Matrix loadImage(const char *filename)
{

    Mat I = imread(filename, 1);
    cout<<"\nI size: "<<I.size()<<endl;
    Mat E;
    cv::cvtColor(I, E, CV_RGB2GRAY);
    cout<<"\nE size: "<<E.size()<<endl;

//    Mat Norm_img;
    E.convertTo(E, CV_64F, 1.0 / 255, 0);
//
    cout<<"\nNormalized IMAGE size: "<<E.size()<<endl;
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

    Matrix image_1;

    image_1.resize(kNewHeight*kNewWidth);
    double *ptrDst[new_mat.rows];
    //Array val;

    for(int i = 0; i < new_mat.rows; ++i)
    {


        ptrDst[i] = new_mat.ptr<double>(i);
       // cout<<ptrDst[i];
       for(int j = 0; j < new_mat.cols; ++j)
       {

        double value = ptrDst[i][j];
        image_1[i].push_back(value);

        }
        //cout<<"first row";
    }

    cout<<"\nLoaded/Input Image Row: "<<image_1.size()<<endl;
    cout<<"\nLoaded/Input Image column: "<<image_1[0].size()<<endl;

//    for(int a=0;a<image_1.size();a++){
//    for(int b=0;b<image_1[0].size();b++)
//    {
//
//    cout<<"["<<a<<"]["<<b<<"]"<<image_1[a][b]<<endl;
//
//    }
//
//
//    }
    cout<<"FINISH!!"<<endl;



    return image_1;


}


Matrix applyFilter(Matrix &image, Matrix &filter){

    //assert(image.size()==1 && filter.size()!=0);
    cout<<"\n\nBelow information stands for preprocessed image & filter"<<endl;

    int height = image[0].size();
    cout<<"\npreprocessed image height: "<<height<<endl;
    int width = image[0].size();
    cout<<"\npreprocessed image width: "<<width<<endl;
    int filterHeight = filter.size();
    cout<<"\nApplied convolution filter height or ROW: "<<filterHeight<<endl;

    int filterWidth = filter[0].size();
    cout<<"\nApplied convolution filter width or COLUMN: "<<filterWidth<<endl;

    cout<<"\nBelow we will see after applying convolution image height and width.\nHere we have not used padding to the input image"<<endl;

    int newImageHeight = height-filterHeight+1;
    newheight= newImageHeight;
    cout<<"\nAfter convolution image height: "<<newImageHeight<<endl;

    int newImageWidth = width-filterWidth+1;
    newwidth= newImageWidth;
    cout<<"\nAfter convolution image width: "<<newImageWidth<<endl;

    int d,i,j,h,w,relu_applied;

    Matrix newImage(Matrix(newImageHeight,Array(newImageWidth)));
    //Matrix newImage;

        for (i=0 ; i<newImageHeight ; i++)
        {
            for (j=0 ; j<newImageWidth ; j++)
            {
                for (h=i ; h<i+filterHeight ; h++)
                {
                    for (w=j ; w<j+filterWidth ; w++)
                    {
                        newImage[i][j] += (filter[h-i][w-j])*(image[h][w]);
                      //  cout<<newImage[i][j];
                    }
                    d=newImage[i][j];
//                    d=d+(0.121489);
                    d=d+(-0.00792259);   //here this value is conv_kernel_bias
                    relu_applied=relu(d);
                    newImage[i][j]= relu_applied; //(-2.121489);
                   // newImage[i][j]= d;

                }
                //cout<<"new_image\n\n"<<newImage[i][j];


            }

        }

        cout<<"\n !!!--------------Convolution Finished-------------------- !!!\n";
        cout<<"\nConvolved image ROW: "<<newImage.size()<<endl;
        cout<<"\nConvolved image COLUMN: "<<newImage[0].size()<<endl;

//        cout<<"\nDisplaying Convolved image's matrix"<<endl;

        for(int x=0;x<newImage.size();x++)
        {   for(int y=0;y<newImage[0].size();y++)

                {
//                cout<<newImage[x][y]<<" ";

                }
//                cout<<endl;


        }

    //cout<<newImage.size();
    cout<<endl;
    return newImage;

}

Matrix resized_conv_relu_image(Matrix &new_im){

Matrix sized_image;
//sized_image=new_im;

sized_image.resize(1);  // It means we have resized it for one row only
sized_image[0].resize(newheight*newwidth); //It means we have made 46*46 column.
//Why we have made it?? for the matrix multiplication with dense kernel weight

cout<<"\nRow of container for resized_conv_relu image: "<<sized_image.size();
cout<<"\nColumn of container for resized_conv_relu image: "<<sized_image[0].size();

int i=0;
int j=0;
//i=sized_image.size();


for (int k=0 ; k<new_im.size() ; k++)

{
    for(int l=0; l<new_im[0].size();l++)
    {

        int value = new_im[k][l];
        sized_image[i][j]=value;
        j++;

            //cout<<"hii"<<sized_image[i][j]<<endl;

    }
}

for (int i=0 ; i<sized_image.size() ; i++)

{
    for(int j=0; j<sized_image[0].size();j++)
    {
//        cout<<sized_image[i][j];

    }
    //cout<<"I ran only once:"<<endl;
    cout<<"\n\nRow of resized_conv_relu image: "<<sized_image.size();
    cout<<"\n\nColumn of resized_conv_relu image: "<<sized_image[0].size();

    cout<<"\n\nHere you can see that COLUMN of resized_conv_relu Matrix and ROW of dense_kernel Matrix is same\nSo, we can do here Matrix Multiplication.";
    cout<<"\nMatrix multiplication will be held in this way >>> \nresized_conv_relu X dense kernel\nHere, 'X' sign used for indicating multiplication.";

                //cout<<endl;
}

return sized_image;
}




Matrix matmul_dense_resized_conv_relu(Matrix &resized_relu, Matrix &dense_kernel_weight)
{

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
	for(i = 0; i < resized_relu.size() ; ++i)
	{
		for(j = 0; j < dense_kernel_weight[0].size(); ++j)
		{
			for(k=0; k<resized_relu[0].size(); ++k)
			{
				multiply_dense_relu[i][j] += resized_relu[i][k] * dense_kernel_weight[k][j];
			}
		}
	}



	int a,b;

	cout << "\n\nOutput Matrix after multiplication between relu o/p and dense kernal:" << endl;
	for(a = 0; a < resized_relu.size(); ++a)
	{
		for(b = 0; b < dense_kernel_weight[0].size(); ++b)
		{
//			cout << multiply_dense_relu[a][b] << " ";
			if(b == dense_kernel_weight[0].size() - 1)
				cout << endl << endl;
		}
	}

	//Matrix dense_bias(1, Array(5));
	/*Matrix dense_bias(Matrix(1,Array(5))); //create matrix to store dense_kernel_bias value
	dense_bias[0][0]=-0.10792767;
	dense_bias[0][1]= 0.18778336;
	dense_bias[0][2]= -0.09822812;
	dense_bias[0][3] = -0.0668834;
	dense_bias[0][4] = 0.08525585;*/

	Matrix dense_bias(Matrix(1,Array(9))); //class 9 ; so 9
	dense_bias[0][0]= -1.3402408;
	dense_bias[0][1]= -0.57729775;
	dense_bias[0][2]= -0.3844931;
	dense_bias[0][3] = 1.2625753;
	dense_bias[0][4] = 1.2135713;

	dense_bias[0][5]=1.3687891;
	dense_bias[0][6]=-0.5227518;
	dense_bias[0][7]=-0.9303516;
	dense_bias[0][8]=-0.08981144;

    cout << "\n\nResultant matrix after adding dense_bias with matrix multiplied resized relu and dense_kernel matrix" << endl;
	for(int x=0;x<multiply_dense_relu.size();x++)
	{ for(int y=0;y<multiply_dense_relu[0].size();y++)

        {
            //int temp=dense_bias[x][y] + multiply_dense_relu[x][y];
           multiply_dense_relu[x][y]= dense_bias[x][y] + multiply_dense_relu[x][y];
           // multiply_dense_relu[x][y]=temp;
            cout<<multiply_dense_relu[x][y]<<" ";

//            cout<<dense_bias[0][0]<<endl;

        }
    }
    cout<<endl;

return multiply_dense_relu;
}




Matrix softmax(Matrix &softmax_value)

{

cout<<"\nROW of container/input array for softmax : "<<softmax_value.size();
cout<<"\nCOLUMN of container/input array for softmax : "<<softmax_value[0].size()<<endl;


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

cout<<"\nOutput after doing exponential operation on Matmul dense resized relu: "<<endl;

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
        cout<<softmax_output[i][j]<<" ";
        exp_value_sum+=softmax_output[i][j];

      //softmax_value[i][j]=exp_value_softmax;
//      cout<<softmax_value[i][j]<<" ";

  }
  cout<<"\nOne time run"<<endl;
    cout<<"exp_value_sum: "<<exp_value_sum<<endl;

}
//double x=0.00000;
cout<<"\n\nSoftmax Output Matrix: "<<endl;
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

cout<<"\nNow you will see in which class the input image will go"<<endl;
for(int s=0;s<softmax_output.size();s++)
{
    for(int t=0;t<softmax_output[0].size();t++)
    {
        x=softmax_output[s][t];
        if(x>z){

        cout<<"Value of ["<<s<<"]["<<t<<"] is: "<<x;
        z=x;
        y=t;
        }
        else{
        }
    }
    cout<<"\nCalculated Class is: "<<y;
}




cout<<endl;

//cout<<"AAAAAA: "<<sum<<endl;
return softmax_output;
}




int main()
{


    Matrix conv_kernal = convkernal(3, 3);

    Matrix preprocessed_image = loadImage("/home/atif/image_classification_c++/thirty_speed.ppm");

    Matrix convImage = applyFilter(preprocessed_image, conv_kernal);

    Matrix dense_kernel = dense_value();

    Matrix resized_conv_relu_image_value = resized_conv_relu_image(convImage);

    Matrix matmul_dense_resized_relu = matmul_dense_resized_conv_relu(resized_conv_relu_image_value,dense_kernel);

    Matrix softmax_calculation = softmax(matmul_dense_resized_relu);

    for(int i=0;i<298;i++)
    {
    for(int j=0;j<298;j++)
    {
//    cout<<newImage[i][j];


    }
    //cout<<endl;

    }


    //declare one new matrix named frame to store the image(convolved with filter and then relu_applied.This image is not resized)
    Mat frame(newheight,newwidth,CV_64F);

    for(int q=0;q<newheight;q++)
    {
    for(int w=0;w<newwidth;w++)
    {


    frame.at<double>(q,w)=(double)convImage[q][w];


    }
    }

    cout<<"\nframe size: "<<frame.size();
//    cout<<"\nframe col: "<<frame[0].size();
    cout<<endl;


//    memcpy(A.data, image_1.data(), image_1.size()*sizeof(double));

    imwrite("/home/atif/image_classification_c++/conv_im_tanbir.jpg",frame);
    Mat B= imread("/home/atif/image_classification_c++/conv_im_tanbir.jpg",CV_LOAD_IMAGE_ANYDEPTH);
    imshow("conv_image",B);

//    Mat R;
//    cv::cvtColor(B, R, CV_RGB2GRAY);
//    imwrite("/home/atif/image_classification_c++/conv_im_tanbir_rgb.jpg",R);
////    Mat X= imread("/home/atif/image_classification_c++/conv_im_tanbir_rgb.jpg",1);
//    imshow("conv_image",X);


    //cout<<frame;
    waitKey();






}
