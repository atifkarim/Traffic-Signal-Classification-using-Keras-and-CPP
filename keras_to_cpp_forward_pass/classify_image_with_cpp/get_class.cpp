#include<iostream>
#include "get_class.h"
#include <cmath>
#include <typeinfo>

using namespace std;

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


Image do_calculation :: applyFilter(Image &image, Image &filter, Matrix &conv_bias_weight)
{
  for(int row_c;row_c<conv_bias_weight.size();row_c++)
  {
    //    cout<<"myyyyyyyy conv bias:   ["<<row_c<<"][0]: "<<conv_bias_weight[row_c][0]<<endl;
  }

  //assert(image.size()==1 && filter.size()!=0);
  //  cout<<"\n\nBelow information stands for preprocessed image & filter"<<endl;

  int height = image[0].size();
//  uint16_t height = image[0].size();
//    cout<<"\npreprocessed image height: "<<height<<endl;

  int width = image[0][0].size();
//  uint16_t width = image[0][0].size();
    cout<<"\npreprocessed image width: "<<width<<endl;

  int filterHeight = filter[0].size();
//  uint16_t filterHeight = filter[0].size();
  //  cout<<"\nApplied convolution filter height or ROW: "<<filterHeight<<endl;

  int filterWidth = filter[0][0].size();
//  uint16_t filterWidth = filter[0][0].size();
  //  cout<<"\nApplied convolution filter width or COLUMN: "<<filterWidth<<endl;

  int feature_map=filter.size();
//  uint16_t feature_map=filter.size();
//    cout<<"feature map: "<<feature_map<<endl;

  //  cout<<"\nRow conv bias: "<<conv_bias_weight.size()<<endl;
  //  cout<<"\nColumn conv bias: "<<conv_bias_weight[0].size()<<endl;


  //  cout<<"\nBelow we will see after applying convolution image height and width.\nHere we have not used padding to the input image"<<endl;

  int newImageHeight = height-filterHeight+1;
//  uint16_t newImageHeight = height-filterHeight+1;
  newheight= newImageHeight;
  //  cout<<"\nAfter convolution image height: "<<newImageHeight<<endl;

  int newImageWidth = width-filterWidth+1;
//  uint16_t newImageWidth = width-filterWidth+1;
  newwidth= newImageWidth;
  //  cout<<"\nAfter convolution image width: "<<newImageWidth<<endl;

  int i,j,k,h,w,relu_applied,p;
//  double d;
  Image newImage(Image(feature_map,Matrix(newheight,Array(newwidth))));
  //  cout<<"OK";

  //  cout<<"RROW: "<<newImage.size()<<"COLL: "<<newImage[0].size()<<endl;
  //  Matrix newImage;
//  for(int row_c=0; row_c<conv_bias_weight.size(); row_c++)
//  {
//        cout<<"\n\n !!!!myyyyyyyy conv bias:   ["<<row_c<<"][0]: "<<conv_bias_weight[row_c][0]<<endl;

    for(k=0;k<feature_map;k++)
    {
      for (i=0 ; i<newImageHeight ; i++)
      {
        for (j=0 ; j<newImageWidth ; j++)
        {
          for (h=i ; h<i+filterHeight ; h++)
          {
            for (w=j ; w<j+filterWidth ; w++)
            {
              newImage[k][i][j] += (filter[k][h-i][w-j])*(image[0][h][w]);//+conv_bias_weight[row_c][0];

//                          cout<<"here conv bias applied: "<<conv_bias_weight[row_c][0]<<endl;
              //            newImage[k][i][j] += (filter[k][h-i][w-j])*(image[0][h][w]); //here zero for grayscale.
                                                   //If RGB then depth will come. Convolution algorithm will change
//                            cout<<newImage[k][i][j]<<" ";
            }
            p =newImage[k][i][j];
            double rel= p; //here this value is conv_kernel_bias
            relu_applied=relu(rel);
            newImage[k][i][j]= relu_applied;

          }
//          cout<<"new_image\n\n"<<newImage[i][j];
        }

      }
    }
//  }

//    cout<<"\n !!!--------------Convolution Finished-------------------- !!!\n";
//    cout<<"\nConvolved image Depth: "<<newImage.size()<<endl;
//    cout<<"\nConvolved image Row: "<<newImage[0].size()<<endl;
//    cout<<"\nConvolved image Column: "<<newImage[0][0].size()<<endl;

//  cout<<"\nDisplaying filter value"<<endl;
//  for(int k=0;k<filter.size();k++)
//  {
//    for(int x=0;x<filter[0].size();x++)
//    {   for(int y=0;y<filter[0][0].size();y++)

//      {
//                cout<<filter[k][x][y]<<" ";
//      }
//           cout<<endl;
//    }
//  }

      cout<<"\nDisplaying Convolved image's matrix before adding convolutional bias value"<<endl;
  for(int k=0;k<newImage.size();k++)
  {
    for(int x=0;x<newImage[0].size();x++)
    {   for(int y=0;y<newImage[0][0].size();y++)

      {
                cout<<newImage[k][x][y]<<" ";
      }
            cout<<endl;
    }
  }
//  cout<<"\nProcedure for adding convolutional bias"<<endl;
//  for(int row_c=0; row_c<conv_bias_weight.size(); row_c++)
//  {
  int row_c=0;

  for(int k=0;k<newImage.size() && row_c<conv_bias_weight.size();k++)
  {
//    cout<<"\nvalue of row_c now: "<<conv_bias_weight[row_c][0]<<endl;
    for(int x=0;x<newImage[0].size();x++)
    {   for(int y=0;y<newImage[0][0].size();y++)

      {
                newImage[k][x][y]+=newImage[k][x][y]+conv_bias_weight[row_c][0];
      }
//            cout<<endl;
    }
    row_c++;
//    cout<<"\nvalue of row_c later: "<<conv_bias_weight[row_c][0]<<endl;
  }
//}
//cout<<"\nDisplaying convolved image after adding convolutional bias value"<<endl;
  for(int k=0;k<newImage.size();k++)
  {
    for(int x=0;x<newImage[0].size();x++)
    {   for(int y=0;y<newImage[0][0].size();y++)

      {
                cout<<newImage[k][x][y]<<" ";
      }
            cout<<endl;
    }
  }

  //  cout<<newImage.size();
  //  cout<<endl;
  return newImage;
}

Matrix do_calculation :: resized_conv_relu_image(Image &new_im)

{

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

        //cout<<"sized_image: "<<sized_image[i][j]<<endl;
      }
    }

  }

  for (int i=0 ; i<sized_image.size() ; i++)
  {
    for(int j=0; j<sized_image[0].size();j++)
    {
              cout<<sized_image[i][j]<<"\n";

    }
    cout<<"I ran only once:"<<endl;
    //    cout<<"\n\nrow of resized_conv_relu image: "<<sized_image.size();
    //    cout<<"\n\ncolumn of resized_conv_relu image: "<<sized_image[0].size();

    //    cout<<"\n\nHere you can see that COLUMN of resized_conv_relu Matrix and ROW of dense_kernel Matrix is same\nSo, we can do here Matrix Multiplication.";
    //    cout<<"\nMatrix multiplication will be held in this way >>> \nresized_conv_relu X dense kernel\nHere, 'X' sign used for indicating multiplication.";

    //    cout<<endl;
  }

  return sized_image;
}

Matrix do_calculation :: matmul_dense_resized_conv_relu(Matrix &resized_relu, Matrix &dense_kernel_weight, Matrix &dense_bias_weight)
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

  //  cout << "\n\nOutput Matrix:" << endl;
  for(a = 0; a < resized_relu.size(); ++a)
  {
    for(b = 0; b < dense_kernel_weight[0].size(); ++b)
    {
      //      cout << multiply_dense_relu[a][b] << " ";
      if(b == dense_kernel_weight[0].size() - 1)
        cout << endl << endl;
      //				cout<<"hey man"<<endl;
    }
  }

  //  cout << "\n\nResultant matrix after adding dense_bias with matrix multiplied resized relu and dense_kernel matrix" << endl;
  for(int x=0;x<multiply_dense_relu.size();x++)
  {
    for(int y=0;y<multiply_dense_relu[0].size();y++)
    {
      //int temp=dense_bias[x][y] + multiply_dense_relu[x][y];
      multiply_dense_relu[x][y]= dense_bias_weight[y][x] + multiply_dense_relu[x][y];
      // multiply_dense_relu[x][y]=temp;
      //      cout<<multiply_dense_relu[x][y]<<" ";
      //      cout<<"d_val: ["<<y<<"]["<<x<<"] :"<<dense_bias_weight[y][x]<<endl;

      //            cout<<dense_bias[0][0]<<endl;

    }
  }
  //  cout<<endl;

  return multiply_dense_relu;

}

Matrix do_calculation :: softmax(Matrix &softmax_value)

{

  //  cout<<"\n-------Softmax function has started to work.Here you will also see the final result of Classification.------"<<endl;

  //  cout<<"\nROW of input array for softmax : "<<softmax_value.size();
  //  cout<<"\nCOLUMN of input array for softmax : "<<softmax_value[0].size()<<endl;


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
  //    cout<<softmax_output[0][b]<<" ";
  //    }
  //
  //cout<<"\none time run"<<endl;
  //}

  //std::vector< double > exp_soft;

  double exp_value_softmax;
  double exp_value_sum=0.0000000;

  //  cout<<"\nArray after doing exponential operation on the input array of Softmax Function"<<endl;

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
      //      cout<<softmax_output[i][j]<<"  ";
      exp_value_sum+=softmax_output[i][j];

      //softmax_value[i][j]=exp_value_softmax;
      //      cout<<softmax_value[i][j]<<" ";

    }
    //    cout<<"\nOne time run"<<endl;
    //    cout<<"Sum of all exp_value: "<<exp_value_sum<<endl;

  }
  //double x=0.00000;
  //  cout<<"\nOutput of Softmax algorithm"<<endl;
  for(int a=0;a<softmax_output.size();a++)
  {
    for(int b=0;b<softmax_output[0].size();b++)
    {
      softmax_output[a][b]= (softmax_output[a][b])/exp_value_sum;
//            cout<<softmax_output[a][b]<<" ";
      //        x+=softmax_output[a][b];

    }

    //    cout<<"\nOne time run"<<endl;
    //cout<<"x: "<<x;
    //cout<<"exp_value_sum: "<<exp_value_sum<<endl;
  }
  double x,y,z;
  x=0.000000;
  y=0.000000;
  z=0.000000;

  //  cout<<"----------Class of loaded Image---------"<<endl;
  for(int s=0;s<softmax_output.size();s++)
  {
    for(int t=0;t<softmax_output[0].size();t++)
    {
      x=softmax_output[s][t];
      if(x>z)
      {
//        cout<<"\npresent val: "<<x;
        z=x;
        y=t;
      }
      else
      {
      }
    }
    cout<<"\nClass is: "<<y;
  }
//  cout<<endl;

  return softmax_output;

}
