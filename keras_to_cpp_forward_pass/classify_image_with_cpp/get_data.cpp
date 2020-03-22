#include<iostream>
#include "get_data.h"

#include<cstring>
#include<string>
//#include <vector>
#include <assert.h>
//#include <cmath>
#include<sstream>
#include <fstream>
#include <stdlib.h>
#include<math.h>

using namespace std;


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
  std::vector<std::string> vecOfStr;
  string temp;

  bool result = getFileContent("/home/atif/training_by_several_learning_process/number_classify/rgb_2_gray/Image-classification/trained_model_text_file/image_size_8_data_type_float_trained_model/conv_kernel.txt", vecOfStr);

  int num_ber=0;
  long double found;
  int filter_length = 3; // given value
  int filter_number;

  if(vecOfStr.size()%filter_length!=0) /* It will check whether the lines in convolution kernel text files and column number
                                       has the correct relation to get valid filter number*/

  {
    //    cout<<"\nRemainder after dividing line number with filter length is: "<<vecOfStr.size()%filter_length<<endl;
    //    cout<<"\nRemainder must be zero. Program terminate.\nChoose, suitable Filter length to make remainder ZERO"<<endl;
    abort();
  }

  filter_number=vecOfStr.size()/filter_length;

  Image conv_kernal_weight(Image(filter_number,Matrix(filter_length,Array())));

  int v=0;
  if(result)
  {
    for(int i=0;i<vecOfStr.size();i++)
    {
      if(num_ber<filter_length)
      {
//        cout<<"\nWho I am ??"<<endl;
        std::string line;
        line= vecOfStr[i];
        stringstream ss;
        ss<<line;
        while(!ss.eof())
        {
          ss>>temp;
          if(stringstream(temp)>>found)
          {
            long double f;
            f=found;
            f=f*1000; //use this line if you want to use integer value
            int int_val = (int)floor(f); //use this line if you want to use integer value
//            conv_kernal_weight[v][num_ber].push_back(f); //use this line if you want to use float value
            conv_kernal_weight[v][num_ber].push_back(int_val); //use this line if you want to use integer value
          }
        }

        num_ber++;
        //this loop will continue filter_length time then it will go to the next loop to check is there single or multi filter.
      }

      if(num_ber==filter_length && 1<filter_number)
      {
        //if this loop valid it will again go to the previous loop and store again kernel value

//                cout<<"EXECUTED & getting new filter value"<<endl;

        num_ber=0;
        filter_number=filter_number-1;
        v++;
//                cout<<"\nRemaining filter number is: "<<filter_number<<endl;
      }
    }
  }

  //  cout<<endl;

//cout<<"\nConvolutional kernel value after getting from text file"<<endl;
  for(int a=0;a<conv_kernal_weight.size();a++)
  {
    for(int b= 0; b<conv_kernal_weight[0].size();b++)
    {
      for(int c= 0; c<conv_kernal_weight[0][0].size();c++)
      {

//                        cout<<conv_kernal_weight[a][b][c]<<" "; //uncomment it if you want to see the vector output where the convolution kernel weights are stored

      }
//                  cout<<endl; //uncomment it if uncomment previous line
    }
  }
  //  cout<<endl;
//      cout<<endl<<"\n\nNumber of filter in Convolution kernal is: "<<conv_kernal_weight.size()<<"\n\nRow of convolution kernal is: "<<conv_kernal_weight[0].size()<<endl;
//      cout<<"\n\nColumn of Convolution kernal is: "<<conv_kernal_weight[0][0].size()<<endl;

  return conv_kernal_weight;
}


Matrix Fetch_Data :: conv_bias_value()

{
  std::vector<std::string> vecOfStr;
  string temp;
  long double found;

  bool result = getFileContent("/home/atif/training_by_several_learning_process/number_classify/rgb_2_gray/Image-classification/trained_model_text_file/image_size_8_data_type_float_trained_model/conv_bias.txt", vecOfStr);


  Matrix conv_bias_val(Matrix(vecOfStr.size(),Array()));
  //  std::vector<std::vector<long double> > numbers;

  if(result)
  {
    //     Print the vector contents
    for(int i=0;i<vecOfStr.size();i++)
    {
      std::string line;
      line= vecOfStr[i];
      stringstream ss;
      ss<<line;
      while(!ss.eof())
      {
        ss>>temp;
        if(stringstream(temp)>>found)
        {
          long double f;
          f=found;
          f=f*1000; //use this line if you want to use integer value
          int int_val = (int)floor(f); //use this line if you want to use integer value
//          cout<<"value of f is: "<<f<<endl;
//          conv_bias_val[i].push_back(f); //use this line if you want to use float value
          conv_bias_val[i].push_back(int_val); //use this line if you want to use integer value
        }
      }
    }
  }


  for(int i= 0; i<conv_bias_val.size();i++)
  {
    for(int j= 0; j<conv_bias_val[0].size();j++)
    {

//                  cout<<"\nconv_bias_value: "<<conv_bias_val[i][j]<<" "; //uncomment it if you want to see the vector output where the dense kernel weights are stored

    }
//            cout<<endl; //uncomment it if uncomment previous line
  }
//      cout<<endl<<"\n\nRow of conv bias value: "<<conv_bias_val.size()<<"\n\nColumn of conv bias value:"<<conv_bias_val[0].size()<<endl<<endl;

  return conv_bias_val;
}



Matrix Fetch_Data :: dense_value()

{
  std::vector<std::string> vecOfStr;
  string temp;
  long double found;

  bool result = getFileContent("/home/atif/training_by_several_learning_process/number_classify/rgb_2_gray/Image-classification/trained_model_text_file/image_size_8_data_type_float_trained_model/dense_kernel.txt", vecOfStr);


  Matrix dense_kernel_weight(Matrix(vecOfStr.size(),Array()));
  //  std::vector<std::vector<long double> > numbers;

  if(result)
  {
    //     Print the vector contents
    for(int i=0;i<vecOfStr.size();i++)
    {
      std::string line;
      line= vecOfStr[i];
      stringstream ss;
      ss<<line;
      while(!ss.eof())
      {
        ss>>temp;
        if(stringstream(temp)>>found)
        {
          long double f;
          f=found;
          f=f*1000; //use this line if you want to use integer value
          int int_val = (int)floor(f); //use this line if you want to use integer value
//          cout<<"value of f is: "<<f<<endl;
//          dense_kernel_weight[i].push_back(f); //use this line if you want to use float value
          dense_kernel_weight[i].push_back(int_val); //use this line if you want to use integer value
          //std::cout<<f<<std::endl;
        }
      }
    }
  }


  for(int i= 0; i<dense_kernel_weight.size();i++)
  {
    for(int j= 0; j<dense_kernel_weight[0].size();j++)
    {

//                  cout<<dense_kernel_weight[i][j]<<" "; //uncomment it if you want to see the vector output where the dense kernel weights are stored

    }
//            cout<<endl; //uncomment it if uncomment previous line
  }
//      cout<<endl<<"\n\nRow of dense kernel: "<<dense_kernel_weight.size()<<"\n\nColumn of dense kernel:"<<dense_kernel_weight[0].size()<<endl<<endl;

  return dense_kernel_weight;
}


Matrix Fetch_Data :: dense_bias_value()

{
  std::vector<std::string> vecOfStr;
  string temp;
  long double found;

  bool result = getFileContent("/home/atif/training_by_several_learning_process/number_classify/rgb_2_gray/Image-classification/trained_model_text_file/image_size_8_data_type_float_trained_model/dense_bias.txt", vecOfStr);


  Matrix dense_bias_value(Matrix(vecOfStr.size(),Array()));

  if(result)
  {
    //     Print the vector contents
    for(int i=0;i<vecOfStr.size();i++)
    {
      std::string line;
      line= vecOfStr[i];
      stringstream ss;
      ss<<line;
      while(!ss.eof())
      {
        ss>>temp;
        if(stringstream(temp)>>found)
        {
          long double f;
          f=found;
          f=f*1000; //use this line if you want to use integer value
          int int_val = (int)floor(f); //use this line if you want to use integer value
//          cout<<"value of f is: "<<int_val<<endl;
//          dense_bias_value[i].push_back(f); //use this line if you want to use float value
          dense_bias_value[i].push_back(int_val); //use this line if you want to use integer value
        }
      }
    }
  }


  for(int i= 0; i<dense_bias_value.size();i++)
  {
    for(int j= 0; j<dense_bias_value[0].size();j++)
    {

//                  cout<<dense_bias_value[i][j]<<" "; //uncomment it if you want to see the vector output where the dense kernel weights are stored

    }
//            cout<<endl; //uncomment it if uncomment previous line
  }
//      cout<<endl<<"\n\nRow of dense bias value: "<<dense_bias_value.size()<<"\n\nColumn of dense bias value:"<<dense_bias_value[0].size()<<endl<<endl;

  return dense_bias_value;
}
