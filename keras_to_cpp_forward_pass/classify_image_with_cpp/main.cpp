#include<iostream>
#include "get_data.h"

using namespace std;


int main()
{

     Fetch_Data obj2;

     Image convolution_filter_1 = obj2.convolution_kernal();

     Matrix conv_bias= obj2.conv_bias_value();

     Matrix dense_kernel = obj2.dense_value();

     Matrix dense_bias = obj2.dense_bias_value();




     return 0;




}
