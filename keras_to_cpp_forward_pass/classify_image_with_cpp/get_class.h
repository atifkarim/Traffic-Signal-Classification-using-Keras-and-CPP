#ifndef GET_CLASS_H
#define GET_CLASS_H


#include <vector>
using namespace std;

typedef vector<double> Array;
typedef vector<Array> Matrix;
typedef vector<Matrix> Image;

using namespace std;

double relu( double & a);

class do_calculation
{
public:


  int newheight;
  int newwidth;

  Image applyFilter(Image &image, Image &filter, Matrix &conv_bias_weight);
  Matrix resized_conv_relu_image(Image &new_im);
  Matrix matmul_dense_resized_conv_relu(Matrix &resized_relu, Matrix &dense_kernel_weight, Matrix &dense_bias_weight);
  Matrix softmax(Matrix &softmax_value);
};

#endif // GET_CLASS_H
