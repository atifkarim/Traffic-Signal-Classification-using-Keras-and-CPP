#ifndef GET_DATA_H
#define GET_DATA_H

#include <vector>
#include<string>
#include<cstring>
using namespace std;

bool getFileContent(std::string fileName, std::vector<std::string> & vecOfStrs); // function for opening text file and storing content as string

typedef vector<double> Array;
typedef vector<Array> Matrix;
typedef vector<Matrix> Image;

class Fetch_Data
{/*
private:
  std::vector<std::string> vecOfStr;
  string temp;
  long double found;

  int filter_number;
  std::string line;
  long double f;*/


public:


  Image convolution_kernal ();
  Matrix conv_bias_value ();
  Matrix dense_value ();
  Matrix dense_bias_value ();

};

#endif // GET_DATA_H
