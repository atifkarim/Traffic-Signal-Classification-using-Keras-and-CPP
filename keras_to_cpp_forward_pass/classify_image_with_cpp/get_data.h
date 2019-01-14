#ifndef GET_DATA_H
#define GET_DATA_H

#include <vector>
using namespace std;
/

bool getFileContent(std::string fileName, std::vector<std::string> & vecOfStrs);

typedef vector<double> Array;
typedef vector<Array> Matrix;
typedef vector<Matrix> Image;
typedef vector<Image> Container;


class Fetch_Data
{

public:

    Image convolution_kernal ();
    Matrix conv_bias_value ();
    Matrix dense_value ();
    Matrix dense_bias_value ();


};



#endif // GET_DATA_H
