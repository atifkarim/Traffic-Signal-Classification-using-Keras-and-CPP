/* Reading image to preprocess
*/
#ifndef GET_IMAGE_H
#define GET_IMAGE_H

//#include<cstring>
//#include<string>
#include <vector>

using namespace std;

typedef vector<double> Array;
typedef vector<Array> Matrix;
typedef vector<Matrix> Image;
//typedef vector<Image> Container;

class Get_Image
{
public:

  Image loadImage(const string &filename);
};

#endif // GET_IMAGE_H
