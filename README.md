# Image-classification
### Image classification using KERAS and conversion of trained model for forward pass using Python(without using any deep learning library) and C++ to reduce the classification time
**Motivation** <br />
The objective of the task is to classify image using CNN and then implement the **testing part/forward pass** on FPGA using VHDL. As FPGA works with VHDL and don't support API of the CNN framework so I have tried to implement the Forward Pass using only Numpy library in Python and then convert it into C++. Priliminay, the work stands only for Grayscale Image. Padding hasn't used here. I have also choosen a minimum layer model(1 convolution layer and Dense layer). As dataset [GTSRB](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) image is used which anyone can download from [here](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html).

*Any contribution is highly appreciated.*

**Requirements:** <br />
A [requirements.txt](https://github.com/atifkarim/Image-classification/blob/master/requirements.txt) file is added <br />
In **Ubuntu 16.04** I have tested it.

**Introduction of source code** <br />
1/ Keras [training](https://github.com/atifkarim/Image-classification/blob/master/train_image_keras.py) and [testing](https://github.com/atifkarim/Image-classification/blob/master/test_image_keras.py) file is where for which a [configuration](https://github.com/atifkarim/Image-classification/blob/master/variable_config.json) is also introduced.
<br />
2/ After getting the traioned model from number 1 step [this file](https://github.com/atifkarim/Image-classification/blob/master/classify_image.py) will assist to fetch the information from the model file and store them in a text file which will be used later for doing the classification of image. This file consist no **deep learning library**.
<br />
3/ Now the [**C++**](https://github.com/atifkarim/Image-classification/tree/master/keras_to_cpp_forward_pass/classify_image_with_cpp) part.





Update: 11 January, 2019

Python code for training and testing(using keras API & scratch Numpy) is working flawlessly.

Check for training -- train_image_keras.py
Check for testing -- classify_image.py


Update 12 January, 2019

Work for converting "classify_image.py" to cpp has started. FIle name --- image_classification_with_cpp.cpp


Update 15 January, 2019

Work to make a complete C++ project for the classification. Name -- keras_to_cpp_forward_pass/classify_image_with_cpp

Update 16 January, 2019

Preprocess image with OpenCV for CPP and Python

Update 27 January, 2019

Calculated time for each step of classification
