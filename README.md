# Image-classification
Image classification using Keras and implement predict function using Python and C++ for FPGA

The objective of the task is to classify image using CNN and then implement the testing part/Forward pass on FPGA using VHDL. As FPGA works with VHDL and don't support API of the CNN framework so I have tried to implement the Forward Pass using only Numpy library in Python and then convert it into C++. Priliminay, the work stands only for Grayscale Image. Padding hasn't used here. I have also choosen a minimum layer model(1 convolution layer and Dense layer). Any contribution is highly appreciated.
