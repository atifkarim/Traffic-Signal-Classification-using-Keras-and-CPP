# Image-classification
Image classification using Keras and implement predict function using Python and C++ for FPGA

The objective of the task is to classify image using CNN and then implement the testing part/Forward pass on FPGA using VHDL. As FPGA works with VHDL and don't support API of the CNN framework so I have tried to implement the Forward Pass using only Numpy library in Python and then convert it into C++. Priliminay, the work stands only for Grayscale Image. Padding hasn't used here. I have also choosen a minimum layer model(1 convolution layer and Dense layer). Any contribution is highly appreciated.

Update: 11 January, 2019

Python code for training and testing(using keras API & scratch Numpy) is working flawlessly.

Check for training -- train_image_keras.py
Check for testing -- classify_image.py


Update 12 January, 2019

Work for converting "classify_image.py" to cpp has started. FIle name --- image_classification_with_cpp.cpp


Update 15 January, 2019

Work to make a complete C++ project for the classification. Name -- keras_to_cpp_forward_pass/classify_image_with_cpp
