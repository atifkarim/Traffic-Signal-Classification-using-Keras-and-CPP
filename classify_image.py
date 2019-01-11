#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 21:50:57 2019

@author: atif
"""

#############################
##Importing Library##########
#############################

import numpy as np
from skimage import io, color, exposure, transform
from skimage.color import rgb2gray
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split  #it came from update scikit learn. https://stackoverflow.com/questions/40704484/importerror-no-module-named-model-selection
import os
import glob
import h5py

from matplotlib import pyplot as plt
#%matplotlib inline

NUM_CLASSES = 9 #Used class for the training
IMG_SIZE = 48 #required size. This size has also maintained during training. User defined value

########################################################################
#############Extracting weight from the trained model file##############
########################################################################

import pandas as pd
from keras.models import load_model

model = load_model('/home/atif/image_classification_c++/multi_filter_cpp/traffic_2_filter_no_pad_gray_ep_100_for_cpp.h5')

layer_list =[]
# f = open('/home/atif/path_for_storing_all_layer_info.txt', 'w') #uncomment it if you want to store all layer info at a time.
for layer in model.layers:
    g=layer.get_config()
    h=layer.get_weights()
    
    layer_list.append(h)
#     print ("g== ",g,"\n") #for printing layer name and verbal info

#     print ("h== ",h,"\n\n") # for printing layer numeric value, eg: weight, bias value
#     print("type_of g == ",type(g),"\n")
#     print("type_of h == ",type(h),"\n")

# below lines till f.close() used for writing in text file. To do this you have to uncomment the above line started with f.open() also.

#     g1=str(g) # declaring a string variable g1 to store the info of g
#     h1=str(h) #declaring a string variable h1 to store the info of h
#     g_type=str(type(g)) #declaring a string variable g1 to store the type of g
#     h_type=str(type(h)) #declaring a string variable h1 to store the type of h
    
#     f.write("layer_definition: "+g1+"\n\n")
#     f.write("layer_type: "+g_type+"\n\n")
#     #f.write("\n")
#     f.write("layer_weight: "+h1+"\n\n")
#     f.write("weight_type: "+h_type+"\n\n\n")
#     f.write("\n")
    
# f.close()

# layer_name=['conv_layer','flatten_layer','dense_layer']



#From here the code has started which will extract every layer's info which you can use further in this file
        
conv_kernel=layer_list[0][0]
conv_kernel=conv_kernel.transpose()
print("conv_kernel: \n",conv_kernel,"\n\n")
print("conv_kernel shape:\t",conv_kernel.shape,"\n\n")
print("conv kernel dimension:\t",conv_kernel.ndim,"\n\n")
print("type_conv_kernel:",type(conv_kernel),"\n")

#conv_kernel_reshape=conv_kernel.reshape(conv_kernel[3],conv_kernel[2],conv_kernel[1],conv_kernel[0])
#print("re:  ",conv_kernel_reshape.shape)



conv_bias=layer_list[0][1]
print("conv_bias_value: ",conv_bias)
print("conv_bias ndim: ",conv_bias.ndim,"\n\n")



dense_kernel=layer_list[2][0]
print("dense_kernel: \n",dense_kernel,"\n\n")
print("dense_kernel shape:\t",dense_kernel.shape,"\n\n")
print("dense_kernel dimension:\t",dense_kernel.ndim,"\n\n")
print("type_dense_kernel:",type(dense_kernel),"\n")
print("dense_kernel size: ",dense_kernel.size,"\n")
# dense_1_transpose=dense__1.transpose()
# print("dense_1_transpose: ",dense_1_transpose,"\n\n")


dense_bias=layer_list[2][1]
print("dense_bias: ",dense_bias)
print("dense_bias_shape: ",dense_bias.shape)
dense_bias=dense_bias.reshape(1,9) # here chenge 5 to the number of your used class
print("dense_bias_shape: ",dense_bias.shape)
# print(dense_2[0])


#########################################################################
#####Storing convolution kernel##########################################
#########################################################################

conv_kernel=layer_list[0][0]
conv_kernel=conv_kernel.transpose() # This has made to print it like a Matrix form. 
i_list=[]
for i in conv_kernel:
#     print(i)
    i_list.append(i)
#     for k in i:
#         print(k)
#         i_list.append(k)
# print(i_list)
i_list_array=[]
i_list_array=np.array(i_list)
# print(i_list_array.shape)
# i_list_array=i_list_array.reshape(2,3,3)
# print(i_list_array.ndim)

for p in i_list_array:
#     for a in p:
#         print(a)
    print(p)
    ww=str(p)
    ww=ww.replace('[','')
    ww=ww.replace(']','')
    f=open('/home/atif/conv_k_spy.txt','a')
    f.write(ww)
    f.write("\n")
f.close()

#####################################################################
######Storing convolution bias#######################################
#####################################################################

conv_bias=layer_list[0][1]
conv_bias_list=[]
for i in conv_bias:
    conv_bias_list.append(i)

conv_bias_array=[]
conv_bias_array=np.array(conv_bias_list)
print(conv_bias_array)
np.savetxt('/home/atif/conv_b_spy.txt', conv_bias_array, fmt='%1.8e',delimiter=' ')

for x in conv_bias_array:
    print(x)
    
    


#####################################################################
#####Storing dense kernel weight#####################################
#####################################################################
    
dense_kernel=layer_list[2][0]
i_list=[] #declare a list to store the weight of dense kernel
for i in dense_kernel:
#     print(i)
    i_list.append(i) #appended it in the declared list
#     for k in i:
# #         print(k)
#         i_list.append(k)
# print(i_list)
i_list_array=[] #declared an array
i_list_array=np.array(i_list) # store the value of list in the array
print(i_list_array)
np.savetxt('/home/atif/dense_k_spy.txt', i_list_array, fmt='%1.8e',delimiter=' ') #writing on a text file from array
# %.8f
# fmt='%1.8e' #add this above line after i_list_aray


#####################################################################
########Storing dense bias###########################################
#####################################################################

dense_bias=layer_list[2][1]
dense_bias_list=[]
for i in dense_bias:
    dense_bias_list.append(i)

dense_bias_array=[]
dense_bias_array=np.array(dense_bias_list)
print(dense_bias_array)
np.savetxt('/home/atif/dense_b_spy.txt', dense_bias_array, fmt='%1.8e',delimiter=' ')
