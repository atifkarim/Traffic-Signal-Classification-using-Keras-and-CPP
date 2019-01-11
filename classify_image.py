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
import sys
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


#########################################################################
#####Storing convolution kernel, needed for cpp testing##################
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
    #f=open('/home/atif/conv_k_spy.txt','a') #uncomment from here till f.close() if you want to save text file
    #f.write(ww)
    #f.write("\n")
#f.close()

#####################################################################
######Storing convolution bias, needed for cpp testing ##############
#####################################################################

conv_bias=layer_list[0][1]
conv_bias_list=[]
for i in conv_bias:
    conv_bias_list.append(i)

conv_bias_array=[]
conv_bias_array=np.array(conv_bias_list)
#print(conv_bias_array)
#np.savetxt('/home/atif/conv_b_spy.txt', conv_bias_array, fmt='%1.8e',delimiter=' ')

#for x in conv_bias_array:
    #print(x)
    
    


#####################################################################
#####Storing dense kernel weight, needed for cpp testing ############
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
#print(i_list_array)
#np.savetxt('/home/atif/dense_k_spy.txt', i_list_array, fmt='%1.8e',delimiter=' ') #writing on a text file from array

# %.8f #you can use it to get float value
# fmt='%1.8e' #add this above line after i_list_aray


#####################################################################
########Storing dense bias, needed for cpp testing ##################
#####################################################################

dense_bias=layer_list[2][1]
dense_bias_list=[]
for i in dense_bias:
    dense_bias_list.append(i)

dense_bias_array=[]
dense_bias_array=np.array(dense_bias_list)
#print(dense_bias_array)
#np.savetxt('/home/atif/dense_b_spy.txt', dense_bias_array, fmt='%1.8e',delimiter=' ')



#####################################################################
######### Reshaping convolution kernel for further process###########
#####################################################################

print("conv_kernel:\n",conv_kernel,"\n")
print("conv_kernel_shape:",conv_kernel.shape,"\tconv_kernel ndim:",conv_kernel.ndim,"\n")
print("length of conv_kernel:",len(conv_kernel),"\n")

conv_kernel_reshape=conv_kernel.reshape(2,3,3) # 2 for 2 filter. change it according to your filter number
print("conv_kernel_reshape:\n",conv_kernel_reshape,"\n")
print("conv_kernel_reshape shape:",conv_kernel_reshape.shape,"\tconv_kernel_reshape ndim:",conv_kernel_reshape.ndim,"\n")
print("length of conv_kernel_reshape:",len(conv_kernel_reshape[0]),"\n")

convolution_kernel_filter=[]
convolution_kernel_filter=np.zeros((2,3,3)) # 2 for 2 filter. change it according to your filter number
convolution_kernel_filter[:,:,:]=np.array(conv_kernel_reshape)
print("convolution_kernel_filter: \n",convolution_kernel_filter,"\n")
print("convolution_kernel_filter shape:",convolution_kernel_filter.shape,"\tconvolution_kernel_filter ndim:",convolution_kernel_filter.ndim,"\n")
print("length of convolution_kernel_filter:",len(convolution_kernel_filter),"\n")




#########################################################################################
################# Code for different steps for classification############################
#########################################################################################


def preprocess_img(img):
#     Histogram normalization in y
#     hsv = color.rgb2hsv(img)
#     hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
#     img = color.hsv2rgb(hsv)

    # central scrop
    min_side = min(img.shape[:-1])
    centre = img.shape[0]//2, img.shape[1]//2
    img = img[centre[0]-min_side//2:centre[0]+min_side//2,
              centre[1]-min_side//2:centre[1]+min_side//2,
              :]
    img = rgb2gray(img)

    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

    # roll color axis to axis 0
    img = np.rollaxis(img,-1)

    return img

def conv_(img, conv_filter):
    print("\nconv_ function start to work\n")
    filter_size = conv_filter.shape[1] #output is 3
    print("filter_size: ",filter_size)
    print("img shape: ",img.shape)
    result = np.zeros((img.shape))
    #Looping through the image to apply the convolution operation.
    
    for x in conv_bias_array:  # to get the value of convolution bias
        print("i am the conv_bias: ",x)
        
        for r in np.uint16(np.arange(filter_size/2.0,img.shape[0]-filter_size/2.0+1)):
        
            for c in np.uint16(np.arange(filter_size/2.0,img.shape[1]-filter_size/2.0+1)):
            
                curr_region = img[r-np.uint16(np.floor(filter_size/2.0)):r+np.uint16(np.ceil(filter_size/2.0)), c-np.uint16(np.floor(filter_size/2.0)):c+np.uint16(np.ceil(filter_size/2.0))]
        
                #Element-wise multipliplication between the current region and the filter.
            
                curr_result = curr_region * conv_filter
#             print("curr_result: ",curr_result)
                curr_result= curr_result+x
                
#             print("conv_bias: ",conv_bias_new)
#                 print("new curr res: ",curr_result)
                conv_sum = np.sum(curr_result) #Summing the result of multiplication.
                result[r, c] = conv_sum#Saving the summation in the convolution layer feature map.
#             print("conv_sum_shape: ",conv_sum.shape)
        print("now x is: ",x)
        #print(curr_region)
    #Clipping the outliers of the result matrix.
    final_result = result[np.uint16(filter_size/2.0):result.shape[0]-np.uint16(filter_size/2.0),np.uint16(filter_size/2.0):result.shape[1]-np.uint16(filter_size/2.0)]
    print("\nconv_ function finish\n")
    return final_result



def conv(img, conv_filter):
#     print("conv_filter:: ", conv_filter.shape[0])
    
    if len(img.shape) > 2 or len(conv_filter.shape) > 3: # Check if number of image channels matches the filter depth.
        if img.shape[-1] != conv_filter.shape[-1]:
            print("Error: Number of channels in both image and filter must match.")
            sys.exit()
    if conv_filter.shape[1] != conv_filter.shape[2]: # Check if filter dimensions are equal.
        print('Error: Filter must be a square matrix. I.e. number of rows and columns must match.')
        sys.exit()
    if conv_filter.shape[1]%2==0: # Check if filter diemnsions are odd.
        print('Error: Filter must have an odd size. I.e. number of rows and columns must be odd.')
        sys.exit()

    # An empty feature map to hold the output of convolving the filter(s) with the image.
    feature_maps = np.zeros((img.shape[0]-conv_filter.shape[1]+1, img.shape[1]-conv_filter.shape[1]+1, conv_filter.shape[0]))

    # Convolving the image by the filter(s).
    
    for filter_num in range(conv_filter.shape[0]):
#         print("filter num: ",filter_num)
        print("Filter ", filter_num + 1)
        curr_filter = conv_filter[filter_num, :] # getting a filter from the bank.
#         print(curr_filter)
        print("curr_fiter_shape: ",curr_filter.shape)
#         print("length of curr_fiter_shape: ",len(curr_filter.shape))


        if len(curr_filter.shape) > 2:
            print("CONV function has worked")
            conv_map = conv_(img[:, :, 0], curr_filter[:, :, 0]) # Array holding the sum of all feature maps.
            for ch_num in range(1, curr_filter.shape[-1]): # Convolving each channel with the image and summing the results.
                conv_map = conv_map + conv_(img[:, :, ch_num], 
                                  curr_filter[:, :, ch_num])
        else: # There is just a single channel in the filter.
            print("go to conv_ function ")
            conv_map = conv_(img, curr_filter)
            print(conv_map)
            
        feature_maps[:, :, filter_num] = conv_map # Holding feature map with the current filter.
#         print("feature_maps from conv_map: ",feature_maps)
    return feature_maps # Returning all feature maps.


def relu(feature_map):
    #Preparing the output of the ReLU activation function.
    relu_out = np.zeros(feature_map.shape)
    for map_num in range(feature_map.shape[-1]):
        for r in np.arange(0,feature_map.shape[0]):
            for c in np.arange(0, feature_map.shape[1]):
                relu_out[r, c, map_num] = np.max([feature_map[r, c, map_num], 0])
    return relu_out

# def softmax_fn(input_array):
#     e_x=np.exp(input_array-np.max(input_array))
#     return e_x/e_x.sum(axis=len(e_x.shape)-1)


# X_test = X_test.reshape(1,IMG_SIZE,IMG_SIZE)

path = r'/home/atif/image_classification_c++/multi_filter_cpp/test_image/'

img_path = glob.glob(path+ '/*.ppm')
for image in img_path:
    print("hey i am image: ",image)
    X_test=[]
    X_test.append(preprocess_img(io.imread(image)))
    X_test = np.array(X_test)
#     plt.imshow(X_test)
    X_test = X_test.reshape(IMG_SIZE,IMG_SIZE)
    
    feature=conv(img=X_test,conv_filter=convolution_kernel_filter)
    relu_out=relu(feature)
    
    # output of feature map / conv function

    print(feature.shape)

    # x_feature_map=np.flipud(feature[0])
    transpose_feature_map=feature.transpose()
    print("transpose_feature_map shape: ",transpose_feature_map.shape)
#     plt.imshow(transpose_feature_map[0])
    print("transpose_feature_map: \n",transpose_feature_map)
    
    print("relu_out shape: ",relu_out.shape)
    relu_out_transpose=relu_out.transpose()
    print("relu_out_transpose shape: ",relu_out_transpose.shape)
#     plt.imshow(relu_out_transpose[0])
    print("relu_out_transpose:\n",relu_out_transpose)
    
    
    
    # matrix multiplication with dense kernel and relu o/p

    flatten_relu_out_transpose=relu_out_transpose.reshape(1,2*46*46)  #if you don't do padd on input image please make it 46*46. how 46 came? 
                                                                                    #the formula of output size.
    print("flatten_relu_out_transpose shape: ",flatten_relu_out_transpose.shape)

    print("dense_kernel shape: ",dense_kernel.shape,"\n")

    matmul_flatt_rel_dense_kernel=np.matmul(flatten_relu_out_transpose,dense_kernel)
    print("matmul_soft_dense_kernel shape",matmul_flatt_rel_dense_kernel.shape,"\n")
    print("matmul_soft_dense_kernel: ",matmul_flatt_rel_dense_kernel,"\n")

    dense_bias_array=np.array(dense_bias)
    dense_bias_array=dense_bias_array.reshape(1,9) # 9 for 9 class
    print("dense_bias_array: ",dense_bias_array,"\n")

    add_matmul_flatt_rel_dense_kernel_and_dense_bias_array=matmul_flatt_rel_dense_kernel+dense_bias_array
    print("value add_matmul_flatt_rel_dense_kernel_and_dense2_array:\n",add_matmul_flatt_rel_dense_kernel_and_dense_bias_array)
    
    def softmax_fn(input_array):
        e_x=np.exp(input_array-np.max(input_array))
        return e_x/e_x.sum(axis=len(e_x.shape)-1)

    op= softmax_fn(add_matmul_flatt_rel_dense_kernel_and_dense_bias_array)
    print("output of FC layer: ",op,"\n")
    ########################
    # Folowing code for finding class##
    ########################
    m=0
    k=0
    # op=[[0.17095664, 0.24349895, 0.172376,   0.19243606, 0.62073235]]
    # op=np.array(op)
    # print(op.shape)
    # print(type(op))

    for h in op:
    
        for index,j in enumerate(h):
    
            o=j
            #print(o)
            if o>m:
                m=o
                print(m)
                k=index
            else:
                pass
    print('class:',k)