import numpy as np
from skimage import io, color, exposure, transform
from skimage.color import rgb2gray
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split  #it came from update scikit learn. https://stackoverflow.com/questions/40704484/importerror-no-module-named-model-selection
import os, json
import glob
import h5py
from matplotlib import pyplot as plt

with open('variable_config.json', 'r') as f:
    config = json.load(f)


NUM_CLASSES = config['DEFAULT']['num_class']
IMG_SIZE = config['DEFAULT']['img_size']
number_filter = config['DEFAULT']['number_filter']
filter_size = config['DEFAULT']['filter_size']
img_depth = config['DEFAULT']['img_depth']
img_type = config['DEFAULT']['img_type']
epochs = config['DEFAULT']['epochs']
batch_size = config['DEFAULT']['batch_size']
train_image_path = config['DEFAULT']['train_image_path']
test_image_path = config['DEFAULT']['test_image_path']
learning_model_path = config['DEFAULT']['learning_model_path']
validation_split = config['DEFAULT']['validation_split']
'''
NUM_CLASSES = 9  #how many different type of class/type of image you are using. Like CAT,DOG, ELEPHANT etc
IMG_SIZE = 48  # You can change it. Always keep same size for width and height. That means square size.
'''

from keras.models import load_model
model = load_model(learning_model_path+'learning_model.h5')
#for gray scale
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

import glob

path = test_image_path
#path = r'/home/atif/training_by_several_learning_process/number_classify/rgb_2_gray/Image-classification/test_image/'

img_path = glob.glob(path+ '/*'+str(img_type))
for image in img_path:
    X_test=[]
    X_test.append(preprocess_img(io.imread(image)))
    X_test = np.array(X_test)
#     plt.imshow(X_test)
    X_test = X_test.reshape(len(X_test),img_depth,IMG_SIZE,IMG_SIZE)
    
    print("\n",image)
    predicted_class = model.predict_classes(X_test)
    print("predicted class: ",predicted_class)
    
    probability = model.predict_proba(X_test)
    print("probability: ",probability)
