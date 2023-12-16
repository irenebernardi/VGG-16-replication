import numpy as np
import pandas as pd
import tensorflow as tf
import os
import shutil
import tensorflow 
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from matplotlib import image
import matplotlib.pyplot as plt
import skimage
from skimage.io import imread
from skimage.transform import resize
import cv2
import os

# Define number of classes to use (this is a way to subset the data)
CLASS_NUM = 3

BATCH_SIZE = 32

#data preprocessing 
# Store paths to base, train set and subset dirs
base_dir = "/kaggle/input/imagenet-object-localization-challenge"
train_dir = base_dir + "/ILSVRC/Data/CLS-LOC/train"
subset_dir = "/kaggle/working/data"

if not os.path.exists(subset_dir):
    os.mkdir(subset_dir)
    print(subset_dir, "created!")
else:
    print(subset_dir, "already exists!")
    
# Select only first n class dirs
class_dirs = os.listdir(train_dir)[:CLASS_NUM]

# Copy class dir from train set to working dir
for class_dir in class_dirs:
    # Define current source and destination paths
    source_dir = train_dir + "/" + class_dir
    destination_dir = subset_dir + "/" + class_dir
    
    # If new class, copy to working dir
    if not os.path.exists(destination_dir):
        shutil.copytree(source_dir, destination_dir)
        print(class_dir, "succesfully copied!")
    # If it exists, don't copy again
    else:
        print(class_dir, "doesn't need copying!")


# Read in image
image = imread(subset_dir + "/n02098413/n02098413_720.JPEG")
# Show image
plt.imshow(image)
plt.show()


#####RESIZE#####

#NOTE: need to change rescaling 

image_files = [f for f in os.listdir(subset_dir) if f.endswith(".JPEG")]

# Create a list to store resized images
resized_images = []

for image_file in image_files:
    #consutrct full image path
    image_path = os.path.join(subset_dir, image_file)

    # Read in the image
    image = imread(image_path)

    #resize the image
    resized_image = skimage.transform.resize(image, (256, 256, 3))

  
    resized_images.append(resized_image)



#####CROP#####
    
def crop(image, start_height, start_width, crop_height, crop_width): 
    #original image shape
    original_shape = image.shape
    #print(f"Original image shape: {original_shape}") just to check rescaling correct
    
    #Make sure you're cropping to a size that's not higher than the original picture
    assert start_height + crop_height <= original_shape[0], "Invalid crop height"
    assert start_width + crop_width <= original_shape[1], "Invalid crop width"
    
    #crop
    cropped_image = image[start_height:start_height + crop_height, start_width:start_width + crop_width, :]
    
    return cropped_image

directory_path = subset_dir

#list to store cropped images
cropped_images_s256 = []
labels = []  #Need to preserve original class labels

#Loop through each class directory
for class_dir in os.listdir(directory_path):
    class_dir_path = os.path.join(directory_path, class_dir)

    #Loop through each image file in the class directory
    for image_file in os.listdir(class_dir_path):
        image_path = os.path.join(class_dir_path, image_file)
        
       
        image = imread(image_path)
        
        #if want to resize and crop in one loop, do here 
        #resized_image = skimage.transform.resize(image, (256, 256, 3))
        
        if resized_image is not None:
            # TODO: Do start values need to be randomized?
            start_height = 0  #from the top
            start_width = 0   #from the left
            crop_height = 224
            crop_width = 224

            labels.append(class_dir)
            
            # Crop the image
            cropped_image = crop(resized_image, start_height, start_width, crop_height, crop_width)
            cropped_images_s256.append(cropped_image)

        
#sanity check
print(cropped_images_s256[0].shape)
print(cropped_images_s256[231].shape)

#convert images to tensors and one-hot encode labels before compiling model
images_ds =tf.constant(cropped_images_s256)
print(images_ds.shape)
labels_one_hot = pd.get_dummies(labels)
#get np array
labels_one_hot = labels_one_hot.values


#build model 

#create convolutional base
#refer to table 1 for architecture 
model = tf.keras.Sequential(name='CNN_11_layer_trial')

#useful comments for understanding layout, once we're all aligned delete: 

#Spatial pooling is carried out by five max-pooling layers, which follow some of the conv. layers (not all the conv. layers are followed
#by max-pooling). 
#Max-pooling is performed over a 2 × 2 pixel window, with stride 2.
#for conv layers: increasing filters by *2 for each conv layer, starting at 64 until 512
#input shape mandatory in first layer: image shape and 3, stands for RGB


model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=(224, 224, 3))) #1
model.add(layers.MaxPooling2D((2, 2), strides = 2))
model.add(layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu')) #2
model.add(layers.MaxPooling2D((2, 2), strides = 2)) 
model.add(layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')) #3
model.add(layers.Conv2D(256, (3, 3),  strides=(1, 1), padding='same', activation='relu')) #4
model.add(layers.MaxPooling2D((2, 2), strides = 2))
model.add(layers.Conv2D(512, (3, 3),  strides=(1, 1), padding='same', activation='relu')) #5
model.add(layers.Conv2D(512, (3, 3),  strides=(1, 1), padding='same', activation='relu')) #6
model.add(layers.MaxPooling2D((2, 2), strides = 2)) #final maxpool before fully connected layers 
model.add(layers.Conv2D(512, (3, 3),  strides=(1, 1), padding='same', activation='relu')) #7
model.add(layers.Conv2D(512, (3, 3),  strides=(1, 1), padding='same', activation='relu')) #8
#5th and last maxpool layer: 
model.add(layers.MaxPooling2D((2, 2), strides = 2))
#9,10,11 are fully conntected layers
#9
model.add(layers.Flatten()) #dense layers expect flat vectors, not tensors
model.add(layers.Dense(4096)) #4096 units
layers.Dropout(rate = 0.5)
#10
model.add(layers.Dense(4096)) #4096 units
layers.Dropout(rate = 0.5)
#11
#layers.Dense (3, activation = 'softmax')
model.add(layers.Dense(3)) #1000 units, 1000- way ILSVRC classification; here 3 for subset
#softmax before output
model.add(tf.keras.layers.Softmax(axis=-1))   #apply softmax to last dimension of input data

model.summary()

#train model 

from keras.optimizers import SGD

# Define the optimizer separately so we have better hyperparams control 
sgd = SGD(lr=0.01, momentum=0.9, weight_decay = 0.0005) #TODO: change learning rate and CHECK WEIGHT DECAU

#in 11 layer, paper initializes weights randomly
#this is redundant as this is the default initialisation method, keeping just for reference in later models w/ > layers 
weights = [np.random.rand(*w.shape) for w in model.get_weights()]
model.set_weights(weights)

# Compile the model
#model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.compile(
    optimizer=sgd,  # Optimizer
    # Loss function to minimize
    loss='categorical_crossentropy', #multinomial logistic regression
    # List of metrics to monitor
    metrics=['categorical_accuracy']
)


print("Fit model on training data")
history = model.fit(
    images_ds, 
    labels_one_hot, #pd.get_dummies(labels)
    batch_size= 2, 
    #validation_split=0.15, #we might not need to do this? as this is just a test 
    epochs= 2 ) #should be 74


#The returned history object holds a record of the loss values and metric values during training:
'''history_df = pd.DataFrame(history.history)

history_df.loc[:, "loss"].plot()
'''
plt.plot(history.history['loss'])

