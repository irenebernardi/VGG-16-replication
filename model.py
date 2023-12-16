# Import packages
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

# Define constants

# Set number of classes to use (this is a way to subset the data)
CLASS_NUM = 3
# Set batch size for mini-batch gradient descent
BATCH_SIZE = 32
#BATCH_SIZE = 256

# Set kernel size for Conv2D layers
KERNEL_SIZE = 3
# Set padding mode for Conv2D layers
PAD_MODE = "same"
# Set activation function for Conv2D layers
ACTIVATION = "relu"

# Set pool size for MaxPool2D layers
POOL_SIZE = 2
# Set strides for MaxPool2D layers
POOL_STRIDES = 2

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
image = image.imread(subset_dir + "/n02098413/n02098413_720.JPEG")
# Show image
plt.imshow(image)

# Import train set as a Dataset object
# (this object type can be used as input to the model)
raw_train_set = keras.utils.image_dataset_from_directory(
    subset_dir,
    image_size=(224, 224),
    batch_size=BATCH_SIZE
)

# Normalise images to 0-centred distribution
train_set = raw_train_set.map(lambda x, y: (x - tf.reduce_mean(x), y))
# Separate images and labels
images, labels = next(iter(train_set))

# Prepare data for training
X_train = images
y_train = pd.get_dummies(labels)

# Check input shape
input_shape = X_train.shape[1:]
print(f"Input shape: {input_shape}")

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
X_train =tf.constant(cropped_images_s256)
print(X_train.shape) # used to be image_ds
y_train = pd.get_dummies(labels)
#get np array
y_train = y_train.values


# Design model
model = keras.models.Sequential([
    
    # 1st convolutional block
    layers.Conv2D(input_shape=input_shape, filters=64, kernel_size=KERNEL_SIZE, padding=PAD_MODE, activation=ACTIVATION),
    layers.MaxPooling2D(pool_size=POOL_SIZE, strides=POOL_STRIDES),
    
    # 2nd convolutional block
    layers.Conv2D(filters=128, kernel_size=KERNEL_SIZE, padding=PAD_MODE, activation=ACTIVATION),
    layers.MaxPooling2D(pool_size=POOL_SIZE, strides=POOL_STRIDES),
    
    # 3rd convolutional block
    layers.Conv2D(filters=256, kernel_size=KERNEL_SIZE, padding=PAD_MODE, activation=ACTIVATION),
    layers.Conv2D(filters=256, kernel_size=KERNEL_SIZE, padding=PAD_MODE, activation=ACTIVATION),
    layers.MaxPool2D(pool_size=POOL_SIZE, strides=POOL_STRIDES),
    
    # 4th convolutional block
    layers.Conv2D(filters=512, kernel_size=KERNEL_SIZE, padding=PAD_MODE, activation=ACTIVATION),
    layers.Conv2D(filters=512, kernel_size=KERNEL_SIZE, padding=PAD_MODE, activation=ACTIVATION),
    layers.MaxPool2D(pool_size=POOL_SIZE, strides=POOL_STRIDES),
    
    # 5th convolutional block
    layers.Conv2D(filters=512, kernel_size=KERNEL_SIZE, padding=PAD_MODE, activation=ACTIVATION),
    layers.Conv2D(filters=512, kernel_size=KERNEL_SIZE, padding=PAD_MODE, activation=ACTIVATION),
    layers.MaxPool2D(pool_size=POOL_SIZE, strides=POOL_STRIDES),
    
    # Classifier head
    layers.Flatten(),
    layers.Dense(4096, activation=ACTIVATION),
    layers.Dropout(rate=0.5),
    layers.Dense(4096, activation=ACTIVATION),
    layers.Dropout(rate=0.5),
    layers.Dense(CLASS_NUM, activation="softmax")
])


# Choose optimiser, loss function and validation metric
model.compile(
    optimizer=keras.optimizers.experimental.SGD(momentum=0.9, weight_decay=0.0005),
    loss="categorical_crossentropy",
    metrics=["categorical_accuracy"]
)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=2,
    batch_size=BATCH_SIZE,
    verbose=True
)

# Store training history as a dataframe
history_df = pd.DataFrame(history.history)

print(f"Train loss: {history_df['loss'].iloc[-1]:.3f}")
print(f"Train accuracy: {history_df['categorical_accuracy'].iloc[-1]:.3f}")

# Visualise loss
history_df.loc[:, "loss"].plot(title="Loss")

# Visualise accuracy
history_df.loc[:, "categorical_accuracy"].plot(title="Accuracy")

# Define model file
model_file = f"/kaggle/working/{CLASS_NUM}class_model.keras"

# Save model into file for replication purposes
model.save(model_file)
