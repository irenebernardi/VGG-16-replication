# Import packages
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import shutil
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from matplotlib import image

# Define constants

# Define number of classes to use (this is a way to subset the data)
CLASS_NUM = 3

BATCH_SIZE = 32

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
print(np.shape(image))

# Import train set as a Dataset object
# (this object type can be used as input to the model)
raw_train_set = tf.keras.utils.image_dataset_from_directory(subset_dir, image_size=(224, 224))

# Normalise images to [0-1] scale
train_set = raw_train_set.map(lambda x, y: (x - tf.reduce_mean(x), y))
# Separate images and labels
images, labels = next(iter(train_set))

# Prepare data for training
X_train = images
y_train = pd.get_dummies(labels)

# Set input shape (this is not working)
input_shape = X_train.shape[1:]
input_shape

plt.figure(figsize=(10,10))
for i in range(10):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i])
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(int(labels[i]))
plt.show()

model = tf.keras.models.Sequential()

model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3),padding= "same")) #conv3-64 layer 1
model.add(layers.MaxPooling2D((2, 2), strides = (2,2))) #maxpool 1

model.add(layers.Conv2D(128, (3, 3), activation='relu',padding= "same")) #conv3-128 layer 2
model.add(layers.MaxPooling2D((2, 2), strides = (2,2))) #maxpool 2

model.add(layers.Conv2D(256, (3, 3), activation='relu',padding= "same")) #conv3-256 layer 3
model.add(layers.Conv2D(256, (3, 3), activation='relu',padding= "same")) #conv3-256 again layer 4
model.add(layers.MaxPooling2D((2, 2), strides = (2,2))) #maxpool 3

model.add(layers.Conv2D(512, (3, 3), activation='relu',padding= "same")) #conv3-512 layer 5
model.add(layers.Conv2D(512, (3, 3), activation='relu',padding= "same")) #conv3-512 again layer 6
model.add(layers.MaxPooling2D((2, 2), strides = (2,2))) #maxpool 4

model.add(layers.Conv2D(512, (3, 3), activation='relu',padding= "same")) #conv3-512 - second round layer 7
model.add(layers.Conv2D(512, (3, 3), activation='relu',padding= "same")) #conv3-512 again layer 8
model.add(layers.MaxPooling2D((2, 2), strides = (2,2))) #maxpool 5 - final

model.add(layers.Flatten()) #necessary for 1D Dense layer
model.add(layers.Dense(4096, activation='relu')) # FC 1 layer 9
model.add(layers.Dropout(0.5))
model.add(layers.Dense(4096, activation='relu')) #FC 2 layer 10
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation='relu')) # FC 3 layer 11
model.add(layers.Softmax())

model.summary()

# Choose optimiser, loss function and validation metric
LR_Decay = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='loss',
    factor=0.1,
    patience=2,
    mode='auto',
    min_delta=0.0001,
    min_lr=0.00001
)

model.compile(optimizer = tf.keras.optimizers.SGD(
    learning_rate=0.01,
    momentum=0.90,
    nesterov=False,
    weight_decay=0.0005,
),  loss="categorical_crossentropy",
    metrics=["categorical_accuracy"]
                )


# Train the model
history = model.fit(
    X_train, y_train,
    epochs=2,
    batch_size=10,
    verbose=True,
    callbacks=[LR_Decay]
)

# Store training history as a dataframe
history_df = pd.DataFrame(history.history)

print(f"Train loss: {history_df['loss'].iloc[-1]:.3f}")
print(f"Train accuracy: {history_df['categorical_accuracy'].iloc[-1]:.3f}")

plt.plot(history.history['categorical_accuracy'], label='accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
# plt.legend(loc='lower right')

model_file = f"/kaggle/working/{CLASS_NUM}class_model.keras"

# Save model into file for replication purposes
model.save(model_file)