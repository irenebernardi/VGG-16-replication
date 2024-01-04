# Very deep convolutional networks for large-scale image recognition 
In this project, we aimed to replicate the results of the model described by [Simonyan & Zisserman 2015](https://arxiv.org/abs/1409.1556v6) ("Very Deep Convolutional Networks for Large-Scale Image Recognition"), also known as VGG16, which uses CNN with varying depths for large-scale image recognition. The CNN configuration achieves great accuracy on the ImageNet dataset thanks to the use of sequential 3x3 convolutional filters. The best results in the paper were achieved with 16 and 19 layer models, but several were tested:  VGG11, VGG13, VGG16, and VGG19.

## Project progress 
Using TensorFlow parallel processing and Kaggle's built in TPU to maximize efficiency, our team managed to:
  - preprocess the images according to the paper's guidelines;
  - fully reproduce the architecture of the 11-layer model and transfer it to the subsequent deeper models for weights inheritance (via transfer learning). The architectures of VGG13, VGG16 and VGG19 were be replicated accordingly;
  - create utility scripts to maximize efficiency;
  - achieve 97% accuracy on a subset of the ImageNet dataset (roughly 150.000 images, batch size of 32, for 50 epochs). TODO: add final parameters we choose to use
  - TODO: insert val accuracy and compare to paper

# Shortcomings and possible future improvements  
  - We also built an analogous pipeline using generators, which would allow to process the entire ImageNet dataset on a GPU, but we could not achieve this on TPU due to compatibility issues. Future work could help fix this so as to process the whole dataset;
  - random weight initialization is required for 11-layer model, and we achieved it via `kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None)`. However, randomizing weights through this and other approaches consistently led to a nan loss. Future efforts should focus on debugging the underlying issue in order to replicate the original paper as closely as possible, and preserving random weights. 
  - TODO: possibly insert here what did not work in preprocessing;
  - 



# Final Files

Description of how to navigate our github and final files 


# Model API documentation 

For all architectures (11, 13, 16, 19 layers), the user can import and run our models and thus make new predictions with them. This API specification aims at making the importing process smooth and concise. 

### Overview 

Our work can be replicated thanks to our [utility scripts](https://github.com/irenebernardi/VGG-16-replication/blob/main/utilities-for-vgg.ipynb), which design the architecture for each model. The user can decide to import whichever architecture is most useful to them.

### Tutorial 

It is recommended to use your Kaggle account, as most of the GitHub links available here are Kaggle Notebooks. 
Please be aware that for the models to run smoothly, you should use a Tensor Processing Unit, as shown below. 
The import steps described below are also available at the LINNKALO example script.


Let's imagine you wish to use the 16-layer architecture: 

 1. On the right sidebar of your Kaggle notebook, add the [utility script](https://www.kaggle.com/code/giuliobenedetti/utilities-for-vgg/notebook) to your Kaggle input files. If you wish to use the same dataset as the one used in our code, [add it](https://www.kaggle.com/competitions/imagenet-object-localization-challenge) to your input files as well, or proceed with a different dataset. If you choose to do the latter, please be aware of the fact that our preprocessing pipeline may not be suited for a different dataset. 
 2. Select Kaggle's TPU by clicking on the accelerator available on the three dots at the top right corner of your screen.
 3. Setup a TPU as per [this cell](https://www.kaggle.com/code/giuliobenedetti/imagenet-reproducing-convnets?scriptVersionId=157234369&cellId=5).
 4. Setup constants, train and test size according to your needs. See our example here, TODO add link
 5. If using our preprocessing pipeline, simply call the utils snippet by using: `<your_ds> = <your_ds>.map(
    lambda path: vggutils.process_path(path, class_names),
    num_parallel_calls=tf.data.AUTOTUNE` 
)
 6. To visualize how the preprocessing affected your model, use: `for image, label in <your_ds>.take(1):
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())`
