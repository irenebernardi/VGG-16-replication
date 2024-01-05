# Very deep convolutional networks for large-scale image recognition 
In this project, we aimed to replicate the results of the model described by [Simonyan & Zisserman 2015](https://arxiv.org/abs/1409.1556v6) ("Very Deep Convolutional Networks for Large-Scale Image Recognition"), also known as VGG16, which uses CNN with varying depths for large-scale image recognition. The CNN configuration achieves great accuracy on the ImageNet dataset thanks to the use of sequential 3x3 convolutional filters. The best results in the paper were achieved with 16 and 19 layer models, but several were tested:  VGG11, VGG13, VGG16, and VGG19.

## Project milestones
Overall, we consider that our models mirror the paper quite well. Using TensorFlow parallel processing and Kaggle's built in TPU to maximize efficiency, our team managed to:
  - preprocess the images according to the paper's guidelines. Namely, mean RGB value was subtracted from all images, which were rescaled by an S factor (S=256) and randomly cropped into a 228x228 square. Data augmentation was obtained thanks to image flipping/changes in color contrast;
  - fully reproduce the architecture of the 11-layer model and transfer it to the subsequent deeper models for weights inheritance (via transfer learning). The architectures of VGG13, VGG16 and VGG19 were set up accordingly;
  - improve latency, troughput and overall performance via caching, shuffling and prefetching of data;
  - use utility scripts to maximize efficiency and keep the code compact;
  - achieve training accuracy of nearly 100%.

# Shortcomings and possible future improvements  
  - Due to time and computational constraints, we were only able to use a train set of size 170000 (11% of total train set), and for only 15 epochs as opposed to . CONTINUA
  - We have built a pipeline analogous to the current one using generators, which would allow to process the entire ImageNet dataset on a GPU, but we could not achieve this on TPU due to compatibility issues. Future work could help fix this so as to process the whole dataset;
  - The S factor in the paper takes two values: 256 and NUMERO. Only the former was utilized because of computational constraints.
  - random weight initialization from a normal distribution is required by the paper for the 11-layer model, and we achieved it thanks to: `kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None)`. However, randomizing weights through this and other approaches consistently led to a nan loss. Thus, we resorted to a weight initialization that draws weights from a Glorot uniform distribution, to avoid the vanishing gradient problem. However, this does not exactly replicate the paper's procedure. Future efforts should focus on debugging the underlying issue in order to replicate the original paper as closely as possible, and preserving random weights. 
  - Unfortunately, whereas training accuracy was extremely high (nearing 100%), accuracy values for the validation set (~50%) show that our models severely overfit. We interpret this difference in performance between the two sets in two possible ways:
    - the model architectures may be too complex for the subset of data we are using;
    - the small amount of data we are using may hint that the two sets are inadvertently biased or         imbalanced; 
    - the validation set was preprocessed improperly with respect to the training set.
    The exact cause can be investigated by using more data and careful debugging of how our               preprocessing affects each set.



# How to navigate the repository

TBD: FINAL FILES ORGANIZATION 


# Project replicability 

For all architectures (11, 13, 16, 19 layers), users can import and run our models and thus make new predictions with them. This section aims at making the replication process smooth and concise. 

### Overview 

Our work can be replicated thanks to our [utility scripts](https://github.com/irenebernardi/VGG-16-replication/blob/main/utilities-for-vgg.ipynb), which design the architecture for each model. The user can decide to import whichever architecture is most useful to them.

### Tutorial 

It is recommended to use a Kaggle account, as most of the GitHub links available here are Kaggle Notebooks. 
Please be aware that for the models to run smoothly, you should use a Tensor Processing Unit, as shown below. 


Let's imagine you wish to use the 16-layer architecture: 

 1. On the right sidebar of your Kaggle notebook, add the [utility script](https://www.kaggle.com/code/giuliobenedetti/utilities-for-vgg/notebook) to your Kaggle input files. If you wish to use the same dataset as the one used in our code, [add it](https://www.kaggle.com/competitions/imagenet-object-localization-challenge) to your input files as well, or proceed with a different dataset. If you choose to do the latter, please be aware of the fact that our preprocessing pipeline may not be suited for a different dataset. 
 2. Select Kaggle's TPU by clicking on the accelerator available on the three dots at the top right corner of your screen.
 3. Setup a TPU as per [this cell](https://www.kaggle.com/code/giuliobenedetti/imagenet-reproducing-convnets?scriptVersionId=157234369&cellId=5).
 4. Setup constants, train and test size according to your needs. If using our preprocessing pipeline, simply call the utils snippet by using: `<your_ds> = <your_ds>.map(
    lambda path: vggutils.process_path(path, class_names),
    num_parallel_calls=tf.data.AUTOTUNE)`
6. To use one of our models for your own projects, you can use `model.save()` to save your desired model in a keras file and `load_model()` (import statement: `from tensorflow.keras.models import load_model`.

