# Very deep convolutional networks for large-scale image recognition 
In this project, we aimed to replicate the results of the model described by [Simonyan & Zisserman 2015](https://arxiv.org/abs/1409.1556v6) ("Very Deep Convolutional Networks for Large-Scale Image Recognition"), also known as VGG16, which uses CNN with varying depths for large-scale image recognition. The CNN configuration achieves great accuracy on the ImageNet dataset thanks to the use of sequential 3x3 convolutional filters. The best results in the paper were achieved with 16 and 19 layer models, but several were tested:  VGG11, VGG13, VGG16, and VGG19.

## Project Milestones and progress 
Using TensorFlow parallel processing and Kaggle's built in TPU to maximize efficiency, our team managed to:
  - preprocess the images according to the paper's guidelines;
  - fully reproduce the architecture of the 11-layer model (TODO: insert other models we fully replicate) and transfer it to the subsequent deeper models for weights inheritance (via transfer learning);
  - achieve 97% accuracy on a subset of the ImageNet dataset (roughly 150.000 images, batch size of 32, for 50 epochs). TODO: add final parameters we choose to use
  - TODO: insert val accuracy and compare to paper

# Shortcomings and possible future improvements  
  - We also built an analogous pipeline using generators, which would allow to process the entire ImageNet dataset on a GPU, but we could not achieve this on TPU due to compatibility issues. Future work could help fix this so as to process the whole dataset;
  - TODO: possibly insert here what did not work in preprocessing;
  - 



# Final Files

Description of how to navigate our github and final files 
