# CNN-imager
A project focused on improving CNN accuracy for image classification from unlabeled data


## Abstract

This project focuses on semi-supervised image classification using the CIFAR-10 dataset. With only a subset of the images having known labels, the objective is to leverage unlabeled examples to improve classification accuracy. The approach involves using clustering and iterative training to infer labels on unlabeled data, then investigating whether this improves the accuracy of a CNN model trained from scratch.

To achieve this, the project first experiments with dimensionality reduction using PCA and K-means clustering to estimate class labels on unlabeled data. The project compares clustering with PCA to clustering using embeddings from pretrained models, and uses the most effective approach in the CNN.

In the second stage, the project investigates whether inferred labels can improve the classification accuracy of a CNN model trained from scratch. The project uses three experimental models to compare the effects of adding inferred labels to the training dataset. Model A trains the CNN only on data with known class labels, while Model B trains the CNN on data with known class labels and labels inferred using KMeans clustering on embeddings. Finally, Model C trains the CNN on data with known class labels and labels inferred using Model A.

## Team Members and Contributions

### Manveer Singh Tamber (mtamber@uwaterloo.ca)
- Devised experiments for investigating the importance of using inferred labels for training CNNs to make classifications.
- Wrote model training and evaluation code within the three experimental settings
- Generated and compared model results across different M values
### Jagvir Dehal (jdehal@uwaterloo.ca)
- Worked on normalizing and reducing the images using PCA, and clustering these using K-means.
- Alongside clustering, created a function that generates a mapping to the real labels using ground truth labels we have
- Investigated the use of different pre-trained models for embeddings vs. the PCA reduction

Overall, the contributions through discussion and cross-checking code was 50/50.

## Code Libraries
### Pytorch

PyTorch (torch/torchvision) was used to define, train, and evaluate the CNN models torchsummary was used to output the details of the CNN model

Install command: `pip3 install torch torchvision torchsummary`

### Keras

Keras was used for the CIFAR-10 dataset, along with running the pretrained models (VGG16, VGG19, EfficientNetV2B2)

Install command: `pip3 install tensorflow`

### sklearn (site)

sklearn was used for the PCA and KMeans functions

Install command: `pip install -U scikit-learn`
