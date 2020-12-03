
# 1 VGG- CIFAR 10
### Goals
- construct a VGG model with TF-Keras
- train and evaluate model on Cifar10 dataset
- perform hyperparameter tuning
- plot accuracy and loss over time

# Convolutional Block
The convolutional block is one of the building blocks of the VGG architecture. The actual composition of the block can vary across tasks. In this lesson, we're going to implement the following stack:

    Block: conv - batch norm - conv - maxpool - dropout
    
However, you are encourage to try other combinations and experiment training different types of blocks. for example:
    
    Block: batch norm - conv - batch norm - conv - maxpool - dropout
    Block: conv - batch norm - conv - batch norm - maxpool
    Block: conv - dropout - conv - dropout - maxpool
    
# Fully Connect Block

Similarly for the dense block there are multiple ways to define them, here is one way:

    Block: dense - dropout

Other ways are:
    
    Block: dense - batch norm
    Block: dense - batch norm - dropout



# 2 Transfer Learning- CIFAR 10
# Using Transfer Learning to Train an Image Classification Model



There are two main Transfer Learning schemes:
- Pre-trained Convolutional layers as fixed feature extractor
- Fine-tuning on pre-trained Convolutional layers.


# Pre-trained Convolutional layers as fixed feature extractor

<img src="/images/transfer_learning_1.jpg" width="400">

This scheme treats the Convolutional layers as a fixed feature extractor for the new dataset. Convolutional layers have fixed weights and therefore are not trained. They are used to extract features and construct a rich vector embedding for every image. Once these embeddings have been computed for all images, they become the new inputs and can be used to train a linear classifier or a fully connected network for the new dataset.


# Fine-tuning on pre-trained Convolutional Layers

To further improve the performance of our image classifier, we can "fine-tune" a pre-trained VGG model alongside the top-level classifier. Fine-tuning consist in starting from a trained network, then re-training it on a new dataset using very small weight updates.

<img src="/images/transfer_learning_2.jpeg" width="900">


This consists of the following steps:

- Load pretrained weights from a model trained on another dataset
- Re-initialize the top fully-connected layers with fresh weights
- Train model on new dataset (freeze or not convolutional layers)

This scheme treats the Convolutional layers as part of the model and applies backpropagation through the model. This fine-tunes the weights of the pretrained network to the new task. It is also possible to keep some of the earlier layers fixed (due to overfitting concerns) and only fine-tune some higher-level portion of the network.
