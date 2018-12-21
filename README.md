# cluster-mil

--------------------------------------------------------------------------- 
### Description: 

Multiple instance learning framework designed around a convolutional deep neural network. 
As data is only available in batches, we pretrain a variational autoencoder (unsupervised) 
and then estimated class labels from weak labels provided during training.

A full description of this work has been submitted to ICML 2019.

#### MNIST-BAG

![Alt text](mnist-bag-sample.png?raw=true "MNIST-BAG")

The implementation of the MNIST-BAG experiment is given in mnist-bag.py and 
utilizes our novel loss function (loss.py). Pretrained autoencoder models are
also provided in /latent_models.



