# cluster-mil

--------------------------------------------------------------------------- 
### Description: 

**cluster-mil** is a weakly supervised learning technique which uses concepts from the Multiple 
Instance Learning (MIL) framework to train a convolutional deep neural network. 
As data is only available in batches, we pretrain a variational autoencoder (unsupervised) 
and then estimate class labels from weak labels provided during training.

This work was presented at ML4H @ NeurIPS 2018 and a full description of the method is
available on ArXiv: [https://arxiv.org/abs/1812.00884](Cluster-Based Learning from Weakly Labeled Bags in Digital Pathology)

#### MNIST-BAG

![Alt text](mnist-bag-sample.png?raw=true "MNIST-BAG")

The implementation of the MNIST-BAG experiment is given in mnist-bag.py and 
utilizes our novel loss function (loss.py). Pretrained autoencoder models are
also provided in /latent_models.



