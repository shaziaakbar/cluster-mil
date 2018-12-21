'''
Author: Shazia Akbar

Description: Customized loss function which appends an estimated class label computed in 
generator and passed to loss function with weak label.
Weak and estimated label are weighted by alpha.

'''

import tensorflow as tf
from keras.backend import categorical_crossentropy


def get_cce(p, y):
    return categorical_crossentropy(y, p)


class LossFunction():
    def __init__(self, type, alpha, classes):
        self.type = type
        self.a = tf.constant(alpha, dtype='float32')
        self.num_classes = classes

    def loss_main(self, params):
        
        # gather custom parameters from generator
        y_true, y_pred, e_true = params

        # compute custom loss
        loss = tf.scalar_mul(self.a, get_cce(y_pred, y_true)) + \
               tf.scalar_mul(tf.subtract(tf.constant(1, 'float32'), self.a), get_cce(y_pred, e_true))

        return loss