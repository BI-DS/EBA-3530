import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from BiCoder import BiCoder

class Decoder(BiCoder, layers.Layer):
    def __init__(self, **kwargs):
        layers.Layer.__init__(self)
        BiCoder.__init__(self,**kwargs)
        self.latent_dim  = kwargs['latent_dim']
        self.dec_name = kwargs['name']
        self.std = 0.75
   
    # Overriding super class to further implement the generative 
    # model. Hence, it generates and reconstructs imgages
    def generate(self, z, N=100, point_estimate_mu=True, one_channel=False):
        x_hat = BiCoder.generate(self, z, point_estimate_mu=point_estimate_mu)
        if one_channel:
            x_hat = tf.reshape(x_hat,(-1,28,28,1)) 
        
        # XXX saving network_output for downstream task
        self.network_output = x_hat
        return x_hat

    def objective_func(self, x, mu, log_sigma):
        sum_axes = tf.range(1, tf.rank(mu))
        k = tf.cast(tf.reduce_prod(tf.shape(mu)[1:]), x.dtype)
        obj = - 0.5 * k * tf.math.log(2*np.pi) \
              - log_sigma \
              - 0.5*tf.reduce_sum(tf.square(x - mu)/tf.math.exp(2.*log_sigma),axis=sum_axes)
        return obj

