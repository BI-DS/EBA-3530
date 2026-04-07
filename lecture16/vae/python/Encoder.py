import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import layers
from BiCoder import BiCoder

class Encoder(BiCoder, layers.Layer):
    def __init__(self, **kwargs):
        layers.Layer.__init__(self)
        BiCoder.__init__(self, **kwargs)
        self.latent_dim  = kwargs['latent_dim']
        self.enc_name = kwargs['name']

    def sampling_prior(self, N=100):
        z = tf.random.normal((N,self.latent_dim))
        
        # XXX saving self.network_output for downstream task
        self.network_output = z
        
        return z 

    # XXX adding this method to have both
    # sampling prior and posterior
    def sampling_posterior(self, inputs):
        z = self.generate(inputs,point_estimate_mu=False)

        # XXX saving self.network_output for downstream task
        self.network_output = z
        
        return z

    def objective_func(self, mu, log_var):
        return 0.5 * tf.reduce_sum(tf.square(mu) + tf.exp(log_var) - log_var - 1, axis=-1)
