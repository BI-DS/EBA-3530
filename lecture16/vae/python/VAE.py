from abc import ABC, abstractmethod
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super().__init__(self)
        self.encoder = encoder
        self.decoder = decoder
        
    def call(self, x):
        # call encoder first
        mu_z, logvar_z, std_z = self.encoder(x)

        # reparamterization trick!
        eps = tf.random.normal(mu_z.shape)
        z = mu_z + eps*std_z

        # call decoder now
        mu_x, logvar_x, sigma_x = self.decoder(z)

        # calculate KL
        kl    = self.encoder.objective_func(mu_z, logvar_z)

        # calculate log density
        logsigma_x = tf.math.log(sigma_x)
        log_prob = self.decoder.objective_func(x,mu_x,logsigma_x)
        
        # calculate ELBO and loss (-ELBO)
        elbo = log_prob - kl
        self.vae_loss = -tf.reduce_mean(elbo)

        return self.vae_loss

    @tf.function
    def train(self, x, optimizer):
        with tf.GradientTape() as tape:
            vae_loss = self.call(x)
        gradients = tape.gradient(self.vae_loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return vae_loss 
