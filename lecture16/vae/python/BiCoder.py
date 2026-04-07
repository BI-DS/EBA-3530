from abc import ABC, abstractmethod
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.models import Sequential
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def plot_grid(images,N=10,C=10,figsize=(24., 28.),name='posterior'):
    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(N, C),  
                     axes_pad=0,  # pad between Axes in inch.
                     )
    for ax, im in zip(grid, images):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('../output/'+name+'.png')
    plt.close()

class BiCoder(ABC):
    def __init__(self, **kwargs):
        self.network_type = kwargs['name']
        activation  = kwargs['activation']
        latent_dim  = kwargs['latent_dim']
        if self.network_type == 'encoder_mlp':
            units = kwargs['units']
            input_shape = kwargs['input_shape']
            self.network = Sequential(
                                    [ 
                                    layers.InputLayer(input_shape=input_shape),
                                    layers.Dense(units,activation=activation),
                                    layers.Dense(2*latent_dim),
                                    ]
                                    )
        elif self.network_type == 'encoder_conv':
            filters     = kwargs['filters'] 
            kernel_size = kwargs['kernel_size']
            strides     = kwargs['strides']
            input_shape = kwargs['input_shape']
            self.network = Sequential(
                                    [
                                    layers.InputLayer(input_shape=input_shape),
                                    layers.Conv2D(
                                      filters=filters,   kernel_size=kernel_size, strides=strides, activation=activation, padding='same'),
                                    layers.Conv2D(
                                      filters=2*filters, kernel_size=kernel_size, strides=strides, activation=activation, padding='same'),
                                    layers.Conv2D(
                                      filters=4*filters, kernel_size=kernel_size, strides=strides, activation=activation, padding='same'),
                                    layers.Flatten(),
                                    layers.Dense(2*latent_dim)
                                    ]
                                    )
        elif self.network_type == 'decoder_mlp':
            units = kwargs['units']
            output_dim  = kwargs['output_dim'] 
            self.network = Sequential(
                                    [
                                    layers.InputLayer(input_shape=latent_dim),
                                    layers.Dense(units=units,activation=activation),
                                    layers.Dense(output_dim),
                                    ]
                                    )
        elif self.network_type == 'decoder_conv':
            target_shape=(4,4,128)
            channel_out=3
            units = np.prod(target_shape)
            filters     = kwargs['filters'] 
            kernel_size = kwargs['kernel_size']
            strides     = kwargs['strides']
            self.network = Sequential(
                                    [
                                    layers.InputLayer(input_shape=(latent_dim,)),
                                    layers.Dense(units=units, activation=activation),
                                    layers.Reshape(target_shape=target_shape),
                                    layers.Conv2DTranspose(
                                        filters=filters*2, kernel_size=kernel_size, strides=strides, padding='same',output_padding=0,
                                        activation=activation),
                                    layers.Conv2DTranspose(
                                        filters=filters, kernel_size=kernel_size, strides=strides, padding='same',output_padding=1,
                                        activation=activation),
                                    layers.Conv2DTranspose(
                                        filters=channel_out, kernel_size=kernel_size, strides=strides, padding='same', output_padding=1),
                                    layers.Activation('linear', dtype='float32'),
                                    ]
                                    )


    @abstractmethod
    def objective_func(self):
        pass

    def call(self, inputs):
        out = self.network(inputs)
        if out.shape[1] == (2*self.latent_dim):
            mu  = out[:,:self.latent_dim]
            log_var = out[:,self.latent_dim:]
            std = tf.math.exp(0.5*log_var)
        else:
            mu = out
            std = self.std
            log_var = tf.math.log(std**2)
        return mu, log_var, std
    
    def generate(self, inputs, point_estimate_mu=True):
        mu, log_var, std = self.call(inputs)
        if point_estimate_mu:
            return mu
        else:
            eps = tf.random.normal(mu.shape)
            return mu + eps*std

    # XXX using polymorphisim for downstream taks!
    # Bicoder doesnt know how self.output_network
    # is calculated and it doesnt care! 
    def downstream_tasks(self, plot_name, y_te=None):
        try:
            size_output = self.network_output.get_shape().as_list()
            # check if output are images (B,H,W,C)
            if len(size_output) > 2:
                print('plotting generated images...')
                x_hat = self.network_output
                if x_hat.shape[0] > 100:
                    x_hat = x_hat[0:100]
                img = tf.clip_by_value(255*x_hat, clip_value_min=0, clip_value_max=255).numpy().astype(np.uint8)
                plot_grid(img,name=plot_name)
            # check if output are vectors (B,dim)
            elif len(size_output)==2:
                z = self.network_output
                print('fitting tsne...')
                tsne_components = TSNE(n_components=2,n_jobs=-1).fit_transform(z.numpy())
                if y_te is not None:
                    plt.scatter(tsne_components[:,0],tsne_components[:,1],s=4, c=y_te)
                else:
                    plt.scatter(tsne_components[:,0],tsne_components[:,1],s=4)
                plt.savefig('../output/'+plot_name+'.png')
                plt.close()
        except:
            print('Sampling from prior/posterior or generate method must be called before!')

