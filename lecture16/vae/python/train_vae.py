import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from Encoder import Encoder
from Decoder import Decoder
from VAE import VAE
from DataLoader import DataLoader
from datetime import timedelta
import time
import numpy as np
import argparse

def mnist_bw():
    input_shape = (28*28,)
    units = 400
    activation  = 'relu'
    latent_dim = 20
    output_dim  = 28*28
    name_enc = 'encoder_mlp'
    name_dec = 'decoder_mlp'

    # models
    enc = Encoder(name=name_enc, activation=activation, latent_dim=latent_dim, units=units, input_shape=input_shape)
    dec = Decoder(name=name_dec, activation=activation, latent_dim=latent_dim, units=units, output_dim=output_dim)
    vae = VAE(enc, dec)

    # data
    data_manager = DataLoader(dset='mnist_bw')
    data_manager.load_data() # loads all files
    tr_data = data_manager.tf_loader(train=True, batch_size=256)

    ################################
    #
    # training
    #
    epochs = 50 
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    print('traning for {} epochs...'.format(epochs))
    start = time.time()
    for e in range(epochs):
        for i, x_batch in enumerate(tr_data):
            loss = vae.train(x_batch,optimizer)
        if (e+1)%10==0:
            print('epoch {} with loss {:.4f}'.format(e+1,loss))

    ################################
    #
    # downstream tasks  
    #
    x_te, y_te = data_manager.tf_loader(train=False)
    z = vae.encoder.sampling_posterior(x_te)
    vae.encoder.downstream_tasks(plot_name='bw_latent_space',y_te=y_te)                        

    x_hat = vae.decoder.generate(z, point_estimate_mu=True, one_channel=True)
    img = tf.clip_by_value(255*x_hat, clip_value_min=0, clip_value_max=255).numpy().astype(np.uint8)
    vae.decoder.downstream_tasks(plot_name='bw_imgs_posterior')                        

    z = vae.encoder.sampling_prior()
    x_hat = vae.decoder.generate(z, point_estimate_mu=True, one_channel=True)
    vae.decoder.downstream_tasks(plot_name='bw_imgs_prior')                        
    print('elapsed time: {}'.format(timedelta(seconds=time.time()-start)))

def mnist_color(modality = 'm0'):
    input_shape = (28,28,3)
    filters     = 32
    kernel_size = 3
    strides     = 2
    activation  = 'relu'
    latent_dim  = 50
    target_shape=(4,4,128)
    channel_out=3
    units = np.prod(target_shape)
    name_enc = 'encoder_conv'
    name_dec = 'decoder_conv'

    # models
    enc = Encoder(name=name_enc, activation=activation, latent_dim=latent_dim, filters=filters, kernel_size=kernel_size, strides=strides, input_shape=input_shape)
    dec = Decoder(name=name_dec, activation=activation, latent_dim=latent_dim, filters=filters, kernel_size=kernel_size, strides=strides, target_shape=target_shape)
    vae = VAE(enc, dec)

    # data
    data_manager = DataLoader(dset='mnist_color')
    data_manager.load_data(modality=modality) # loads all files
    tr_data = data_manager.tf_loader(train=True, batch_size=256)

    ################################
    #
    # training
    #
    epochs = 50
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    print('traning for {} epochs...'.format(epochs))
    start = time.time()
    for e in range(epochs):
        for i, x_batch in enumerate(tr_data):
            loss = vae.train(x_batch,optimizer)
        if (e+1)%10==0:
            print('epoch {} with loss {:.4f}'.format(e+1,loss))
   
    ################################
    #
    # downstream tasks  
    #
    x_te, y_te = data_manager.tf_loader(train=False)
    z = vae.encoder.sampling_posterior(x_te)
    vae.encoder.downstream_tasks(plot_name='color_latent_space',y_te=y_te)                        

    x_hat = vae.decoder.generate(z, point_estimate_mu=True, one_channel=False)
    img = tf.clip_by_value(255*x_hat, clip_value_min=0, clip_value_max=255).numpy().astype(np.uint8)
    vae.decoder.downstream_tasks(plot_name='color_imgs_posterior')                        

    z = vae.encoder.sampling_prior()
    x_hat = vae.decoder.generate(z, point_estimate_mu=True, one_channel=False)
    vae.decoder.downstream_tasks(plot_name='color_imgs_prior')                        
    
    print('elapsed time: {}'.format(timedelta(seconds=time.time()-start)))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dset", default= 'color', help="dataset to use. choose between color and bw (black&white)")
    parser.add_argument("--modality", default= 'm0', help="version of color images. choose between m0-m4")
    args = parser.parse_args()
    print(args)
    
    if args.dset == 'color':
        mnist_color(args.modality)
    elif args.dset == 'bw':
        mnist_bw()
    else:
        print('wrong dset')
