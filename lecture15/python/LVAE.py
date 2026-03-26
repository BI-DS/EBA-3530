import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from MMNISTDataset import MMNISTDataset
import random
import argparse
from utils import compute_ppca_mle, log_p_x_true, plot_grid
from tensorflow.keras import layers
from datetime import timedelta
import time

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions
_LOG_2PI = np.log(2 * np.pi)

class LEnc(layers.Layer):
    def __init__(self,input_dim=28*28,z_dim=200,distribution=tfd.MultivariateNormalDiag):
        super().__init__()
        # std of q(z|x)
        # XXX D-dimensional vector 
        self.log_var_qz_x = tf.Variable(initial_value=np.repeat(1.,z_dim).astype(np.float32), trainable=True)
        # V matrix in mean of q(z|x)
        self.V = tf.Variable(initial_value=np.random.normal(loc=0,scale=0.01,size=(input_dim, z_dim)).astype(np.float32), trainable=True)
        self.distribution = distribution

    @property
    def std(self):
        std = tf.math.exp(0.5*self.log_var_qz_x)
        return std

    def call(self, inputs, mu):
        # mean of q(z|x)
        mean = tf.linalg.matmul(inputs - mu, self.V)
        std = tf.math.exp(0.5*self.log_var_qz_x)
        
        qz_x = self.distribution(mean, std)

        return qz_x

class LDec(layers.Layer):
    def __init__(self, input_dim=28*28, z_dim=200, trainable_std=True, distribution=tfd.MultivariateNormalDiag):
        super().__init__()
        # W matrix in mean of p(x|z)
        self.W = tf.Variable(initial_value=np.random.normal(loc=0,scale=0.01,size=(z_dim, input_dim)).astype(np.float32), trainable=True)
        # mu vector in mean of p(x|z)
        self.mu = tf.Variable(initial_value=np.repeat(0.,input_dim).astype(np.float32), trainable=True)
        # log_std of px_z
        # XXX scalar
        self.log_var_px_z = tf.Variable(initial_value=1., trainable=trainable_std)
        self.distribution = distribution
    
    @property
    def std(self):
        std = tf.math.exp(0.5*self.log_var_px_z)
        return std

    def call(self, z):
        mean = tf.linalg.matmul(z,self.W) + self.mu
        std  = tf.math.exp(0.5*self.log_var_px_z)

        px_z = self.distribution(mean, tf.repeat(std, mean.shape[1]))

        return px_z

class LVAE(tfk.Model):
    def __init__(self,
                 enc,dec,
                 input_dim=784,
                 z_dim=200,
                 trainable_std=True,
                 name = 'linear_vae',
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.enc = enc(input_dim=input_dim,z_dim=z_dim)
        self.dec = dec(input_dim=input_dim,z_dim=z_dim,trainable_std=trainable_std)
        self.prior = tfd.MultivariateNormalDiag(tf.zeros(z_dim), tf.ones(z_dim))
    
    def call(self, inputs):
        outputs = {} 

        mu = self.dec.mu
        qz_x = self.enc(inputs, mu)
        
        z = qz_x.sample()
        px_z = self.dec(z)
        

        log_prob = tf.reduce_mean(px_z.log_prob(inputs))
        kl   = tf.reduce_mean(tfd.kl_divergence(qz_x, self.prior))
        
        elbo = log_prob - kl
        self.loss = -elbo
        
        outputs['kl'] = kl
        outputs['log_prob'] = log_prob
        logvar_px_z = self.dec.log_var_px_z 
        outputs['var_pxz'] = tf.exp(logvar_px_z)
        
        return elbo, outputs
      
    def test(self, inputs):
        mu = self.dec.mu
        qz_x = self.enc(inputs, mu)

        return qz_x

    def analytic_recon(self, x, V, W, mu, logvar_qz_x, logvar_px_z):
        wv = tf.linalg.matmul(V, W)
        x_sub_mu = x - mu
        var_px_z = tf.math.exp(0.5*logvar_px_z)**2

        wvx = tf.matmul(x_sub_mu, wv)
        xvwwvx = tf.reduce_sum(wvx * wvx, 1)
        tr_wdw = tf.linalg.trace(tf.linalg.matmul(W,tf.expand_dims(tf.exp(tf.squeeze(logvar_qz_x)), 1)*W,transpose_a=True))
        xwvx = tf.reduce_sum(wvx * x_sub_mu, 1)
        xx = tf.reduce_sum(x_sub_mu * x_sub_mu, 1)
        recon_loss = 0.5 * (
                    (tr_wdw + xvwwvx - 2.0 * xwvx + xx) / var_px_z + self.input_dim *
                    (_LOG_2PI + logvar_px_z))
        return tf.reduce_mean(tf.squeeze(recon_loss))

    @tf.function
    def train(self, inputs, optimizer):
        with tf.GradientTape() as tape:
            elbo = self.call(inputs)
        gradients = tape.gradient(self.loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return elbo 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dset", default= 'mmnist', help="dataset")
    parser.add_argument("--modality", default= 'm0', help="modality in PolyMNIST")
    parser.add_argument("--z_dim", default=20, type=int, help="number of dimensions")
    parser.add_argument("--batch_size", default=512, type=int, help="batch size")
    parser.add_argument("--data_size", default=2000, type=int, help="size for model training. Use 2000 for mmnist for faster convergence.")
    parser.add_argument("--epochs", default=5000, type=int, help="number of epochs")
    parser.add_argument("--eval_every", default=1000, type=int, help="print elbo every eval_every epochs")
    parser.add_argument("--lr", default=5e-4, type=float, help="learning rate. Use 5e-4 for mmnist and svhn")
    parser.add_argument("--data_shape", default=[28,28,3], type=int, nargs='+', help="data shape")
    args = parser.parse_args()
    print(args)
     
    dset = args.dset
    avg_elbo = tf.keras.metrics.Mean()
    print("Num GPUs:", len(tf.config.list_physical_devices('GPU')))
    if len(tf.config.list_physical_devices('GPU'))==0:
        data_dir = '../data'
    else:
        data_dir = '../../coevae/data'

    batch_size = args.batch_size
    z_dim = args.z_dim
    data_size = args.data_size
    epochs = args.epochs
    eval_every = args.eval_every
    lr = args.lr
    print('loading {} data...'.format(dset))
        
    data_folder = os.path.join(data_dir,'MMNIST')
    modality = args.modality 
    ds_tr = MMNISTDataset(train=True, normalize=True, batch_size=batch_size, shuffle=True, num_modalities=5, folder_path=data_folder)
    tr_dict = ds_tr.load_data(data_dir=data_dir) # dictionaty with modalties
    AUTOTUNE = tf.data.AUTOTUNE
    data = tr_dict[modality]
    data = np.reshape(data, (data.shape[0],-1))
    idx  = random.sample(range(int(data.shape[0])),int(data_size))
    data = data[idx,...]

    input_data = tf.data.Dataset.from_tensor_slices(data).shuffle(data.shape[0]).batch(batch_size)
    model = LVAE(LEnc, LDec, z_dim=z_dim, input_dim=data.shape[1])
    
    covariance = np.cov(data, rowvar=False)
    sigma_mle, W_mle = compute_ppca_mle(covariance, z_dim)
    avg_log_p_mle = log_p_x_true(W_mle, sigma_mle, data_size, covariance)/data_size
    print("Avg. Maximum Likelihood: {:.2f} with var {:.4f}".format(avg_log_p_mle,sigma_mle))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elbos = []
    var = []
    print('training {} dset for {} epochs.'.format(dset,epochs))
    start = time.time()
    for i in range(epochs):
        for input_tensor in input_data:
            elbo,outputs = model.train(input_tensor,optimizer)
            avg_elbo(elbo)
        elbos.append(elbo)
        var.append(outputs['var_pxz'])
        if (i+1)%eval_every == 0:
            print('epoch {} ELBO {:.2f} and dec var {:.4f}'.format(i+1,elbo,outputs['var_pxz']))

    # load test set
    ds_te = MMNISTDataset(train=False, normalize=True, batch_size=batch_size, num_modalities=5, folder_path=data_folder)
    te_dict = ds_te.load_data(data_dir=data_dir) # dictionaty with modalties
    data_te = te_dict[modality]
    data_te = np.reshape(data_te, (data_te.shape[0],-1))
    idx  = random.sample(range(int(data_te.shape[0])),100)
    data_te = data_te[idx,...]
   
    #sample from posterior
    mu = model.dec.mu
    qz_x = model.enc(data_te, mu)
    z = qz_x.sample()
    px_z = model.dec(z)
    x_hat = px_z.mean()
    x_hat = np.reshape(x_hat,(-1,28,28,3))
    x_hat = tf.clip_by_value(255*x_hat, clip_value_min=0, clip_value_max=255).numpy().astype(np.uint8)
    plot_grid(x_hat, name='posterior') 

    #sample from prior
    z = model.prior.sample(100) 
    px_z = model.dec(z)
    x_hat = px_z.mean()
    x_hat = np.reshape(x_hat,(-1,28,28,3))
    x_hat = tf.clip_by_value(255*x_hat, clip_value_min=0, clip_value_max=255).numpy().astype(np.uint8)
    plot_grid(x_hat,name='prior') 
    
    # plot ELBO convergence
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(np.arange(len(elbos)), elbos, color='blue', label='ELBO')
    ax.axhline(avg_log_p_mle, ls='--', color='red', label='pPCA MLE')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('ELBO')
    plt.legend(frameon=True)
    plt.grid()
    plt.title('Dataset {} with modality {}'.format(dset,modality))
    plt.savefig('../output/elbo_{}{}.pdf'.format(args.dset,modality))
    plt.close()
    
    print('elapsed time: {}'.format(timedelta(seconds=time.time()-start)))

