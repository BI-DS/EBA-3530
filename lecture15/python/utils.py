import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

_LOG_2PI = np.log(2 * np.pi)

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


def compute_ppca_mle(covariance, z_dim):
    # XXX as mu_mle is simply avg over x, the log-likelihood
    # can be written using the "trace trick" where we have Sigma^-1 * S
    # where S is the data covariance matrix
    w, u = np.linalg.eigh(covariance)
    eigvals, eigvecs = w[::-1], u[:,::-1]
    missing_eigvals = eigvals[z_dim:]
    sigma_sq_mle = missing_eigvals.sum() / (eigvals.shape[0] - z_dim)

    # XXX L_M
    active_eigvals = np.diag(eigvals[:z_dim])
    # XXX U_M 
    active_components = eigvecs[:,:z_dim]

    W_mle = active_components.dot((active_eigvals - sigma_sq_mle*np.eye(z_dim))**0.5)
    return sigma_sq_mle, W_mle

def log_p_x_true(W, sigma_sq, N, data_cov):
    # XXX -log p(x) = N/2 ln det(Sigma) + N/2 tr(Sigma^-1 XX^T) + Np/2 ln(2pi)
    # p is the dimension of x and N is the number of obs

    d = data_cov.shape[0]
    # C is Sigma in a MVN
    C = W.dot(W.T) + sigma_sq * np.eye(W.shape[0])
    loglik = d * _LOG_2PI + np.linalg.slogdet(C)[1]
    loglik += np.trace(np.linalg.inv(C).dot(data_cov))
    return -loglik * N / 2

