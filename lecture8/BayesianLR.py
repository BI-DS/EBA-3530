import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf
tfd = tfp.distributions

class BayesianLR:
    def __init__(self, beta, alpha, num_basis):
        self.beta = beta
        self.alpha = alpha
        std_0 = np.sqrt(1./alpha)
        self.prior_w = tfd.MultivariateNormalDiag(loc=np.zeros(num_basis),scale_diag=np.repeat(std_0,num_basis))

    def posterior(self, Phi, t):
        """Computes mean and covariance matrix of the posterior distribution."""
        
        S_N_inv = self.alpha*np.eye(Phi.shape[1]) + self.beta * Phi.T.dot(Phi)
        S_N = np.linalg.inv(S_N_inv)
        m_N = self.beta*S_N.dot(Phi.T).dot(t)

        #posterior = tfd.MultivariateNormalFullCovariance(loc=tf.squeeze(m_N), covariance_matrix=S_N)
        posterior = tfd.MultivariateNormalTriL(loc=tf.squeeze(m_N), scale_tril=tf.linalg.cholesky(S_N))        
        return posterior


    def posterior_predictive(self, Phi, m_N, S_N):
        """Computes mean and variances of the posterior predictive distribution."""
        mu = Phi.dot(m_N)
        var = 1 / self.beta + np.sum(Phi.dot(S_N)*Phi, axis=1)
        std = np.sqrt(var)

        predictive = tfd.Normal(loc=mu, scale=std)
        
        return predictive

    def design_matrix(self, x, basis_function, bf_args=None):
        if bf_args is None:
            return np.concatenate([np.ones(x.shape), basis_function(x)], axis=1)
        else:
            return np.concatenate([np.ones(x.shape)] + [basis_function(x, bf_arg) for bf_arg in bf_args], axis=1)

