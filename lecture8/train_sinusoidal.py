from BayesianLR import BayesianLR
from plots import *
from basis_fun import gaussian_basis_function

def g(X, noise_variance):
    '''Sinusoidial function plus noise'''
    return 0.5 + np.sin(2 * np.pi * X) + np.random.normal(scale=np.sqrt(noise_variance), size=X.shape)

def main():
    # Training dataset sizes
    N_list = [3,8,20]

    # variance params in prior and posterior
    beta  = 25.0
    alpha = 2.0
    bayesian_lr = BayesianLR(beta=beta, alpha=alpha)

    # generate data
    X_tr = np.random.rand(N_list[-1], 1)
    y_tr = g(X_tr, noise_variance=1/beta)

    # Test observations
    X_te = np.linspace(0, 1, 100).reshape(-1, 1)

    # Function values without noise
    y_true = g(X_te, noise_variance=0) 

    # Design matrix of test observations
    Phi_test = bayesian_lr.design_matrix(X_te, basis_function=gaussian_basis_function, bf_args=np.linspace(0, 1, 9))
    
    plt.figure(figsize=(10, 10))
    plt.subplots_adjust(hspace=0.4)
    for i, N in enumerate(N_list):
        X_N = X_tr[:N]
        t_N = y_tr[:N]

        # Design matrix of training observations
        Phi_N = bayesian_lr.design_matrix(X_N, basis_function=gaussian_basis_function, bf_args=np.linspace(0, 1, 9))
        
        # Mean and covariance matrix of posterior
        posterior_w = bayesian_lr.posterior(Phi_N, t_N)
        m_N = posterior_w.mean().numpy()
        S_N = posterior_w.covariance().numpy()
        
        # Mean and variances of posterior predictive 
        predictive = bayesian_lr.posterior_predictive(Phi_test, m_N, S_N)
        y = predictive.mean().numpy()
        y_var = predictive.stddev().numpy()**2
        
        # Draw 5 random weight samples from posterior and compute y values
        w_samples = posterior_w.sample(5).numpy().T
        y_samples = Phi_test.dot(w_samples)
        
        plt.subplot(len(N_list), 2, i * 2 + 1)
        plot_data(X_N, t_N)
        plot_truth(X_te, y_true)
        plot_posterior_samples(X_te, y_samples)
        plt.ylim(-1., 2.0)
        plt.legend()

        plt.subplot(len(N_list), 2, i * 2 + 2)
        plot_data(X_N, t_N)
        plot_truth(X_te, y_true, label=None)
        plot_predictive(X_te, y, np.sqrt(y_var))
        plt.ylim(-1., 2.0)
        plt.legend()
    plt.savefig('sinusoidal_model.png')
    plt.close()
    print('Done!')

if __name__ == '__main__':
    main()
