from BayesianLR import BayesianLR
from plots import *
from basis_fun import identity_basis_function

def main():
    # Training dataset sizes
    N_list = [1, 3, 20]

    # variance params in prior and posterior
    beta  = 25.0
    alpha = 2.0

    # true paramters
    f_w0 = -0.3
    f_w1 =  0.5

    # create object
    num_basis = 2 # 2 parameters
    bayesian_lr = BayesianLR(beta=beta, alpha=alpha, num_basis=num_basis)
    
    # generate data
    X_tr = np.random.rand(N_list[-1], 1) * 2 - 1
    y_tr = f_w0 + f_w1*X_tr + np.random.normal(scale=np.sqrt(1/beta), size=X_tr.shape)

    # Test observations
    X_te = np.linspace(-1, 1, 100).reshape(-1, 1)

    # Function values without noise
    y_true = f_w0 + f_w1 * X_te 

    # Design matrix of test observations
    Phi_test = bayesian_lr.design_matrix(X_te, identity_basis_function)
    
    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(hspace=0.4)

    plt.subplot(len(N_list)+1, 3, 1)
    m_0 = np.zeros(num_basis)
    S_0 = 1/alpha * np.eye(num_basis)
    plot_posterior(m_0, S_0, f_w0, f_w1)
    plt.title(f'Prior density')
    plt.legend()
    
    plt.subplot(len(N_list)+1, 3, 2)
    plot_truth(X_te, y_true)
    w_samples = bayesian_lr.prior_w.sample(5).numpy().T
    y_samples = Phi_test.dot(w_samples)
    plot_posterior_samples(X_te, y_samples)
    plt.ylim(-1.5, 1.0)
    plt.legend()

    plt.subplot(len(N_list)+1, 3, 3)
    plot_truth(X_te, y_true, label=None)
    predictive = bayesian_lr.posterior_predictive(Phi_test, m_0, S_0)
    y = predictive.mean().numpy()
    y_var = predictive.stddev().numpy()**2
    plot_predictive(X_te, y, np.sqrt(y_var))
    plt.ylim(-1.5, 1.0)
    plt.legend()

    for i, N in enumerate(N_list):
        X_N = X_tr[:N]
        t_N = y_tr[:N]

        # Design matrix of training observations
        Phi_N = bayesian_lr.design_matrix(X_N, identity_basis_function)
        
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
        
        plt.subplot(len(N_list)+1, 3, i * 3 + 4)
        plot_posterior(m_N, S_N, f_w0, f_w1)
        plt.title(f'Posterior density (N = {N})')
        plt.legend()
        
        plt.subplot(len(N_list)+1, 3, i * 3 + 5)
        plot_data(X_N, t_N)
        plot_truth(X_te, y_true)
        plot_posterior_samples(X_te, y_samples)
        plt.ylim(-1.5, 1.0)
        plt.legend()

        plt.subplot(len(N_list)+1, 3, i * 3 + 6)
        plot_data(X_N, t_N)
        plot_truth(X_te, y_true, label=None)
        plot_predictive(X_te, y, np.sqrt(y_var))
        plt.ylim(-1.5, 1.0)
        plt.legend()
    plt.savefig('linear_model.png')
    plt.close()
    print('Done!')

if __name__ == '__main__':
    main()
