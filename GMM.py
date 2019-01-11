import numpy as np
import matplotlib.pyplot as plt


class GMM():
    def __init__(self, data):
        '''
        input <data>: numpy array containing data. 
        			  Note: rows must contain variables and
              		  columns must contain observations
        '''
        self.data = data
        self.m, self.n = data.shape
    
    
    def EM(self, k, verbose=False):
        '''
        input <k>: the number of states latent variable can attain
        input <verbose>: True/False, indicating whether
        '''
        tol = 1e-8
        self.k = k
        self.phi = np.repeat(1./k, self.n)
        self.params = list()
        
        # Initialize parameters
        samples = np.random.permutation(self.n)[:k]
        for i in range(k):
            self.params.append({'mu': self.data.T[samples[i]],
                                'sigma': np.cov(self.data)})

        
        log_L = [1]  # log-likelihood
        log_L_diff = tol + 1 # ensure while loop starts
        i = 0
        while log_L_diff > tol:
            # Update likelihood
            joint_probs = self.likelihood()
            marginal_probs = joint_probs.sum(axis=0)
            log_L.append(np.log(marginal_probs).sum())
            log_L_diff = np.abs(log_L[-1] - log_L[-2])

            # E-step
            W = joint_probs/marginal_probs

            # M-step
            self.phi = np.mean(W, axis=1) # update phi
            W_normalized = (W.T/np.sum(W, axis=1)).T
            new_mu = self.data.dot(W_normalized.T)
            for j in range(k):
                diff = (self.data.T - self.params[j]['mu']).T
                new_cov = (W_normalized[j,:] * diff).dot(diff.T)
                self.params[j]['mu'] = new_mu[:,j] # update mu
                self.params[j]['sigma'] = new_cov  # update sigma
            
            i += 1
            if verbose == True:
                print('Iteration {}'.format(i))
                print('Log-likelihood: {:.3f}'.format(log_L[-1]))
                
        self.posterior = W
        if verbose == True:
            plt.figure()
            plt.plot(range(len(log_L[1:])), log_L[1:])
            plt.title('Convergence of EM Algorithm')
            plt.ylabel('Log-Likelihood')
            plt.xlabel('Iteration')
            plt.grid()
            plt.show()
                
    
    def gauss_pdf(self, mu, sigma):
        '''
        Evaluates the multivariate gaussian pdf
        at data points in self.data
        input <mu>: mean of gaussian
        input <sigma>: covariance matrix of gaussian
        '''
        diff = (self.data.T - mu).T
        det = np.abs( np.linalg.det(sigma) )

        p = np.linalg.solve(sigma, diff) # more efficient than computing inverse
        p = (diff * p).sum(axis=0)
        p = np.exp(-0.5 * p)
        p /= ( ((2*np.pi)**self.n) * det )**0.5
        return p

    
    def likelihood(self):
        '''
        Computes an array containing joint probabilities 
        of the form P[x^(i), z^(i)=j]. The array is of
        shape (n,k), where n = number of examples and
        j = number of states latent variable can attain
        '''
        p = list()
        for j in range(self.k):
            mu = self.params[j]['mu']
            sigma = self.params[j]['sigma']
            cond_prob = self.gauss_pdf(mu, sigma)
            joint_prob = cond_prob * self.phi[j]
            p.append(joint_prob)
        p = np.vstack(p)
        return p


