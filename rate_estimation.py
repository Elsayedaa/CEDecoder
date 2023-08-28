import numpy as np
from scipy.optimize import minimize
import scipy.signal
from scipy.special import gamma

## apply minmax normalization
class norm:
    def __init__(self):
        pass
    def minmax_norm(self, array):
        if (array.max() - array.min()) != 0:
            return (array - array.min()) / (array.max() - array.min())
        else:
            return array
        
## a class containing the kernels that will be used for evaluation
class kernel:
    def __init__(self):
        pass
    
    def beta(self, support, ctr, width):
            # range of the support to which 
            # the function is applied
            T_post = support[ctr:]-ctr

            # range of the support to which 
            # the function is not applied
            T_pre = np.zeros(ctr)

            tau2 = np.sqrt(width**2/5)
            tau1 = 2*tau2
            
            # set the amplitude to a constant 4
            # to bound the function between 0 and 1
            amp = 4

            e1 = np.exp(-1*(T_post/tau1))
            e2 = np.exp(-1*(T_post/tau2))
            T_post = (amp*(e1-e2))

            return np.append(T_pre, T_post)

    def square(self, support, ctr, width):
        
        side = width/2
        support[np.where(support < ctr-side)[0]] = 0
        support[np.where(support > ctr+side)[0]] = 0
        support[np.nonzero(support)[0]] = 1

        return support

    def gaussian(
        self,
        support, # domain of the function
        ctr, # mean of the function
        std # bandwidth parameter
    ):

        amplitude = 1
        f = np.exp(-1*(((support-ctr)**2)/(2*std**2)))

        return amplitude*f
    
## a class containing the nawrot estimation method
## Nawrot, M., Aertsen, A., & Rotter, S. (1999). 
## Single-trial estimation of neuronal firing rates: 
## From single-neuron spike trains to population activity. 
## Journal of Neuroscience Methods, 94(1), 81–92. 
## https://doi.org/10.1016/s0165-0270(99)00127-2
class nawrot(kernel, norm):
    def __init__(self):
        kernel.__init__(self)
        norm.__init__(self)
        
    ## get the estimate of the rate function
    def get_rate_estimate(
        self,
        std, # bandwdith
        trial, # array containing a time series of spikes
        kfunc = 'square' # the kernel function to use (default = 'square')
    ):
        N = len(trial) # the number of time bins in the trial
        spike_times = np.nonzero(trial)[0] # times at which spikes occured
        n_spikes = len(spike_times) # total number of spikes
        placeholder = np.zeros((n_spikes, N)) # a placeholder structure to hold the data
        
        # generate a kernel function centered at the spike time
        for i, t in enumerate(spike_times):
            if kfunc == 'beta':
                placeholder[i] = trial[t]*self.beta(np.arange(N), t, std)
            if kfunc == 'square':
                placeholder[i] = trial[t]*self.square(np.arange(N), t, std)
            if kfunc == 'gaussian':
                placeholder[i] = trial[t]*self.gaussian(np.arange(N), t, std)
        
        # get the sum of the kernel functions
        return placeholder.sum(0)
    
    def objective_func(self, params, emp_rate, trial, kfunc):
        emp_rate = self.minmax_norm(emp_rate)
        est_rate = self.minmax_norm(self.get_rate_estimate(params[0], trial, kfunc))

        return ((est_rate-emp_rate)**2).mean()

    def optimize_objective_func(self, emp_rate, trial, k, kfunc):
        mse_list = []
        param_list = []

        for i in range(k):
            b = [
                (2,20), # sigma
            ]

            params_guess = [
                np.random.uniform(2, 20)
            ]
            res = minimize(self.objective_func, 
                           params_guess, 
                           args=(emp_rate, trial, kfunc), 
                           bounds = b, 
                           tol=1e-5, 
                           method="Nelder-Mead")

            param_fits_ = res.x  
            mse = self.objective_func(param_fits_, emp_rate, trial, kfunc)
            mse_list.append(mse)
            param_list.append(param_fits_)

        best_param_index = mse_list.index(min(mse_list))
        best_params = param_list[best_param_index]

        return best_params, min(mse_list)
    
## a class containing the BAKS estimation method
## Ahmadi, N., Constandinou, T. G., & Bouganis, C.-S. (2018). 
## Estimation of neuronal firing rate using Bayesian Adaptive Kernel Smoother (BAKS). 
## PLoS ONE, 13(11), e0206794. 
## https://doi.org/10.1371/journal.pone.0206794
class baks(kernel, norm):
    def __init__(self):
        kernel.__init__(self)
        norm.__init__(self)
        
    def get_ht(
        self,
        trial,
        alpha,
        beta
    ):
        N = len(trial) # the number of time bins in the trial
        spike_times = np.nonzero(trial)[0]/1000 # times at which spikes occured
        n_spikes = len(spike_times) # total number of spikes

        # placeholders for the numerator and denominator vectors
        numerator = np.zeros((n_spikes, N)) 
        denominator = np.zeros((n_spikes, N))

        # weight multiplied by the summation of the numerator & denominator
        num_weight = gamma(alpha)
        denom_weight = gamma(alpha+0.5)

        for i, t in enumerate(spike_times):
            n0 = ((np.arange(0, N/1000, 1/1000) - t)**2)/2
            n1 = 1/beta

            numerator[i] = (n0+n1)**-alpha

            d0 = n0
            d1 = n1
            d2 = -alpha-0.5

            denominator[i] = (d0+d1)**d2

        ht = (
            (num_weight*numerator.sum(0))
            /(denom_weight*denominator.sum())
        )

        return 1/np.sqrt(ht)

    ## get the estimate of the rate function
    def get_rate_estimate(
        self,
        ht, # adaptive bandwidth parameter
        trial # array containing a time series of spikes
    ):
        N = len(trial) # the number of time bins in the trial
        spike_times = np.nonzero(trial)[0] # times at which spikes occured
        n_spikes = len(spike_times) # total number of spikes
        placeholder = np.zeros((n_spikes, N)) # a placeholder structure to hold the data

        for i, t in enumerate(spike_times):
            bandwidth = ht[t]
            placeholder[i] = trial[t]*self.gaussian(np.arange(N), t, bandwidth)

        return placeholder.sum(0)
    
    def objective_func(self, params, emp_rate, trial):
        N = len(trial) # the number of time bins in the trial
        spike_times = np.nonzero(trial)[0] # times at which spikes occured
        n_spikes = len(spike_times) # total number of spikes

        ht = self.get_ht(trial, params[0], n_spikes**(2/5))
        est_rate = self.minmax_norm(self.get_rate_estimate(ht, trial))
        emp_rate = self.minmax_norm(emp_rate)

        return ((est_rate-emp_rate)**2).mean()

    def optimize_objective_func(self, emp_rate, trial, k):
        mse_list = []
        param_list = []

        for i in range(k):
            b = [
                (0.001,np.inf), # alpha
            ]

            params_guess = [
                np.random.uniform(0.001,10)
            ]
            res = minimize(self.objective_func, 
                           params_guess, 
                           args=(emp_rate, trial,), 
                           bounds = b, 
                           tol=1e-5, 
                           method="Nelder-Mead")

            param_fits_ = res.x  
            mse = self.objective_func(param_fits_, emp_rate, trial)
            mse_list.append(mse)
            param_list.append(param_fits_)

        best_param_index = mse_list.index(min(mse_list))
        best_params = param_list[best_param_index]

        return best_params, min(mse_list)
    
## a class containing the readIFR (Rate estimation via adaptation 
## to dynamics of intrinsic firing rate) estimation method
class readIFR(kernel, norm):
    def __init__(self):
        kernel.__init__(self)
        norm.__init__(self)
        self.SAMPLING_RATE = 1000
        self.bandwidths = []
    
    def bandpass_filter(self, rate, low, high):
        
        nyqs = 0.5 * self.SAMPLING_RATE
        low = low/nyqs
        high = high/nyqs

        order = 2

        b, a = scipy.signal.butter(order, [low, high], 'bandpass', analog = False)
        y = scipy.signal.filtfilt(b, a, rate, axis=0)

        return y
    
    def get_adaptive_bandwidth(self, rate):
        
        # ideally, rate parameter should be
        # (1) smoothed via a rolling average and
        # (2) calculated in spikes/sec
        bandwidth = np.e*np.log2(rate.std())
        self.bandwidths.append(bandwidth)
        return 0.01+((bandwidth-0.01) * self.minmax_norm(self.bandpass_filter(rate, 1, 30)))
    
    def get_rate_estimate(
        self,
        ht, # adaptive bandwidth parameter
        trial, # array containing a time series of spikes
        kfunc = 'square' # the kernel function to use (default = 'square')
    ):
        N = len(trial) # the number of time bins in the trial
        spike_times = np.nonzero(trial)[0] # times at which spikes occured
        n_spikes = len(spike_times) # total number of spikes
        placeholder = np.zeros((n_spikes, N)) # a placeholder structure to hold the data

        # generate a kernel function centered at the spike time
        for i, t in enumerate(spike_times):
            bandwidth = ht[t]
            if kfunc == 'beta':
                min_bandwidth = 2
                if bandwidth < min_bandwidth:
                    bandwidth = min_bandwidth
                placeholder[i] = trial[t]*self.minmax_norm(self.beta(np.arange(N), t, bandwidth))
            if kfunc == 'square':
                bandwidth = 4.5*bandwidth
                min_bandwidth = 16
                if bandwidth < min_bandwidth:
                    bandwidth = min_bandwidth
                placeholder[i] = trial[t]*self.minmax_norm(self.square(np.arange(N), t, bandwidth))
            if kfunc == 'gaussian':
                min_bandwidth = 2
                if bandwidth < min_bandwidth:
                    bandwidth = min_bandwidth
                placeholder[i] = trial[t]*self.minmax_norm(self.gaussian(np.arange(N), t, bandwidth))


        return placeholder.sum(0)