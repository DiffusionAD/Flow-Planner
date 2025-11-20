import math
import torch

class TimeSampler():

    def __init__(self, sample_method='uniform', **sample_params):
        self.sample_method = sample_method
        self.eps = sample_params['eps']
        self.device = sample_params['device']
        self.sample_params = sample_params

        assert sample_method in ['uniform', 'logit_normal', 'cos_map', 'cos_sin', 'beta', 'cosh'], f'in valid \
            sample_mathod {sample_method}'

    def sample(self, shape):

        if self.sample_method == 'uniform':
            t = torch.rand(shape, device=self.device) * (1 - self.eps) + self.eps
            t0 = self.eps * torch.ones(shape, device=self.device)

        elif self.sample_method == 'logit_normal':
            t = torch.randn(shape, device=self.device)
            t = math.sqrt(self.sample_params['s']) * t + self.sample_params['m']
            t = 1 / (1 + torch.exp(-t))

        elif self.sample_method == 'cos_map':
            u = torch.rand(shape, device=self.device)
            t = 1 - 1 / (torch.tan(0.5 * torch.pi * u) + 1)

        elif self.sample_method == 'cosh':
            '''
            p(x) \propto cosh(alpha * (x - mu))
            :param
                alpha: the smaller, the more flat
                mu: the shifting towards 0 or 1
            '''
            assert 'alpha' in self.sample_params.keys() and 'mu' in self.sample_params.keys(), 'alpha and mu is required for cosh sampling'
            u = torch.rand(shape, device=self.device)
            alpha = self.sample_params['alpha']
            mu = self.sample_params['mu']
            Z_0 = (math.sinh(alpha * (1-mu)) + math.sinh(alpha * mu)) / alpha
            w = (alpha * Z_0 * u - math.sinh(alpha * mu)) + torch.sqrt((alpha * Z_0 * u - math.sinh(alpha * mu))**2 + 1)
            t = torch.log(w) / alpha + mu
        
        elif self.sample_method == 'beta':
            t = torch.distributions.Beta(self.sample_params['alpha'], self.sample_params['beta']).sample((shape,))
            t = self.eps + (1 - self.eps) * t

        return t