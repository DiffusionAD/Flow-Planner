from abc import abstractmethod, ABC
import torch
from torch import nn
from torch.nn import functional as F

class DiffusionADPlanner(nn.Module):
    '''
    Training:
        Input -> Encoder -> Decoder -> Output
        
    Inference:
        Input -> Encoder -| Sampler[Decoder] |-> Output
                            
    '''

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def forward_train(self, *args, **kwargs):
        raise NotImplementedError

    def forward_inference(self, *args, **kwargs):
        raise NotImplementedError

    def encoder(self, *args, **kwargs):
        '''
        Encoder will be apply only once
        '''
        raise NotImplementedError

    def decoder(self, *args, **kwargs):
        '''
        Decoder will be packed into sampler
        '''
        raise NotImplementedError
    
class Scheduler:
    
    def __init__(self,):
        raise NotImplementedError
    
    def sample(self, x_data):
        raise NotImplementedError
    
    def generate(self, data, **sample_params):
        raise NotImplementedError