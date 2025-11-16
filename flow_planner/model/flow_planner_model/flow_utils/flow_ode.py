# flow ode: responsible for training and inference sampling.
# time sampler, flow interpolation, ode solver, cfg related

import torch
from flow_planner.model.flow_planner_model.flow_utils.velocity_model import VelocityModel
from flow_planner.model.model_base import Scheduler
from flow_matching.solver.ode_solver import ODESolver

class FlowODE(Scheduler):
    
    def __init__(self,
                 path, 
                 time_sampler,
                 cfg_weight=1.5,
                 **sample_params
                 ):
        '''
        params:
            path: defines the flow ode
            time_sampler: a random sampler used for training
            sample_params: including basic "sample_temperature" "sample_steps" and "sample_method"
        '''
        self.path = path
        self.cfg_weight = cfg_weight
        self.time_sampler = time_sampler
        self.sample_params = sample_params
        self.translation_funcs = self._get_translation_funcs()
    
    def sample(self, x_data, target_type):
        B = x_data.shape[0]
        t = self.time_sampler.sample(B).to(x_data.device)
        x_0 = torch.randn_like(x_data, device=x_data.device)
        path_sample = self.path.sample(x_0=x_0, x_1=x_data, t=t)
        
        if target_type == 'velocity':
            target = path_sample.dx_t
        elif target_type == 'x_start':
            target = path_sample.x_1
        elif target_type == 'noise':
            target = x_0
            
        return path_sample.x_t, target, t
    
    def generate(self, x_init, model_fn, model_pred_type, use_cfg, **model_extra):
        '''
        Sample a the data using the model_fn. If use_cfg, the predicted velocity will be chunked and treats the first chunk as conditioned velocity, the second as unconditioned one.
        '''
        velocity_func = self.translation_funcs[(model_pred_type, 'velocity')]
        velocity_model = VelocityModel(model_fn, self.path, velocity_func, use_cfg=use_cfg, cfg_weight=self.cfg_weight)
        
        solver = ODESolver(velocity_model=velocity_model)
        
        x_init = x_init * self.sample_params['sample_temperature']
        step_size = 1.0 / self.sample_params['sample_steps']
        sample = solver.sample(x_init=x_init,
                               step_size=step_size,
                               method=self.sample_params['sample_method'],
                               **model_extra)
        
        return sample
    
    def identity(self, x, xt, t):
        return x
    
    def _get_translation_funcs(self):
        '''
        Transform the prediction according to the flow path. Loss function is computed in the same 
        representation as "target_type", so the prediction is first transformed before fed into loss
        function.
        
        Affine flow path: x_t = alpha_t * x_1 + sigma_t * x_0
        CondOT: x_t = t * x_1 + (1 - t) * x_0 
        '''
            
        return {('velocity', 'x_start'): self.path.velocity_to_target,
                ('velocity', 'noise'): self.path.velocity_to_epsilon,
                ('x_start', 'velocity'): self.path.target_to_velocity,
                ('x_start', 'noise'): self.path.target_to_epsilon,
                ('noise', 'velocity'): self.path.epsilon_to_velocity,
                ('noise', 'x_start'): self.path.epsilon_to_target,
                ('velocity', 'velocity'): self.identity,
                ('x_start', 'x_start'): self.identity,
                ('noise', 'noise'): self.identity}