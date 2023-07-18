from core.dynamics import LearnedAffineDynamics, AffineDynamics, ScalarDynamics
from core.util import differentiate

import torch
import gpytorch
from torch import Tensor as tarray

import numpy as np
from numpy import array, zeros, newaxis

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # Kernel setup
        active_dims_u1 = np.array([0])
        active_dims_u2 = np.array([1])
        active_dims_rest = np.array([2, 3, 4, 5, 6, 7])

        ku1 = gpytorch.kernels.LinearKernel(active_dims=active_dims_u1)
        ku2 = gpytorch.kernels.LinearKernel(active_dims=active_dims_u2)

        ka1 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(active_dims=active_dims_rest, ard_num_dims=6))
        ka2 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(active_dims=active_dims_rest, ard_num_dims=6))
        
        self.k11 = ku1*ka1
        self.k12 = ku2*ka2
        self.k1 = self.k11 + self.k12

        self.k2 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(active_dims=active_dims_rest, ard_num_dims=6))

        self.covar_module = self.k1 + self.k2

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class LearnedQuadSafety_gpy(LearnedAffineDynamics):
    def __init__(self, quad_safety, device):
        self.dynamics = quad_safety
        self.residual_model = None
        self.comparison_safety = None
        self.input_data_tensor = []
        self.input_dim = 6
        self.ctr_input_dim = 8
        self.preprocess_mean = torch.zeros((self.input_dim, ))
        self.preprocess_std = torch.ones((self.input_dim, ))
        self.us_scale = 1
        self.device = device
              
    def process_drift(self, x, t):
        return x

    def process_act(self, x, t):
        return x
    
    def process_drift_torch(self, x_torch, t):
        return x_torch

    def process_act_torch(self, x_torch, t):
        return x_torch
    
    def eval(self, x, t):
        return self.dynamics.eval(x, t)

    def drift_learned(self, x, t):
        """
          Find mean and variance of control-independent dynamics b after residual modeling.
        """
        xtorch = torch.from_numpy( x )
        
        xfull = torch.cat((torch.Tensor([1.0, 1.0]), torch.divide( self.process_drift_torch(xtorch, t) - self.preprocess_mean, self.preprocess_std )))
        xfull = torch.reshape(xfull, (-1, self.ctr_input_dim)).float().to(self.device)
        
        cross11 = self.residual_model.k11(xfull, self.input_data_tensor)
        cross11 = cross11.evaluate()
        #cross11 = cross11*(1/self.us_scale[0])

        cross12 = self.residual_model.k12(xfull, self.input_data_tensor)
        cross12 = cross12.evaluate()
        #cross12 = cross12*(1/self.us_scale[1])
        
        cross2 = self.residual_model.k2(xfull, self.input_data_tensor)
        cross2 = cross2.evaluate().float()

        mean_b = torch.matmul(cross2, self.alpha)
        variance_b = (self.residual_model.k2(xfull, xfull)).evaluate() - torch.matmul( torch.matmul(cross2, self.Kinv), cross2.T )
        varab = torch.Tensor([-torch.matmul( torch.matmul(cross11, self.Kinv), cross2.T), -torch.matmul( torch.matmul(cross12, self.Kinv), cross2.T)])

        return [self.dynamics.drift(x, t) + mean_b.detach().cpu().numpy().ravel(), 
                variance_b.detach().cpu().numpy().ravel(), 
                varab.detach().cpu().numpy().ravel()]
    
    def get_cbf_params(self, x, t):
        """
          Find mean and variance of control-independent dynamics b after residual modeling.
        """
        xtorch = torch.from_numpy( x )
        
        xfull = torch.cat((torch.Tensor([1.0, 1.0]), torch.divide(self.process_drift_torch(xtorch, t) - self.preprocess_mean, self.preprocess_std)))
        xfull = torch.reshape(xfull, (-1, self.ctr_input_dim)).float().to(self.device)
        
        cross11 = self.residual_model.k11(xfull, self.input_data_tensor)
        cross11 = cross11.evaluate()
        #cross11 = cross11*(1/self.us_scale[0])

        cross12 = self.residual_model.k12(xfull, self.input_data_tensor)
        cross12 = cross12.evaluate()
        #cross12 = cross12*(1/self.us_scale[1])
        
        cross2 = self.residual_model.k2(xfull, self.input_data_tensor)
        cross2 = cross2.evaluate().float()

        drift_mean = torch.matmul(cross2, self.alpha)
        drift_variance = (self.residual_model.k2(xfull, xfull)).evaluate() - torch.matmul( torch.matmul(cross2, self.Kinv), cross2.T )
        varab = torch.Tensor([-torch.matmul( torch.matmul(cross11, self.Kinv), cross2.T), -torch.matmul( torch.matmul(cross12, self.Kinv), cross2.T)])

        act_mean = torch.Tensor([torch.matmul(cross11, self.alpha), torch.matmul(cross12, self.alpha)])
        act_variance_1 = self.residual_model.k11(xfull, xfull).evaluate() - torch.matmul( torch.matmul(cross11, self.Kinv), cross11.T)
        act_variance_2 = self.residual_model.k12(xfull, xfull).evaluate() - torch.matmul( torch.matmul(cross12, self.Kinv), cross12.T)
        
        act_variance = torch.Tensor([act_variance_1[0, 0], act_variance_2[0, 0]])
        phi0_nominal, phi1_nominal = self.dynamics.get_cbf_params(x, t)

        hfunc = self.dynamics.eval(x, t)
        phi0_learned = -drift_mean.detach().cpu().numpy().ravel()/(hfunc**2 + 1e-8)
        phi1_learned = -act_mean.detach().cpu().numpy().ravel()/(hfunc**2 + 1e-8)
        phi0_variance = drift_variance.detach().cpu().numpy().ravel()/(hfunc**2 + 1e-8)
        cross_variance = varab.detach().cpu().numpy().ravel()
        phi1_variance = act_variance.detach().cpu().numpy().ravel()/(hfunc**2 + 1e-8)

        return [phi0_nominal + phi0_learned, phi0_variance, cross_variance, phi1_nominal + phi1_learned, phi1_variance]

    def act_learned(self, x, t):
        """
          Find mean and variance of control-dependent dynamics a after residual modeling.
        """
        xtorch = torch.from_numpy( x )
        
        xfull = torch.cat((torch.Tensor([1.0, 1.0]), torch.divide(self.process_drift_torch(xtorch, t) - self.preprocess_mean, self.preprocess_std)))
        xfull = torch.reshape(xfull, (-1, self.ctr_input_dim)).float().to(self.device)
        
        cross11 = self.residual_model.k11(xfull, self.input_data_tensor)
        cross11 = cross11.evaluate()
        #cross11 = cross11*(1/self.us_scale[0])

        cross12 = self.residual_model.k12(xfull, self.input_data_tensor)
        cross12 = cross12.evaluate()
        #cross12 = cross12*(1/self.us_scale[1])
        
        act_mean = torch.Tensor([torch.matmul(cross11, self.alpha), torch.matmul(cross12, self.alpha)])
        act_variance_1 = self.residual_model.k11(xfull, xfull).evaluate() - torch.matmul( torch.matmul(cross11, self.Kinv), cross11.T)
        act_variance_2 = self.residual_model.k12(xfull, xfull).evaluate() - torch.matmul( torch.matmul(cross12, self.Kinv), cross12.T)
        
        act_variance = torch.Tensor([act_variance_1[0, 0], act_variance_2[0, 0]])
        

        return self.dynamics.act(x, t) + act_mean.detach().cpu().numpy().ravel(), act_variance.detach().cpu().numpy().ravel()
    
    def process_episode(self, xs, us, ts, window=3):
     
        half_window = (window - 1) // 2
        xs = xs[:len(us)]
        ts = ts[:len(us)]

        drift_inputs = array([self.process_drift(x, t) for x, t in zip(xs, ts)])
        act_inputs = array([self.process_act(x, t) for x, t in zip(xs, ts)])

        reps = array([self.dynamics.eval(x, t) for x, t in zip(xs, ts)])
        rep_dots = differentiate(reps, ts)

        rep_dot_noms = array([self.dynamics.eval_dot(x, u, t) for x, u, t in zip(xs, us, ts)])
        
        drift_inputs = drift_inputs[half_window:-half_window]
        act_inputs = act_inputs[half_window:-half_window]
        rep_dot_noms = rep_dot_noms[half_window:-half_window]
        
        residuals = rep_dots - rep_dot_noms

        return drift_inputs, act_inputs, us, residuals
    
    
    def init_data(self, d_drift_in, d_act_in, m, d_out):
        return [zeros((0, d_drift_in)), zeros((0, d_act_in)), zeros((0, m)), zeros(0)]
        
