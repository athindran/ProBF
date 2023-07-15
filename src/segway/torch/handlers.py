import torch
import gpytorch
from torch import Tensor as tarray

import numpy as np
from numpy import concatenate, array, zeros, dot

from core.util import differentiate
from core.dynamics import LearnedAffineDynamics, AffineDynamics, ScalarDynamics

class ExactGPModel(gpytorch.models.ExactGP):
    """GPytorch model with explicit modeling of kernel"""
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
            
        # Define kernels and covariance function of GP
        active_dimsu = np.array([0])
        ku = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel(active_dims=active_dimsu))

        active_dimsv = np.array([1, 2, 3, 4, 6, 8])
        ka = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(active_dims=active_dimsv, ard_num_dims=6))
        self.k1 = ku*ka

        kb = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(active_dims=active_dimsv, ard_num_dims=6))
        self.k2 = kb
        self.covar_module = self.k1 + self.k2

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Angle-Angle Rate Safety Function Definition
class SafetyAngleAngleRate(AffineDynamics, ScalarDynamics):
    """
      Definition of CBF for Segway and its accompanying Lie derivatives.
    """
    def __init__(self, segway, theta_e, angle_max, coeff):
        self.dynamics = segway
        self.theta_e = theta_e
        self.angle_max = angle_max
        self.coeff = coeff
        
    def eval( self, x, t ):
        """
          Definition of CBF
        """
        theta = x[1]
        theta_dot = x[3]
        return 0.5 * ( self.angle_max ** 2 - self.coeff * ( theta_dot ** 2 ) - ( theta - self.theta_e ) ** 2 )
    
    def dhdx( self, x , t ):
        """
          Derivative of CBF wrt state  
        """
        theta = x[1]
        theta_dot = x[3]
        return array( [ 0, - ( theta - self.theta_e ), 0, - self.coeff * theta_dot ] )

    def dhdx_torch( self, x , t ):
        """
          Derivative of CBF wrt state in torch
        """
        theta = x[1]
        theta_dot = x[3]
        return tarray( [ 0, - ( theta - self.theta_e ), 0, - self.coeff * theta_dot ] )
    
    def drift( self, x, t ):
        """
          Lie derivative wrt control-independent dynamics
        """
        return dot( self.dhdx( x, t ), self.dynamics.drift( x, t ) )
        
    def act(self, x, t):
        """
          Lie derivative wrt control-dependent dynamics
        """   
        return dot( self.dhdx( x, t ), self.dynamics.act( x, t ) )
    
# Learned Segway Angle-Angle Rate Safety
class LearnedSegwaySafetyAAR_gpytorch(LearnedAffineDynamics):
    """
    Learned Segway Angle-Angle Rate Safety
        Interface to use GP for residual dynamics
    """

    def __init__(self, segway_est, device):
        """
          Initialize with estimate of segway dynamics
        """
        self.dynamics = segway_est
        self.residual_model = None
        self.input_data_tensor = []
        self.preprocess_mean = torch.zeros((8,))
        self.preprocess_std = 1
        self.residual_std = 1
        self.residual_mean = 0
        self.us_scale = 1
        self.device = device
              
    def process_drift(self, x, t):
        dhdx = self.dynamics.dhdx( x, t )
        return concatenate([x, dhdx])

    def process_act(self, x, t):
        dhdx = self.dynamics.dhdx( x, t )
        return concatenate([x, dhdx])

    def process_drift_torch(self, x_torch, t):
        dhdx_torch = self.dynamics.dhdx_torch( x_torch, t )
        return torch.cat([x_torch, dhdx_torch])

    def process_act_torch(self, x_torch, t):
        dhdx_torch = self.dynamics.dhdx_torch( x_torch, t )
        return torch.cat([x_torch, dhdx_torch])
    
    def eval(self, x, t):
        return self.dynamics.eval(x, t)
    
    def drift_estimate(self, x, t):
        return self.dynamics.drift(x, t)

    def act_estimate(self, x, t):
        return self.dynamics.act(x, t)

    def drift_learned(self, x, t):
        """
          Find mean and variance of control-independent dynamics b after residual modeling.
        """
        xtorch = torch.from_numpy( x )
        xfull = torch.cat((torch.Tensor([1.0]), torch.divide(self.process_drift_torch(xtorch, t) - self.preprocess_mean, self.preprocess_std)))
        xfull = torch.reshape(xfull, (-1, 9)).float().to(self.device)

        cross1 = self.residual_model.k1(xfull, self.input_data_tensor)
        cross1 = cross1.evaluate()
        cross1 = cross1*(1/self.us_scale)
        cross2 = self.residual_model.k2(xfull, self.input_data_tensor)
        cross2 = cross2.evaluate().float()
        bmean = torch.matmul(cross2, self.alpha)
        
        bvariance = (self.residual_model.k2(xfull, xfull)).evaluate() - torch.matmul( torch.matmul(cross2, self.Kinv), cross2.T )
        varab = -torch.matmul( torch.matmul(cross1, self.Kinv), cross2.T)
        return [self.dynamics.drift(x, t) + bmean.detach().cpu().numpy().ravel() + self.comparison_safety(self.eval(x, t)), 
                bvariance.detach().cpu().numpy().ravel(), varab.detach().cpu().numpy().ravel()]
    
    def act_learned(self, x, t):
        """
          Find mean and variance of control-dependent dynamics a after residual modeling.
        """
        xtorch = torch.from_numpy( x )
        xfull = torch.cat((torch.Tensor([1.0]), torch.divide(self.process_act_torch(xtorch, t) - self.preprocess_mean, self.preprocess_std)))
        xfull = torch.reshape(xfull, (-1, 9)).float().to(self.device)

        cross = self.residual_model.k1(xfull, self.input_data_tensor).evaluate()/self.us_scale
        mean = torch.matmul(cross, self.alpha)
        avariance = self.residual_model.k1(xfull, xfull).evaluate()/(self.us_scale)**2 - torch.matmul( torch.matmul(cross, self.Kinv), cross.T)
        return self.dynamics.act(x, t) + mean.detach().cpu().numpy().ravel(), avariance.detach().cpu().numpy().ravel()
    
    def drift_act_learned(self, x, t):
        xtorch = torch.from_numpy( x )
        xfull = torch.cat((torch.Tensor([1.0]), torch.divide(self.process_drift_torch(xtorch, t) - self.preprocess_mean, self.preprocess_std)))
        xfull = torch.reshape(xfull, (-1, 9)).float().to(self.device)

        cross1 = self.residual_model.k1(xfull, self.input_data_tensor)
        cross1 = cross1.evaluate()
        cross1 = cross1*(1/self.us_scale)
        cross2 = self.residual_model.k2(xfull, self.input_data_tensor)
        cross2 = cross2.evaluate().float()
        
        bmean = torch.matmul(cross2, self.alpha)
        bvariance = (self.residual_model.k2(xfull, xfull)).evaluate() - torch.matmul( torch.matmul(cross2, self.Kinv), cross2.T )
        varab = -torch.matmul( torch.matmul(cross1, self.Kinv), cross2.T)

        amean = torch.matmul(cross1, self.alpha)
        avariance = self.residual_model.k1(xfull, xfull).evaluate()/(self.us_scale)**2 - torch.matmul( torch.matmul(cross1, self.Kinv), cross1.T)

        return [self.dynamics.drift(x, t) + bmean.detach().cpu().numpy().ravel() + self.comparison_safety(self.eval(x, t)), 
                bvariance.detach().cpu().numpy().ravel(), varab.detach().cpu().numpy().ravel(),
                self.dynamics.act(x, t) + amean.detach().cpu().numpy().ravel(), avariance.detach().cpu().numpy().ravel()]
        
    def process_episode(self, xs, us, ts, window=3):
        """
            Data pre-processing step to generate plots
        """
        #------------------------- truncating data -----------------------#
        # for tstart in range(len(us)):
        #    if np.all(np.abs(np.array(us[tstart:]))<6e-3):
        #      break
        
        tend = len(us)
        endpoint = tend

        half_window = (window - 1) // 2
        xs = xs[:len(us)]
        ts = ts[:len(us)]
        
        drift_inputs = array([self.process_drift(x, t) for x, t in zip(xs, ts)])
        act_inputs = array([self.process_act(x, t) for x, t in zip(xs, ts)])

        reps = array([self.dynamics.eval(x, t) for x, t in zip(xs, ts)])
        rep_dots = differentiate(reps, ts)
        rep_dot_noms = array([self.dynamics.eval_dot(x, u, t) for x, u, t in zip(xs, us, ts)])
        
        """
        apreds = zeros(ts.size, )
        bpreds = zeros(ts.size, )
        apredsvar = zeros(ts.size, )
        bpredsvar = zeros(ts.size, )
        respreds = zeros(ts.size, )

        j = 0
        if self.residual_model is not None:
          for x,u,t in zip(xs,us,ts):
            meanb, varb, _ = self.drift_learned(x,t)
            meana, vara = self.act_learned(x,t)
            apreds[j] = meana - self.dynamics.act(x, t)
            bpreds[j] = meanb - self.comparison_safety( self.eval(x,t) ) - self.dynamics.drift(x, t)
            apredsvar[j] = vara
            bpredsvar[j] = varb
            respreds[j] = apreds[j]*u + bpreds[j]
            j = j+1

        apreds = apreds[half_window:-half_window]
        apredsvar = apredsvar[half_window:-half_window]
        bpreds = bpreds[half_window:-half_window]
        respreds = respreds[half_window:-half_window]
        bpredsvar = bpredsvar[half_window:-half_window]
        """
        
        us = us[0:-2*half_window]
        drift_inputs = drift_inputs[half_window:-half_window]
        act_inputs = act_inputs[half_window:-half_window]
        rep_dot_noms = rep_dot_noms[half_window:-half_window]
        
        #apreds = apreds[0:endpoint]
        #apredsvar = apredsvar[0:endpoint]
        #bpreds = bpreds[0:endpoint]
        #bpredsvar = bpredsvar[0:endpoint]
        #respreds = respreds[0:endpoint]
        
        us = us[0:endpoint]
        drift_inputs = drift_inputs[0:endpoint]
        act_inputs = act_inputs[0:endpoint]
        rep_dot_noms = rep_dot_noms[0:endpoint]
        rep_dots = rep_dots[0:endpoint]
        
        residuals = rep_dots - rep_dot_noms
        
        return drift_inputs, act_inputs, us, residuals, endpoint
        #return drift_inputs, act_inputs, us, residuals, apreds, bpreds, apredsvar, bpredsvar, respreds, endpoint

    def init_data(self, d_drift_in, d_act_in, m, d_out):
        return [zeros((0, d_drift_in)), zeros((0, d_act_in)), zeros((0, m)), zeros(0), zeros(0), zeros(0), zeros(0), zeros(0), zeros(0)]
    