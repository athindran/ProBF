from core.dynamics import LearnedAffineDynamics, AffineDynamics, ScalarDynamics
from core.util import differentiate

import torch
import gpytorch
from torch import Tensor as tarray

import numpy as np
from numpy import array, concatenate, dot, zeros

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # Kernel setup
        active_dims_u1 = np.array([0])
        active_dims_u2 = np.array([1])
        active_dims_rest = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])

        ku1 = gpytorch.kernels.LinearKernel(active_dims=active_dims_u1)
        ku2 = gpytorch.kernels.LinearKernel(active_dims=active_dims_u2)

        ka1 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(active_dims=active_dims_rest, ard_num_dims=16))
        ka2 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(active_dims=active_dims_rest, ard_num_dims=16))
        
        self.k11 = ku1*ka1
        self.k12 = ku2*ka2
        self.k1 = self.k11 + self.k12

        self.k2 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(active_dims=active_dims_rest,ard_num_dims=16))

        self.covar_module = self.k1 + self.k2

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

"""
class SafetyCoordinate(AffineDynamics, ScalarDynamics):
    #Safety function setup: Quadrotor should not get close to a ball
    def __init__(self, ex_quad, x_e, y_e, rad_square):
        self.dynamics = ex_quad
        self.x_e = x_e
        self.y_e = y_e
        self.rad_square = rad_square
        self.derivative_weight = 0.1
        
    def eval( self, x, t ):
        Check the following paper for choice of safety function
        https://hybrid-robotics.berkeley.edu/publications/ACC2016_Safety_Control_Planar_Quadrotor.pdf
        We have to use an extended higher-order CBF as described in this paper
        https://arxiv.org/pdf/2011.10721.pdf
        xpos = x[0]
        ypos = x[1]
        #theta = x[2]
        #xposdd = x[6]
        #yposdd = x[7]
        #s = sin(x[2])*(xpos-self.x_e)+cos(x[2])*(ypos-self.y_e)
        return 0.5*((xpos-self.x_e)**2 + (ypos-self.y_e)**2 - 1.0*self.rad_square - 0.2) 
    
    def dhdx( self, x , t ):
        # Note that these can be obtained by taking the 4th derivative of CBF
        derivs = self.dynamics.eval(x,t)
        #r = derivs[0:2]
        rd = derivs[2:4]
        rdd = derivs[4:6]
        rddd = derivs[6:8]
        xpos = x[0]
        ypos = x[1]
        #theta = x[2]
        #xpdot = x[3]
        #ypdot = x[4]
        #thetadot = x[5]
        #xposdd = x[6]
        #yposdd = x[7]
        #return array( [(xpos-self.x_e), (ypos-self.y_e), 0, 0, 0, 0, 0, 0])
        return array( [2*rddd[0],2*rddd[1],3*rdd[0],3*rdd[1],2*rd[0],2*rd[1],
                       (xpos-self.x_e),(ypos-self.y_e) ] )
    
    def dhdx_torch( self, x , t ):
        # Note that these can be obtained by taking the 4th derivative of CBF
        derivs = self.dynamics.eval(x,t)
        #r = derivs[0:2]
        rd = derivs[2:4]
        rdd = derivs[4:6]
        rddd = derivs[6:8]
        xpos = x[0]
        ypos = x[1]
        #theta = x[2]
        #xpdot = x[3]
        #ypdot = x[4]
        #thetadot = x[5]
        #xposdd = x[6]
        #yposdd = x[7]
        return tarray( [2*rddd[0],2*rddd[1],3*rdd[0],3*rdd[1],2*rd[0],2*rd[1],
                       (xpos-self.x_e),(ypos-self.y_e) ])
    
    def drift( self, x, t ):
        #print("Drift",dot( self.dhdx( x, t ), self.dynamics.drift( x, t ) ))
        return dot( self.dhdx( x, t ), self.dynamics.drift( x, t ) )
        
    def act(self, x, t):
        #print("Act",dot(self.dhdx( x, t ), self.dynamics.act( x, t ) ))
        return dot( self.dhdx( x, t ), self.dynamics.act( x, t ) )
"""

class SafetyCoordinate(AffineDynamics, ScalarDynamics):
    """
    Safety function setup: Quadrotor should not get close to a ball
    """
    def __init__(self, ex_quad, x_e, y_e, rad_square):
        self.dynamics = ex_quad
        self.x_e = x_e
        self.y_e = y_e
        self.rad_square = rad_square
        self.derivative_weight = 0.1
        
    def eval( self, x, t ):
        """
        Check the following paper for choice of safety function
        https://hybrid-robotics.berkeley.edu/publications/ACC2016_Safety_Control_Planar_Quadrotor.pdf
        We have to use an extended higher-order CBF as described in this paper
        https://arxiv.org/pdf/2011.10721.pdf
        """
        xpos = x[0]
        ypos = x[1]
        #theta = x[2]
        #xposdd = x[6]
        #yposdd = x[7]
        #s = sin(x[2])*(xpos-self.x_e)+cos(x[2])*(ypos-self.y_e)
        return 0.5*((xpos-self.x_e)**2 + (ypos-self.y_e)**2 - 1.0*self.rad_square - 0.2) 
    
    def dhdx( self, x , t ):
        # Note that these can be obtained by taking the 4th derivative of CBF
        derivs = self.dynamics.eval(x,t)
        #r = derivs[0:2]
        rd = derivs[2:4]
        rdd = derivs[4:6]
        rddd = derivs[6:8]
        xpos = x[0]
        ypos = x[1]
        #theta = x[2]
        #xpdot = x[3]
        #ypdot = x[4]
        #thetadot = x[5]
        #xposdd = x[6]
        #yposdd = x[7]
        #return array( [(xpos-self.x_e), (ypos-self.y_e), 0, 0, 0, 0, 0, 0])
        return array( [2*rddd[0],2*rddd[1],3*rdd[0],3*rdd[1],2*rd[0],2*rd[1],
                       (xpos-self.x_e),(ypos-self.y_e) ] )
    
    def dhdx_torch( self, x , t ):
        # Note that these can be obtained by taking the 4th derivative of CBF
        derivs = self.dynamics.eval(x,t)
        #r = derivs[0:2]
        rd = derivs[2:4]
        rdd = derivs[4:6]
        rddd = derivs[6:8]
        xpos = x[0]
        ypos = x[1]
        #theta = x[2]
        #xpdot = x[3]
        #ypdot = x[4]
        #thetadot = x[5]
        #xposdd = x[6]
        #yposdd = x[7]
        return tarray( [2*rddd[0],2*rddd[1],3*rdd[0],3*rdd[1],2*rd[0],2*rd[1],
                       (xpos-self.x_e),(ypos-self.y_e) ])
    
    def drift( self, x, t ):
        #print("Drift",dot( self.dhdx( x, t ), self.dynamics.drift( x, t ) ))
        return dot( self.dhdx( x, t ), self.dynamics.drift( x, t ) )
        
    def act(self, x, t):
        #print("Act",dot(self.dhdx( x, t ), self.dynamics.act( x, t ) ))
        return dot( self.dhdx( x, t ), self.dynamics.act( x, t ) )


"""
Safety function setup: Quadrotor should not get close to a ball
"""
class SafetyCoordinateReduced(AffineDynamics, ScalarDynamics):
    def __init__(self, ex_quad, x_e, y_e, rad):
        self.dynamics = ex_quad
        self.x_e = x_e
        self.y_e = y_e
        self.rad = rad
        self.subset = [0, 1, 2, 3, 4, 5]
        
    def eval( self, x, t ):
        """
        Check the following paper for choice of safety function
        https://hybrid-robotics.berkeley.edu/publications/ACC2016_Safety_Control_Planar_Quadrotor.pdf
        We have to use an extended higher-order CBF as described in this paper
        https://arxiv.org/pdf/2011.10721.pdf
        """
        xpos = x[0]
        ypos = x[1]
        return 0.5*((xpos-self.x_e)**2+(ypos-self.y_e)**2-1.0*self.rad)
    
    def dhdx( self, x , t ):
        # Note that these can be obtained by taking the 4th derivative of CBF
        xpos = x[0]
        ypos = x[1]
        return array( [(xpos-self.x_e), (ypos-self.y_e), 0, 0, 0, 0])
    
    def dhdx_torch( self, x , t ):
        # Note that these can be obtained by taking the 4th derivative of CBF
        xpos = x[0]
        ypos = x[1]
        return tarray( [(xpos-self.x_e), (ypos-self.y_e), 0, 0, 0, 0])
    
    def drift( self, x, t ):
        return dot( self.dhdx( x, t ), self.dynamics.quad.drift( x[self.subset], t ) )
        
    def act(self, x, t):
        return dot( self.dhdx( x, t ), self.dynamics.quad.act( x[self.subset], t ) )


class LearnedQuadSafety_gpy(LearnedAffineDynamics):
    def __init__(self, quad_safety, device):
        self.dynamics = quad_safety
        self.residual_model = None
        self.comparison_safety = None
        self.input_data_tensor = []
        self.preprocess_mean = torch.zeros((16, ))
        self.preprocess_std = torch.ones((16, ))
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

    def drift_learned(self, x, t):
        """
          Find mean and variance of control-independent dynamics b after residual modeling.
        """
        xtorch = torch.from_numpy( x )
        
        xfull = torch.cat((torch.Tensor([1.0, 1.0]), torch.divide( self.process_drift_torch(xtorch, t) - self.preprocess_mean, self.preprocess_std )))
        xfull = torch.reshape(xfull, (-1, 18)).float().to(self.device)
        
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

        return [self.dynamics.drift(x, t) + mean_b.detach().cpu().numpy().ravel() + self.comparison_safety(self.eval(x, t)), 
                variance_b.detach().cpu().numpy().ravel(), 
                varab.detach().cpu().numpy().ravel()]
    
    def drift_act_learned(self, x, t):
        """
          Find mean and variance of control-independent dynamics b after residual modeling.
        """
        xtorch = torch.from_numpy( x )
        
        xfull = torch.cat((torch.Tensor([1.0, 1.0]), torch.divide(self.process_drift_torch(xtorch, t) - self.preprocess_mean, self.preprocess_std)))
        xfull = torch.reshape(xfull, (-1, 18)).float().to(self.device)
        
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

        return [self.dynamics.drift(x, t) + drift_mean.detach().cpu().numpy().ravel() + self.comparison_safety(self.eval(x, t)), 
                drift_variance.detach().cpu().numpy().ravel(), varab.detach().cpu().numpy().ravel(), self.dynamics.act(x, t) + act_mean.detach().cpu().numpy().ravel(), 
                act_variance.detach().cpu().numpy().ravel()]


    def act_learned(self, x, t):
        """
          Find mean and variance of control-dependent dynamics a after residual modeling.
        """
        xtorch = torch.from_numpy( x )
        
        xfull = torch.cat((torch.Tensor([1.0, 1.0]), torch.divide(self.process_drift_torch(xtorch, t) - self.preprocess_mean, self.preprocess_std)))
        xfull = torch.reshape(xfull, (-1, 18)).float().to(self.device)
        
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
    
    def process_episode(self, xs, us, ts, window=9):
        half_window = (window - 1) // 2
        xs = xs[:len(us)]
        ts = ts[:len(us)]
        
        drift_inputs = array([self.process_drift(x, t) for x, t in zip(xs, ts)])
        act_inputs = array([self.process_act(x, t) for x, t in zip(xs, ts)])

        reps = array([self.dynamics.eval(x, t) for x, t in zip(xs, ts)])
        rep_dots = differentiate(reps, ts)
        rep_ddots = differentiate(rep_dots,ts[1:-1])
        rep_dddots = differentiate(rep_ddots,ts[2:-2])
        rep_ddddots = differentiate(rep_dddots,ts[3:-3])
        
        rep_dot_noms = array([self.dynamics.eval_dot(x, u, t) for x, u, t in zip(xs, us, ts)])
        j = 0
        
        drift_inputs = drift_inputs[half_window:-half_window]
        act_inputs = act_inputs[half_window:-half_window]
        rep_dot_noms = rep_dot_noms[half_window:-half_window]
        
        us = us[0:-2*half_window,:]
        
        residuals = rep_ddddots - rep_dot_noms
        
        return drift_inputs, act_inputs, us, residuals
    
    
    def init_data(self, d_drift_in, d_act_in, m, d_out):
        return [zeros((0, d_drift_in)), zeros((0, d_act_in)), zeros((0, m)), zeros(0)]
        
