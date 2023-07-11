from core.dynamics import AffineDynamics, ConfigurationDynamics, PDDynamics, ScalarDynamics, LearnedAffineDynamics
from core.systems import Segway
from core.controllers import FilterController, PDController
from core.util import differentiate
from matplotlib.pyplot import cla, figure, grid, legend, plot, subplot, xlabel, ylabel
import numpy as np
from numpy import array, dot, identity, linspace, zeros, concatenate

import torch
from torch import Tensor as tarray

class SegwayOutput(ConfigurationDynamics):
    """
      Class to represent observable output which is the first two dimensions for Segway
    """
    def __init__(self, segway):
        ConfigurationDynamics.__init__(self, segway, 1)
        
    def y(self, q):
        return q[1:] - .1383
    
    def dydq(self, q):
        return array([[0, 1]])
    
    def d2ydq2(self, q):
        return zeros((1, 2, 2))

class SegwayPD(PDDynamics):
    """
      Return proportional and derivative terms for use by the PD controller
    """
    def proportional(self, x, t):
        return x[0:2] - array([0, .1383])
    
    def derivative(self, x, t):
        return x[2:4]
    
# Learned Segway Angle-Angle Rate Safety
class LearnedSegwaySafetyAAR(LearnedAffineDynamics):
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
        self.input_data = []
        self.preprocess_mean = torch.zeros((8,))
        self.preprocess_std = 1
        self.residual_std = 1
        self.residual_mean = 0
        self.usstd = 1
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
        return concatenate([x_torch, dhdx_torch])
    
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
        cross1 = cross1*(1/self.usstd)
        cross2 = self.residual_model.k2(xfull, self.input_data_tensor)
        cross2 = cross2.evaluate().float()
        bmean = torch.matmul(cross2, self.alpha)
        
        mean = bmean*self.residual_std
        variance = (self.residual_model.k2(xfull, xfull)).evaluate() - torch.matmul( torch.matmul(cross2, self.Kinv), cross2.T )
        varab = -torch.matmul( torch.matmul(cross1, self.Kinv), cross2.T)
        return [self.dynamics.drift(x, t) + mean.detach().cpu().numpy().ravel() + self.residual_mean + self.comparison_safety(self.eval(x, t)), variance.detach().cpu().numpy().ravel(), varab.detach().cpu().numpy().ravel()]
    
    def act_learned(self, x, t):
        """
          Find mean and variance of control-dependent dynamics a after residual modeling.
        """
        xtorch = torch.from_numpy( x )
        xfull = torch.cat((torch.Tensor([1.0]), torch.divide(self.process_drift_torch(xtorch, t) - self.preprocess_mean, self.preprocess_std)))
        xfull = torch.reshape(xfull, (-1, 9)).float().to(self.device)

        cross = self.residual_model.k1(xfull, self.input_data_tensor).evaluate()/self.usstd
        mean = torch.matmul(cross, self.alpha)
        variancequad = self.residual_model.k1(xfull, xfull).evaluate()/(self.usstd)**2 - torch.matmul( torch.matmul(cross, self.Kinv), cross.T)
        return self.dynamics.act(x, t) + mean.detach().cpu().numpy().ravel(), variancequad.detach().cpu().numpy().ravel()
    
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
    
    """
    def actvar(self, x, t):
        xfull = np.concatenate(([1],np.divide(self.process_act(x, t)-self.preprocess_mean,self.preprocess_std)))
        cross = self.k1(xfull, self.input_data)
        mean = np.dot(cross, self.alpha)*self.residual_std
        variancequad = self.k1(xfull, xfull) - np.dot( np.dot( cross, self.Kinv ), cross)
        variancequad = variancequad.cpu().numpy()
        sigma1 = 0
        return [self.dynamics.act(x, t) + mean, -sigma1*np.sqrt(variancequad)]
    """ 

def initializeSystem():
    # ## Segway Setup & FBLin/PD + PD Simulation

    # Parameter Estimates
    m_b = 44.798 
    m_w = 2.485 
    J_w = 0.055936595310797 
    a_2 = -0.023227187592750 
    c_2 = 0.166845864363019 
    B_2 = 2.899458828344427 
    R = 0.086985141514373 
    K = 0.141344665167821 
    r = 0.195
    g = 9.81
    f_d = 0.076067344020759 
    f_v = 0.002862586216301 
    V_nom = 57

    param_est = array( [ m_b, m_w, J_w, a_2, c_2, B_2, R, K, r, g, f_d, f_v, V_nom ] )

    # err_perc= 0.15
    # param_true = array( [ param * uniform( 1 - err_perc, 1 + err_perc ) for param in param_est ] )
    param_true = array([5.10993081e+01, 2.40285994e+00, 5.87607726e-02, -2.62878375e-02, 1.68920195e-01,  2.69109501e+00,  
       7.63837672e-02,  1.20480362e-01, 2.22933943e-01,  9.70960195e+00,  8.21860118e-02,  2.78607935e-03, 4.99833464e+01])

        
    seg_est = Segway(*param_est)
    seg_true = Segway(*param_true)
    sego_est = SegwayOutput(seg_est)
    sego_true = SegwayOutput(seg_true)
    segpd = SegwayPD()

    Q, R = 1000 * identity(2), identity(1)
    K_p = -2*array([[0, 0.8]])
    K_d = -2*array([[0.5, 0.3]])
    pd = PDController(segpd, K_p, K_d) 

    return seg_est, seg_true, sego_est, sego_true, pd

# PD 500 Hz Simulation   
def simulatePDplot(seg_est, seg_true, pd):
    """
    Simulates the PD controller and plot the comparison of the trajectories between estimated and true dynamics
    
    Inputs:
      seg_est: Segway dynamics with parameter estimates
      seg_true: Segway dynamics with true paramters
      pd: PD controller
    """   
    freq = 500 # Hz
    tend = 3
    x_0 = array([0, 0.2, 1, 1])
    ts_pd = linspace(0, tend, tend*freq + 1)

    pd_est_data = seg_est.simulate(x_0, pd, ts_pd)
    xs_est_pd, us_est_pd = pd_est_data

    pd_true_data = seg_true.simulate(x_0, pd, ts_pd)
    xs_true_pd, us_true_pd = pd_true_data

    figure(figsize=(16, 16*2/3))

    subplot(221)
    cla()
    plot(ts_pd, xs_est_pd[:, 0], '--b', linewidth=3, label = 'Est-PD')
    plot(ts_pd, xs_true_pd[:, 0], '--r', linewidth=3, label = 'True-PD')
    grid()
    xlabel('Time (sec)', fontsize=16)
    ylabel('$x (m)$', fontsize=16)
    legend(fontsize =16)

    subplot(222)
    plot(ts_pd, xs_est_pd[:, 1], '--b', linewidth=3, label = 'Est-PD')
    plot(ts_pd, xs_true_pd[:, 1], '--r', linewidth=3, label = 'True-PD')
    grid()
    xlabel('Time (sec)', fontsize=16)
    ylabel('$\\theta (rad)$', fontsize=16)
    legend(fontsize =16)

    subplot(223)
    plot(ts_pd, xs_est_pd[:, 2], '--b', linewidth=3, label = 'Est-PD')
    plot(ts_pd, xs_true_pd[:, 2], '--r', linewidth=3, label = 'True -PD')
    grid()
    xlabel('Time (sec)', fontsize=16)
    ylabel('$\\dot{x} (m)$', fontsize=16)
    legend(fontsize =16)

    subplot(224)
    plot(ts_pd, xs_est_pd[:, 3], '--b', linewidth=3, label = 'Est-PD')
    plot(ts_pd, xs_true_pd[:, 3], '--r', linewidth=3, label = 'True -PD')
    grid()
    xlabel('Time (sec)', fontsize=16)
    ylabel('$\\dot{\\theta} (rad)$', fontsize=16)
    legend(fontsize =16)

    figure(figsize = (7.75, 7.75*2/3))
    plot(ts_pd[:-1], us_est_pd[:, 0], '--b', linewidth=3, label = 'Est-PD')
    plot(ts_pd[:-1], us_true_pd[:, 0], '--r', linewidth=3, label = 'True-PD')
    grid()
    xlabel('Time (sec)', fontsize=16)
    ylabel('$\\tau_1$', fontsize=16)
    legend(fontsize =16)

# Angle-Angle Rate Safety QP Setup
def initializeSafetyFilter(seg_est, seg_true, pd):
    """
    Initialize CBFs for the true and estimated system.
    
    Inputs:
      seg_est - Segway nominal parametric model
      seg_true - Segway true parametric model
      pd - PD controller
    
    Outputs:
      safety_est- CBF for seg_est
      safety_true - CBF for seg_true
      flt_est - CBF-QP filter for seg_est
      flt_True - CBF-QP filter for seg_true
    """
    
    theta_e = 0.1383
    angle_max = 0.2617 
    coeff = 1

    safety_est = SafetyAngleAngleRate( seg_est, theta_e, angle_max, coeff )
    safety_true = SafetyAngleAngleRate( seg_true, theta_e, angle_max, coeff)
    alpha = 10
    comp_safety = lambda r: alpha * r
    phi_0_est = lambda x, t: safety_est.drift( x, t ) + comp_safety( safety_est.eval( x, t ) )
    phi_1_est = lambda x, t: safety_est.act( x, t )
    phi_0_true = lambda x, t: safety_true.drift( x, t ) + comp_safety( safety_true.eval( x, t ) )
    phi_1_true = lambda x, t: safety_true.act( x, t )

    flt_est = FilterController( seg_est, phi_0_est, phi_1_est, pd )
    flt_true = FilterController( seg_true, phi_0_true, phi_1_true, pd)
    
    return safety_est, safety_true, flt_est, flt_true

def simulateSafetyFilter(seg_true, seg_est, flt_true, flt_est):
    """
    Simulate the system with the CBF-QP filtering the actions.

    Inputs:
        seg_true: True dynamics of Segway.
        seg_est: Dynamics estimate of Segway.
        flt_true: Safety Filter using true dynamics.
        flt_est: Safety Filter using estimated dynamics.

    Outputs:
        qp_estest_data: Trajectory when using flt_est on seg_est
        qp_trueest_data: Trajectory when using flt_est on seg_true
        qp_truetrue_data: Trajectory when using flt_true on seg_true
        ts_qp: Sampling instants array
    """
    # Angle-Angle Rate Safety QP Simulation
    freq = 500 # Hz
    tend = 3
    x_0 = array([0, 0.2, 0.2, 0.1])
    ts_qp = linspace(0, tend, tend*freq + 1)

    # Estimated System - Estimated Controller
    qp_estest_data = seg_est.simulate(x_0, flt_est, ts_qp)
    xs_qp_estest, us_qp_estest = qp_estest_data

    # True System - True Controller
    qp_truetrue_data = seg_true.simulate(x_0, flt_true, ts_qp)
    xs_qp_truetrue, us_qp_truetrue = qp_truetrue_data

    # True System - Estimated Controller
    qp_trueest_data = seg_true.simulate(x_0, flt_est, ts_qp)
    xs_qp_trueest, us_qp_trueest = qp_trueest_data

    return qp_estest_data, qp_truetrue_data, qp_trueest_data, ts_qp
    
    
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