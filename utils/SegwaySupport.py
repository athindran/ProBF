from core.dynamics import AffineDynamics, ConfigurationDynamics, PDDynamics, ScalarDynamics
from core.systems import Segway
from core.controllers import FilterController,PDController
from core.util import differentiate
from matplotlib.pyplot import cla, clf, figure, grid, legend, plot, show, subplot, xlabel, ylabel
import numpy as np
from numpy import array, concatenate, dot, identity, linspace, ones, savetxt, size, sqrt, zeros
from torch import Tensor as tarray
from numpy.random import uniform,seed
from numpy.random import permutation
from numpy import clip
from core.dynamics import LearnedAffineDynamics

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