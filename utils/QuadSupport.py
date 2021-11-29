from core.dynamics import AffineDynamics, ConfigurationDynamics, LearnedDynamics, PDDynamics, ScalarDynamics
from core.systems import PlanarQuadrotor
from core.controllers import Controller, FBLinController, LQRController, FilterController,PDController, QPController, FilterControllerVar
from core.util import differentiate
from matplotlib.pyplot import cla, clf, figure, grid, legend, plot, savefig, show, subplot, title, xlabel, ylabel, fill_between
import numpy as np
from numpy import array, concatenate, dot, identity, linspace, ones, savetxt, size, sqrt, zeros
from numpy.random import uniform,seed
from numpy.random import permutation
from numpy import clip
import os
from core.dynamics import LearnedAffineDynamics

def initializeSystem():# System Definitions
    m = 1.5
    g = 9.8
    J = 1.3
    m_true = 1.8
    J_true = 1.1

    quad = PlanarQuadrotor(m, J)
    quad_true = PlanarQuadrotor(m_true, J_true)
    ex_quad = PlanarQuadrotor.Extension(quad)
    ex_quad_true = PlanarQuadrotor.Extension(quad_true)
    ex_quad_output = PlanarQuadrotor.Output(ex_quad)
    ex_quad_true_output = PlanarQuadrotor.Output(ex_quad_true)

    Q = 200 * identity(8)
    R = 1*identity(2)
    lqr = LQRController.build(ex_quad_true_output, Q, R)
    fb_lin_true = FBLinController(ex_quad_true_output, lqr)

    return ex_quad, ex_quad_true, ex_quad_output, ex_quad_true_output, fb_lin_true  

# PD 500 Hz Simulation   
def simulatePDplot(ex_quad, ex_quad_true, fb_lin):    
    # FBLin 400 Hz Simulation
    freq = 400 # Hz
    tend = 12

    # Simulating with the true system
    x_0 = array([2.0, 2.0, 0, 0, 0, 0, m * g, 0])
    ts_fblin = linspace(0, tend, tend*freq + 1)
    fblin_data = ex_quad_true.simulate(x_0, fb_lin_true, ts_fblin)
    xs_fblin, us_fblin = fblin_data

    # Simulating with approximate model
    x_0 = array([2.0, 2.0, 0, 0, 0, 0, m * g, 0])
    ts_fblin = linspace(0, tend, tend*freq + 1)
    fblin_data = ex_quad.simulate(x_0, fb_lin, ts_fblin)
    xse_fblin, use_fblin = fblin_data

    # PLOTS
    figure(figsize=(20, 5))

    subplot(231)
    plot(ts_fblin, xs_fblin[:, 0], 'r--', linewidth=3,label="True")
    plot(ts_fblin, xse_fblin[:, 0], 'b--', linewidth=3,label="Estimate")
    grid()
    legend()
    xlabel('Time (sec)', fontsize=16)
    ylabel('$x (m)$', fontsize=16)

    subplot(232)
    plot(ts_fblin, xs_fblin[:, 1], 'r--', linewidth=3,label="True")
    plot(ts_fblin, xse_fblin[:, 1], 'b--', linewidth=3,label="Estimate")
    grid()
    legend()
    xlabel('Time (sec)', fontsize=16)
    ylabel('$z (m)$', fontsize=16)

    subplot(233)
    plot(ts_fblin, xs_fblin[:, 2], 'r--', linewidth=3, label="True")
    plot(ts_fblin, xse_fblin[:, 2], 'b--', linewidth=3, label="Estimate")
    grid()
    legend()
    xlabel('Time (sec)', fontsize=16)
    ylabel('$\\theta (rad)$', fontsize=16)

    subplot(234)
    plot(ts_fblin, xs_fblin[:, 6], 'r--', linewidth=3, label="True")
    plot(ts_fblin, xse_fblin[:, 6], 'b--', linewidth=3, label="Estimate")
    grid()
    legend()
    xlabel('Time (sec)', fontsize=16)
    ylabel('$f_t$', fontsize=16)

    subplot(235)
    plot(ts_fblin[:-1], us_fblin[:, 0], 'r--', linewidth=3, label="True")
    plot(ts_fblin[:-1], use_fblin[:, 0], 'b--', linewidth=3, label="Estimate")
    grid()
    legend()
    xlabel('Time (sec)', fontsize=16)
    ylabel('$\\ddot{f}_t$', fontsize=16)
    show()
    
"""
Safety function setup: Quadrotor should not get close to a ball
"""
class SafetyCoordinate(AffineDynamics, ScalarDynamics):
    def __init__(self, ex_quad, x_e, y_e, rad):
        self.dynamics = ex_quad
        self.x_e = x_e
        self.y_e = y_e
        self.rad = rad
        
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
        return 0.5*((xpos-self.x_e)**2+(ypos-self.y_e)**2-1.0*self.rad)
    
    def dhdx( self, x , t ):
        # Note that these can be obtained by taking the 4th derivative of CBF
        derivs = self.dynamics.eval(x,t)
        r = derivs[0:2]
        rd = derivs[2:4]
        rdd = derivs[4:6]
        rddd = derivs[6:8]
        #[r,rd,rdd,rddd] = 
        xpos = x[0]
        ypos = x[1]
        theta = x[2]
        xpdot = x[3]
        ypdot = x[4]
        thetadot = x[5]
        xposdd = x[6]
        yposdd = x[7]
        return array( [2*rddd[0],2*rddd[1],3*rdd[0],3*rdd[1],2*rd[0],2*rd[1],
                       (xpos-self.x_e),(ypos-self.y_e) ])
    
    def drift( self, x, t ):
        #print("Drift",dot( self.dhdx( x, t ), self.dynamics.drift( x, t ) ))
        return dot( self.dhdx( x, t ), self.dynamics.drift( x, t ) )
        
    def act(self, x, t):
        #print("Act",dot(self.dhdx( x, t ), self.dynamics.act( x, t ) ))
        return dot( self.dhdx( x, t ), self.dynamics.act( x, t ) )

def initializeSafetyFilter(ex_quad, ex_quad_true, ex_quad_output, ex_quad_true_output, fb_lin):
    #x_e = 1.8
    #y_e = 0.6
    #rad = 0.32
    x_e = 1.85
    y_e = 0.6
    rad = 0.28
    
    safety_est = SafetyCoordinate( ex_quad_output, x_e, y_e, rad)
    safety_true = SafetyCoordinate( ex_quad_true_output, x_e, y_e, rad)

    # Alpha tuning very critical
    alpha = 10
    comp_safety = lambda r: alpha * r
    phi_0_est = lambda x, t: safety_est.drift( x, t ) + comp_safety( safety_est.eval( x, t ) )
    phi_1_est = lambda x, t: safety_est.act( x, t )
    phi_0_true = lambda x, t: safety_true.drift( x, t ) + comp_safety( safety_true.eval( x, t ) )
    phi_1_true = lambda x, t: safety_true.act( x, t )

    # IMPORTANT: There is a key assumption here. The stabilizing controller knows the true system. It is an oracle.
    # IMPORTANT: BUT THE SAFETY FILTER DOES NOT KNOW THE TRUE SYSTEM
    flt_est = FilterController( ex_quad, phi_0_est, phi_1_est, fb_lin)
    flt_true = FilterController( ex_quad_true, phi_0_true, phi_1_true, fb_lin)
    
    return safety_est, safety_true, flt_est, flt_true

def simulateSafetyFilter(x_0, ex_quad_true, ex_quad, flt_true, flt_est):
    # Angle-Angle Rate Safety QP Simulation
    freq = 200 # Hz
    tend = 12
    #x_0 = array([2.0, 2.0, 0, 0, 0, 0, m * g, 0])
    ts_qp = linspace(0, tend, tend*freq + 1)

    # Estimated System - Estimated Controller
    qp_estest_data = ex_quad.simulate(x_0, flt_est, ts_qp)
    xs_qp_estest, us_qp_estest = qp_estest_data

    # True System - True Controller
    qp_truetrue_data = ex_quad_true.simulate(x_0, flt_true, ts_qp)
    xs_qp_truetrue, us_qp_truetrue = qp_truetrue_data

    # True System - Estimated Controller
    qp_trueest_data = ex_quad_true.simulate(x_0, flt_est, ts_qp)
    xs_qp_trueest, us_qp_trueest = qp_trueest_data

    return qp_estest_data, qp_truetrue_data, qp_trueest_data, ts_qp