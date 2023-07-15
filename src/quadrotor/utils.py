from core.systems import PlanarQuadrotor
from core.controllers import FBLinController, LQRController

from numpy import identity, linspace
from matplotlib.pyplot import subplot, plot, grid, legend, xlabel, ylabel, figure
from numpy import array, dot, linspace

def initializeSystem():# System Definitions
    """
        Initialize true planar quadrotor parameters and estimates
    """
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
    # IMPORTANT: There is a key assumption here. The stabilizing controller knows the true system. It is an oracle.
    lqr = LQRController.build(ex_quad_true_output, Q, R)
    fb_lin_true = FBLinController(ex_quad_true_output, lqr)

    return ex_quad, ex_quad_true, ex_quad_output, ex_quad_true_output, fb_lin_true  


def simulateSafetyFilter(x_0, ex_quad_true, ex_quad, flt_true, flt_est):
    """
    Simulate safety filters on estimate system with true safety filter and safety filter with estimates
    Args:
        x_0: Initial state
        ex_quad: Extended planar quadrotor with parameter estimates 
        ex_quad_true: Extended planar quadrotor with true parameters 
        ex_quad_output: Output of estimated quadrotor
        ex_quad_true_output: Output of true quadrotor
        fb_lin: Feedback linearized dynamics
    """
    # Angle-Angle Rate Safety QP Simulation
    freq = 200 # Hz
    tend = 12
    #x_0 = array([2.0, 2.0, 0, 0, 0, 0, m * g, 0])
    ts_qp = linspace(0, tend, tend*freq + 1)

    # Estimated System - Estimated Safety Filter
    qp_estest_data = ex_quad.simulate(x_0, flt_est, ts_qp)

    # True System - True Safety Filter
    qp_truetrue_data = ex_quad_true.simulate(x_0, flt_true, ts_qp)

    # True System - Estimated Safety Filter
    qp_trueest_data = ex_quad_true.simulate(x_0, flt_est, ts_qp)

    return qp_estest_data, qp_truetrue_data, qp_trueest_data, ts_qp

# PD 500 Hz Simulation   
def simulatePDplot(ex_quad, ex_quad_true, fb_lin):    
    # FBLin 400 Hz Simulation
    freq = 400 # Hz
    tend = 12

    m = ex_quad.m
    g = ex_quad.g
    # Simulating with the true system
    x_0 = array([2.0, 2.0, 0, 0, 0, 0, m * g, 0])
    ts_fblin = linspace(0, tend, tend*freq + 1)
    fblin_data = ex_quad_true.simulate(x_0, fb_lin, ts_fblin)
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