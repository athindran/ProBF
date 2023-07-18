from src.quadrotor.dynamics.planar_quadrotor  import PlanarQuadrotor2D as PlanarQuadrotor
from src.quadrotor.controllers.sqp_controller import SequentialQPController
from src.quadrotor.handlers import virtualpositionCLF, orientationCLF

from numpy import identity, linspace, arctan, array, ones, zeros, diag
from matplotlib.pyplot import subplot, plot, grid, legend, xlabel, ylabel, figure
from numpy import array, dot, linspace

from copy import deepcopy as copy


def initializeSystemAndController(x_d, x_dd, freq, ts_qp):# System Definitions
    """
        Initialize true planar quadrotor parameters and estimates
    """
    m = 0.7
    J = 0.4
    m_true = 1.0
    J_true = 0.5

    quad = PlanarQuadrotor(m, J)
    quad_true = PlanarQuadrotor(m_true, J_true)

    vp_clf = virtualpositionCLF(quad_true, ts_qp=ts_qp, x_d=x_d, x_dd=x_dd, freq=freq, k=0.5, epsilon=0.7, eta=0.5)
    Q = diag([1, 0.01])
    affine_orientation = orientationCLF(copy(quad_true), 0, 1, 1, 10)
    sqp_true = SequentialQPController(vp_clf, Q, affine_orientation)

    return quad, quad_true, sqp_true


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