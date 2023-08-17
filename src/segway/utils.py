from matplotlib.pyplot import cla, figure, grid, legend, plot, subplot, xlabel, ylabel
from numpy import array, dot, identity, linspace

from core.systems import Segway
from core.controllers import FilterController, PDController

from .handlers import SegwayOutput, SegwayPD

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


def simulateSafetyFilter(seg_true, seg_est, flt_true, flt_est, freq, tend, x_0=array([0, 0.2, 0.2, 0.1])):
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
    ts_qp = linspace(0, tend, tend*freq + 1)

    # Estimated System - Estimated Filter
    qp_estest_data = seg_est.simulate(x_0, flt_est, ts_qp)

    # True System - True Filter
    qp_truetrue_data = seg_true.simulate(x_0, flt_true, ts_qp)

    # True System - Estimated Filter
    qp_trueest_data = seg_true.simulate(x_0, flt_est, ts_qp)

    return qp_estest_data, qp_truetrue_data, qp_trueest_data, ts_qp
    
    
