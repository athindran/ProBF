import numpy as np
import os
import imageio
from numpy import linspace, array, ones, zeros
from numpy.random import seed

from matplotlib import pyplot as plt
from matplotlib.pyplot import Circle
import matplotlib.patches as patches

from core.util import differentiate

from src.quadrotor.controllers.filter_controller import FilterController
from src.quadrotor.handlers import QuadrotorObstacleSafety
from src.quadrotor.utils import initializeSystemAndController
from src.utils import generateQuadPoints, findSafetyData

from src.plotting.plotting import plotQuadStatesv2, make_animation

def test_quadrotor_cbf(rnd_seed, work_dir ):
    seed(rnd_seed)

    # initial points for testing
    x_0 = array([0.0, -1.0, 0, 0, 0, 0])
    num_tests = 50
    ic_prec = 0.5
    x_0s_test = generateQuadPoints(x_0, num_tests, ic_prec)

    freq = 200
    tend = 14
    ts_qp = linspace(0, tend, tend*freq + 1)
    x_d = array([8*ones((ts_qp.size, )), 9*ones((ts_qp.size,))])
    x_dd = zeros((2, ts_qp.size))
    quad, quad_true, sqp_true = initializeSystemAndController(x_d, x_dd, freq, ts_qp)

    obstacle_position = array([1.5, 6])
    obstacle_rad2 = 4.0
    cbf_gamma = 1.2
    cbf_beta = 1.1

    safety_true = QuadrotorObstacleSafety( quad_true, obstacle_position, obstacle_rad2, gamma=cbf_gamma, beta=cbf_beta)
    flt_true = FilterController( safety_true, sqp_true)

    safety_est = QuadrotorObstacleSafety( quad, obstacle_position, obstacle_rad2, gamma=cbf_gamma, beta=cbf_beta)
    flt_est = FilterController( safety_est, sqp_true)
    
    truetrue_violations = 0
    trueest_violations = 0

    fig1 = plt.figure(figsize=(6, 4))
    ax1 = plt.gca()
    for j in range(num_tests):
        x_0_test = x_0s_test[j, :]
        
        sim_data = quad_true.simulate(x_0_test, sqp_true, ts_qp)
        xs_qp_nocbf, _ = sim_data

        qp_truetrue_data = quad_true.simulate(x_0_test, flt_true, ts_qp)
        xs_qp_truetrue, us_qp_truetrue = qp_truetrue_data

        qp_trueest_data = quad.simulate(x_0_test, flt_est, ts_qp)
        xs_qp_trueest, us_qp_trueest = qp_trueest_data

        hs_qp_truetrue, _, _, hdots_qp_truetrue = findSafetyData(safety_true, qp_truetrue_data, ts_qp)
        hs_qp_trueest, _, _, hdots_qp_trueest = findSafetyData(safety_true, qp_trueest_data, ts_qp)

        if np.any(hs_qp_trueest<0):
            trueest_violations += 1
        
        if np.any(hs_qp_truetrue<0):
            truetrue_violations += 1

        if(j==0):
            ax1.plot(xs_qp_nocbf[:, 0], xs_qp_nocbf[:, 1], 'k--', linewidth=1, label='No CBF')
            ax1.plot(xs_qp_truetrue[:, 0], xs_qp_truetrue[:, 1], 'g', linewidth=1, label='True-True')
            ax1.plot(xs_qp_trueest[:, 0], xs_qp_trueest[:, 1], 'r', linewidth=1, label='True-Est')
        else:
            ax1.plot(xs_qp_nocbf[:, 0], xs_qp_nocbf[:, 1], 'k--', linewidth=1)
            ax1.plot(xs_qp_truetrue[:, 0], xs_qp_truetrue[:, 1], 'g', linewidth=1)
            ax1.plot(xs_qp_trueest[:, 0], xs_qp_trueest[:, 1], 'r', linewidth=1)

        fig2, axes2 = plt.subplots(2, 3, figsize=(13,8))
        plotQuadStatesv2(axes2, ts_qp, xs_qp_trueest, us_qp_trueest, hs_qp_trueest, hdots_qp_trueest, label='TrueEst', clr='r')
        plotQuadStatesv2(axes2, ts_qp, xs_qp_truetrue, us_qp_truetrue, hs_qp_truetrue, hdots_qp_truetrue, label='TrueTrue', clr='g')
        fig2.savefig(os.path.join(work_dir, str(rnd_seed) + '_' + 'run' + str(j) + 'quadrotor_states.png'))
        plt.close()

        if(j==0):
            make_animation(xs_qp_truetrue, x_d, obstacle_position, obstacle_rad2, fig_folder=os.path.join(work_dir,'animation/'))

    circle = Circle((obstacle_position[0], obstacle_position[1]), np.sqrt(obstacle_rad2), color="y")
    ax1.add_patch(circle)
    ax1.plot(x_d[0, :], x_d[1, :], 'k*', label='Desired')
    ax1.set_xticks([-2, obstacle_position[0], x_d[0, 0], 13])
    ax1.set_yticks([-2, obstacle_position[1], x_d[1, 0], 13])
    ax1.set_ylabel('Y position')
    ax1.set_xlabel('X position')
    ax1.set_xlim([-2, 13])
    ax1.set_ylim([-2, 13])
    ax1.legend()

    rect = patches.Rectangle((x_0[0]-ic_prec, x_0[1]-ic_prec), 2*ic_prec, 2*ic_prec, linewidth=1, edgecolor='k', facecolor='b', alpha=0.3, label='Initial region')
    ax1.add_patch(rect)
    fig1.savefig(os.path.join(work_dir, 'quadrotor_trajectory.png'))
    plt.close()

    print('True True violations', truetrue_violations)
    print('True Est violations', trueest_violations)
    
if __name__=='__main__':
    work_dir = '/scratch/gpfs/arkumar/ProBF/test_planar_quadrotor/'
    #work_dir = './test_planar_quadrotor/'
    if not os.path.isdir(work_dir):
        os.mkdir(work_dir)
    test_quadrotor_cbf(123, work_dir)