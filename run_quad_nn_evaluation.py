import numpy as np
import os
import pickle
import sys
import time

from numpy import linspace, ones, array, zeros
from numpy.random import seed

from matplotlib import pyplot as plt
from matplotlib.pyplot import Circle, plot
import matplotlib.patches as patches

from src.quadrotor.controllers.filter_controller import FilterController
from src.quadrotor.utils import initializeSystemAndController, simulateSafetyFilter
from src.quadrotor.keras.utils import initializeSafetyFilter
from src.utils import generateQuadPoints, findSafetyData, findLearnedSafetyData_nn
from utils.print_logger import PrintLogger
from src.quadrotor.handlers import CombinedController
from src.quadrotor.keras.handlers import KerasResidualScalarAffineModel, LearnedQuadSafety_NN
from src.plotting.plotting import plotQuadStatesv2, make_animation, plotQuadTrajectory


def run_qualitative_evaluation(quad, quad_true, flt_est, flt_true, sqp_true, safety_learned, safety_est, 
                        safety_true, x_d, figure_dir):
    # Phase Plane Plotting
    
    freq = 200 # Hz 
    tend = 14
    ts_post_qp = linspace(0, tend, tend*freq + 1)
    
    x_0s_test = np.zeros((2, 6))
    x_0s_test[0, :] = array([0, -1, 0., 0., 0., 0.])
    x_0s_test[1, :] = array([0.5, -0.5, 0., 0., 0., 0.])
    
    flt_learned = FilterController( safety_learned, sqp_true)

    for i in range(2):
      print("Test ", i)
      x_0_test = x_0s_test[i, :]
      _, qp_truetrue_data, qp_trueest_data, _ = simulateSafetyFilter(x_0=x_0_test, quad_true=quad_true, quad=quad, flt_true=flt_true, flt_est=flt_est)

      xs_qp_trueest, _ = qp_trueest_data
      xs_qp_truetrue, _ = qp_truetrue_data

      # Safe Set
      f = plt.figure(figsize=(5, 4))
      ax = f.gca()
      savename = figure_dir + "/learned_pp_run{}.png".format(str(i))

      start_time = time.time()
      qp_data_post = quad_true.simulate(x_0_test, flt_learned, ts_post_qp)
      end_time = time.time()
      print('Average control cycle time: ', (end_time-start_time)/(tend*freq))

      xs_post_qp, _ = qp_data_post   
        
      # Final Result
      plot(xs_post_qp[:, 0], xs_post_qp[:, 1], 'b', linewidth=1.5, label='LCBF-NN' )

      pickle.dump( xs_post_qp, open( figure_dir + "/learned_pp_run{}.pkl".format(str(i)) , 'wb') )  

      plot(xs_qp_trueest[:, 0], xs_qp_trueest[:, 1], 'g', label='Nominal Model')
      plot(xs_qp_truetrue[:, 0], xs_qp_truetrue[:, 1], 'k', label='True model')
      # Create a Rectangle patch
      rect = patches.Rectangle((-0.5, -1.5), 1.0, 1.0, linewidth=1, edgecolor='k', facecolor='b', alpha=0.3)

      # Add the patch to the Axes
      ax.add_patch(rect)

      obstacle_position = safety_true.obstacle_position
      rad_square = safety_true.obstacle_radius2
      circle = Circle((obstacle_position[0], obstacle_position[1]), np.sqrt(rad_square),color="y")
      ax.add_patch(circle)
      ax.plot(x_d[0, :], x_d[1, :], 'k*', label='Desired')
      ax.set_xticks([-2, obstacle_position[0], x_d[0, 0], 13])
      ax.set_yticks([-2, obstacle_position[1], x_d[1, 0], 15])
      ax.set_ylabel('Y position')
      ax.set_xlabel('X position')
      ax.set_xlim([-2, 13])
      ax.set_ylim([-2, 15])
      ax.legend(ncol=3, fontsize=7)
      ax.set_title('Quadrotor Safety', fontsize = 8)

      f.savefig(savename, bbox_inches='tight')

def run_full_evaluation(rnd_seed, quad, quad_true, flt_est, flt_true, sqp_true, state_data, safety_learned, safety_est, 
                        safety_true, x_0s_test, num_tests, num_episodes, save_dir):                       
  # test for 10 different random points
  num_violations = 0

  flt_learned = FilterController( safety_learned, sqp_true )
  
  trueest_violations = 0
  truetrue_violations = 0
  for i in range(num_tests):
    # Learned Controller Simulation
    # Use Learned Controller
    print("Test", i)

    x_0_test = x_0s_test[i,:]
    _, qp_truetrue_data, qp_trueest_data, ts_qp = simulateSafetyFilter(x_0=x_0_test, quad_true=quad_true, quad=quad, flt_true=flt_true, flt_est=flt_est)

    hs_qp_truetrue, _, _, hdots_qp_truetrue = findSafetyData(safety_true, qp_truetrue_data, ts_qp)
    hs_qp_trueest, _, _, hdots_qp_trueest = findSafetyData(safety_true, qp_trueest_data, ts_qp)

    xs_qp_trueest, us_qp_trueest = qp_trueest_data
    xs_qp_truetrue, us_qp_truetrue = qp_truetrue_data

    freq = 200 # Hz
    tend = 14

    ts_post_qp = linspace(0, tend, tend*freq + 1)

    qp_data_post = quad_true.simulate(x_0_test, flt_learned, ts_post_qp)
    xs_post_qp, us_post_qp = qp_data_post

    savename = save_dir+"residual_predict_seed{}_run{}.pdf".format(str(rnd_seed),str(i))
    _, _, hdots_learned_post_qp, hs_post_qp, _ = findLearnedSafetyData_nn(safety_learned, qp_data_post, ts_post_qp)
   
    # check violation of safety
    if np.any(hs_post_qp < 0.0):
      num_violations += 1

    if np.any(hs_qp_trueest<0):
      trueest_violations += 1
        
    if np.any(hs_qp_truetrue<0):
      truetrue_violations += 1
    
    #_, drifts_post_qp, acts_post_qp, hdots_post_qp = findSafetyData(safety_est, qp_data_post, ts_post_qp)
    #_, drifts_true_post_qp, acts_true_post_qp, hdots_true_post_qp = findSafetyData(safety_true, qp_data_post, ts_post_qp)

    # Plotting
    savename = save_dir + "learned_controller_seed{}_run{}.png".format(str(rnd_seed),str(i))
    fig2, axes2 = plt.subplots(2, 3, figsize=(13,8))
    plotQuadStatesv2(axes2, ts_qp, xs_qp_trueest, us_qp_trueest, hs_qp_trueest, hdots_qp_trueest, label='True model', clr='r')
    plotQuadStatesv2(axes2, ts_qp, xs_qp_truetrue, us_qp_truetrue, hs_qp_truetrue, hdots_qp_truetrue, label='Nominal model', clr='g')
    plotQuadStatesv2(axes2, ts_qp, xs_post_qp, us_post_qp, hs_post_qp, hdots_learned_post_qp, label='LCBF-NN', clr='b')
    fig2.savefig(savename)

    # Trajectory Plotting
    savename = save_dir+"learned_traj_seed{}_run{}.png".format(str(rnd_seed), str(i))
    pickle.dump(xs_post_qp, open(savename[0:-4]+".p", "wb"))
    plotQuadTrajectory(state_data, num_episodes, xs_post_qp=xs_post_qp, xs_qp_trueest=xs_qp_trueest, xs_qp_truetrue=xs_qp_truetrue, 
                       obstacle_position=safety_true.obstacle_position, rad_square=safety_true.obstacle_radius2, x_d=sqp_true.affine_dynamics_position.x_d,
                       savename=savename, title_label='LCBF-NN')

  # record violations
  print("seed: {}, num of violations: {}".format(rnd_seed, str(num_violations)))
  print("Trueest violations", trueest_violations)
  print("Truetrue violations", truetrue_violations)
  return num_violations

def run_quadrotor_nn_training(rnd_seed, num_episodes, num_tests, save_dir, run_quant_evaluation=True, run_qual_evaluation=False):
    fileh = open(save_dir+"viol.txt","w",buffering=5)
    seed(rnd_seed)

    freq = 200
    tend = 14
    ts_qp = linspace(0, tend, tend*freq + 1)
    x_d = array([8*ones((ts_qp.size, )), 9*ones((ts_qp.size,))])
    x_dd = zeros((2, ts_qp.size))

    quad, quad_true, sqp_true = initializeSystemAndController(x_d=x_d, x_dd=x_dd, freq=freq, ts_qp=ts_qp)

    obstacle_position = array([1.5, 6])
    obstacle_rad2 = 4.0
    cbf_gamma = 1.2
    cbf_beta = 1.1
    
    safety_est, safety_true, flt_est, flt_true = initializeSafetyFilter(quad=quad, quad_true=quad_true, sqp_true=sqp_true, 
                                                                        obstacle_position=obstacle_position, obstacle_rad2=obstacle_rad2,
                                                                        cbf_gamma=cbf_gamma, cbf_beta=cbf_beta)

    x_0 = array([0.0, -1.0, 0, 0, 0, 0])
    ic_prec = 0.5  
    d_drift_in_seg = 6
    d_act_in_seg = 6
    d_hidden_seg= 300
    d_out_seg = 1
    us_scale = array([1.0, 1.0])
    
    # initial points Setup
    x_0s = generateQuadPoints(x_0, num_episodes, ic_prec)

    # initial points for testing
    x_0s_test = generateQuadPoints(x_0, num_tests, ic_prec)

    print('x_0s:', x_0s)
    print('x_0s_test:', x_0s_test)

    res_model_seg = KerasResidualScalarAffineModel(d_drift_in_seg, d_act_in_seg, d_hidden_seg, 2, d_out_seg, us_scale)
    safety_learned = LearnedQuadSafety_NN(safety_est, res_model_seg)

    # Episodic Parameters
    weights = linspace(0, 1, num_episodes)

    # Controller Setup
    flt_baseline = FilterController( safety_est, sqp_true)
    flt_learned = FilterController( safety_learned, sqp_true )

    # Data Storage Setup
    state_data = [zeros((0, 6))]
    data = safety_learned.init_data(d_drift_in_seg, d_act_in_seg, 2, d_out_seg)

    # Episodic Learning
    # Iterate through each episode
    for i in range(num_episodes):
        print("Episode:", i+1)
        # Controller Combination
        flt_combined = CombinedController( flt_baseline, flt_learned, array([1-weights[i], weights[i]]) )
    
        # Simulation
        x_0 = x_0s[i,:]
        print("x_0", x_0)
        sim_data = quad_true.simulate(x_0, flt_combined, ts_qp)

        # Data Handling
        xs, us = sim_data
        data_episode = safety_learned.process_episode(xs, us, ts_qp)

        state_data = [np.concatenate((old, new)) for old, new in zip(state_data, [xs])]
        data = [np.concatenate((old, new)) for old, new in zip(data, data_episode)]
  
        res_model_seg = KerasResidualScalarAffineModel(d_drift_in_seg, d_act_in_seg, d_hidden_seg, 2, d_out_seg, us_scale)
        safety_learned = LearnedQuadSafety_NN(safety_est, res_model_seg)
    
        #fit residual model on data
        safety_learned.fit(data, 16, num_epochs=10, validation_split=0.1)

        # Controller Update
        flt_learned = FilterController( safety_learned, sqp_true )
    
    num_violations = None
    if run_quant_evaluation:
      figure_quant_dir = save_dir + "quant/" 
      if not os.path.isdir(figure_quant_dir):
        os.mkdir(figure_quant_dir)
      num_violations = run_full_evaluation(rnd_seed, quad=quad, quad_true=quad_true, flt_est=flt_est, flt_true=flt_true, sqp_true=sqp_true, state_data=state_data, 
                                       safety_learned=safety_learned, safety_est=safety_est, safety_true=safety_true,
                                         x_0s_test=x_0s_test, num_tests=num_tests, num_episodes=num_episodes, save_dir=figure_quant_dir )
    
    if run_qual_evaluation:
      figure_qual_dir = save_dir + "qual/" 
      if not os.path.isdir(figure_qual_dir):
        os.mkdir(figure_qual_dir)
      run_qualitative_evaluation(quad=quad, quad_true=quad_true, flt_est= flt_est, flt_true=flt_true, sqp_true=sqp_true, 
                                 safety_learned=safety_learned, safety_true=safety_true, safety_est=safety_est, x_d=x_d, figure_dir=figure_qual_dir)

  
    return num_violations

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
    
    _, safety_true, flt_est, flt_true = initializeSafetyFilter(quad, quad_true, sqp_true, obstacle_position=obstacle_position, obstacle_rad2=obstacle_rad2
                                                                        ,cbf_gamma=cbf_gamma, cbf_beta=cbf_beta)
    truetrue_violations = 0
    trueest_violations = 0

    fig1 = plt.figure(figsize=(6, 4))
    ax1 = plt.gca()
    for j in range(num_tests):
        x_0_test = x_0s_test[j, :]
        
        sim_data = quad_true.simulate(x_0_test, sqp_true, ts_qp)
        xs_qp_nocbf, _ = sim_data
        _, qp_truetrue_data, qp_trueest_data, ts_qp = simulateSafetyFilter(x_0=x_0_test, quad_true=quad_true, quad=quad, flt_true=flt_true, flt_est=flt_est)

        #qp_truetrue_data = quad_true.simulate(x_0_test, flt_true, ts_qp)
        xs_qp_truetrue, us_qp_truetrue = qp_truetrue_data

        #qp_trueest_data = quad_true.simulate(x_0_test, flt_est, ts_qp)
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
  rnd_seed_list = [123]
  #rnd_seed_list = [ 123, 234, 345, 456, 567, 678, 789, 890, 901, 12]
  
  # Episodic Learning Setup
  experiment_name = "check_base_quad_nn"

  parent_path = "/scratch/gpfs/arkumar/ProBF/"
  parent_path = os.path.join(parent_path, experiment_name)
  
  baseline_dir = os.path.join(parent_path, "baseline")
  if not os.path.isdir(parent_path):
    os.mkdir(parent_path)
    os.mkdir(baseline_dir)
    os.mkdir( os.path.join(parent_path, "exps") )
    os.mkdir( os.path.join(parent_path, "models") )
 
  test_quadrotor_cbf(123, baseline_dir)

  figure_path = os.path.join(parent_path, "exps/quad_modular_nn/")
  model_path = os.path.join(parent_path, "models/quad_modular_nn/")

  if not os.path.isdir(figure_path):
    os.mkdir(figure_path)

  if not os.path.isdir(model_path):
    os.mkdir(model_path)

  num_violations_list = []
  num_episodes = 7
  num_tests = 10
  print_logger = None
  for rnd_seed in rnd_seed_list:
    dirs = figure_path + str(rnd_seed) + "/"

    if not os.path.isdir(dirs):
      os.mkdir(dirs) 
  
    print_logger = PrintLogger(os.path.join(dirs, 'log.txt'))
    sys.stdout = print_logger
    sys.stderr = print_logger

    num_violations = run_quadrotor_nn_training(rnd_seed, num_episodes, num_tests, dirs, run_quant_evaluation=False, run_qual_evaluation=True)
    num_violations_list.append(num_violations)

  print_logger.reset(os.path.join(figure_path, 'log.txt'))
  print_logger.reset(os.path.join(figure_path, 'log.txt')) 
  print("num_violations_list: ", num_violations_list)