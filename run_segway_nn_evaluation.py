#from core.dynamics import AffineDynamics, ConfigurationDynamics, LearnedDynamics, PDDynamics, ScalarDynamics
#from core.systems import Segway
from numpy import array, linspace, ones, size, sqrt, zeros
from numpy.random import seed

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import numpy as np
import os
import time
import sys
import pickle
#from tensorflow.python.client import device_lib

from src.segway.utils import initializeSystem, simulateSafetyFilter
from src.segway.keras.utils import initializeSafetyFilter
from src.segway.handlers import CombinedController
from src.segway.keras.handlers import LearnedSegwaySafetyAAR_NN, KerasResidualScalarAffineModel
from src.plotting.plotting import plotTestStates, plotPhasePlane, plotLearnedCBF
from src.utils import findSafetyData, findLearnedSafetyData_nn, generateInitialPoints

from utils.print_logger import PrintLogger

from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.pyplot import grid, legend, plot, title, xlabel, ylabel

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def run_qualitative_evaluation(seg_est, seg_true, flt_est, flt_true, pd, safety_learned, comp_safety,
                        safety_true, figure_dir="./"):
    from core.controllers import FilterController
    # Phase Plane Plotting
    # Use Learned Controller
    phi_0_learned = lambda x, t: safety_learned.drift( x, t ) + comp_safety( safety_learned.eval( x, t ) )
    phi_1_learned = lambda x, t: safety_learned.act( x, t )
    flt_learned = FilterController( seg_est, phi_0_learned, phi_1_learned, pd )

    _, _, qp_trueest_data, _ = simulateSafetyFilter(seg_true, seg_est, flt_true, flt_est)

    freq = 500 # Hz 
    tend = 3
    ts_post_qp = linspace(0, tend, tend*freq + 1)

    xs_qp_trueest, _ = qp_trueest_data
    
    x_0s_test = np.zeros((2, 4))
    x_0s_test[0, :] = array([0, 0.2, 0.2, 0.1])
    x_0s_test[1, :] = array([0, 0.22, 0.2, 0.09])
    
    for i in range(2):
      fig = plt.figure(figsize=(6, 4))
      ax = plt.gca()
      # Safe Set
      epsilon = 1e-6
      theta_h0_vals = linspace(safety_true.theta_e-safety_true.angle_max+epsilon, safety_true.theta_e + safety_true.angle_max - epsilon, 1000)
      theta_dot_h0_vals = array([sqrt((safety_true.angle_max ** 2 - (theta - safety_true.theta_e) ** 2) /safety_true.coeff) for theta in theta_h0_vals])
  
      plot(theta_h0_vals, theta_dot_h0_vals, 'k', linewidth=2.0, label='$\partial S$')
      plot(theta_h0_vals, -theta_dot_h0_vals, 'k', linewidth=1.5)
      # Initial Result
      plot(xs_qp_trueest[:, 1], xs_qp_trueest[:, 3], 'g', linewidth=1.5, label='Nominal model')
      savename = figure_dir + "/learned_pp_run{}.png".format(str(i))
      
      x_0 = x_0s_test[i, :]
      flt_learned = FilterController( seg_est, phi_0_learned, phi_1_learned, pd)
      qp_data_post = seg_true.simulate(x_0, flt_learned, ts_post_qp)
      xs_post_qp, _ = qp_data_post 

      pickle.dump( xs_post_qp, open( figure_dir + "/learned_pp_run{}.pkl".format(str(i)) , 'wb') )  
        
      # Final Result
      plot(xs_post_qp[:, 1], xs_post_qp[:, 3], 'b', linewidth=1.5, label='LCBF-NN')
      
      # Create a Rectangle patch
      rect = patches.Rectangle((0.15, 0.075), 0.1, 0.025, linewidth=1, edgecolor='k', facecolor='b', alpha=0.3)

      # Add the patch to the Axes
      ax.add_patch(rect)

      xlabel('$\\theta (rad)$', fontsize=8)
      ylabel('$\\dot{\\theta} (rad/s)$', fontsize=8)
      title('Segway Safety', fontsize = 8)
      legend(fontsize = 8)
      fig.savefig(savename, bbox_inches='tight')


def run_full_evaluation(seg_est, seg_true, flt_est, flt_true, pd, state_data, safety_learned, safety_est, safety_true, comp_safety, x_0s_test, num_tests, save_dir):                       
  from core.controllers import FilterController
  # test for 10 different random points
  num_violations = 0
  _, qp_truetrue_data, qp_trueest_data, ts_qp = simulateSafetyFilter(seg_true, seg_est, flt_true, flt_est)
  #hs_qp_estest, drifts_qp_estest, acts_qp_estest, hdots_qp_estest = findSafetyData(safety_est, qp_estest_data, ts_qp)
  hs_qp_truetrue, _, _, _ = findSafetyData(safety_true, qp_truetrue_data, ts_qp)
  hs_qp_trueest, _, _, _ = findSafetyData(safety_true, qp_trueest_data, ts_qp)

  #xs_qp_estest, us_qp_estest = qp_estest_data
  xs_qp_trueest, us_qp_trueest = qp_trueest_data
  xs_qp_truetrue, us_qp_truetrue = qp_truetrue_data

  freq = 500 # Hz
  tend = 3
  ts_post_qp = linspace(0, tend, tend*freq + 1)

  # Use Learned Controller
  phi_0_learned = lambda x, t: safety_learned.drift( x, t ) + comp_safety( safety_learned.eval( x, t ) )
  phi_1_learned = lambda x, t: safety_learned.act( x, t )
  flt_learned = FilterController( seg_est, phi_0_learned, phi_1_learned, pd )

  ebs = int(len(state_data[0])/num_episodes)
  
  for i in range(num_tests):
    print("Test: ", i+1)
    x_0 = x_0s_test[i,:]
    qp_data_post = seg_true.simulate(x_0, flt_learned, ts_post_qp)
    xs_post_qp, us_post_qp = qp_data_post

    #data_episode = safety_learned.process_episode(xs_post_qp, us_post_qp, ts_post_qp)
    savename = save_dir+"residual_predict_seed{}_run{}.png".format(str(rnd_seed),str(i))
    drifts_learned_post_qp, acts_learned_post_qp, hdots_learned_post_qp, hs_post_qp, _ = findLearnedSafetyData_nn(safety_learned, qp_data_post, ts_post_qp)
   
    # check violation of safety
    if np.any(hs_post_qp < 0):
      num_violations += 1
    
    _, drifts_post_qp, acts_post_qp, hdots_post_qp = findSafetyData(safety_est, qp_data_post, ts_post_qp)
    _, drifts_true_post_qp, acts_true_post_qp, hdots_true_post_qp = findSafetyData(safety_true, qp_data_post, ts_post_qp)
    
    theta_bound_u = ( safety_true.theta_e + safety_true.angle_max ) * ones( size( ts_post_qp ) )
    theta_bound_l = ( safety_true.theta_e - safety_true.angle_max ) * ones( size( ts_post_qp ) )

    # Plotting
    savename = save_dir+"learned_controller_seed{}_run{}.png".format(str(rnd_seed),str(i))
    plotTestStates(ts_qp, ts_post_qp, xs_qp_trueest, xs_qp_truetrue, xs_post_qp, us_qp_trueest, us_qp_truetrue, 
                    us_post_qp, hs_qp_trueest, hs_qp_truetrue, hs_post_qp, hdots_post_qp, hdots_true_post_qp, hdots_learned_post_qp , 
                      drifts_post_qp, drifts_true_post_qp, drifts_learned_post_qp, acts_post_qp, acts_true_post_qp, acts_learned_post_qp, 
                      theta_bound_u, theta_bound_l, savename)

    hs_all = []

    for ep in range(num_episodes):
      xs_curr = state_data[0][ ep*ebs:(ep+1)*ebs ] 
      hs_curr = array([safety_learned.eval(x,t) for x, t in zip(xs_curr, ts_post_qp)])
      hs_all.append( hs_curr.ravel() )

    # # LEARNED CONTROLLER
    savename = save_dir+"learned_h_seed{}_run{}.png".format(str(rnd_seed), str(i))
    plotLearnedCBF(ts_qp, hs_qp_trueest, np.array( hs_all ).ravel(), ts_post_qp, hs_post_qp, ebs, num_episodes, savename)
    
    # Phase Plane Plotting
    epsilon=1e-6
    theta_h0_vals = linspace(safety_true.theta_e-safety_true.angle_max+epsilon, safety_true.theta_e + safety_true.angle_max - epsilon, 1000)
    theta_dot_h0_vals = array([sqrt((safety_true.angle_max ** 2 - (theta - safety_true.theta_e) ** 2) /safety_true.coeff) for theta in theta_h0_vals])
    ebs = int(len(state_data[0])/num_episodes)
    
    savename = save_dir+"learned_pp_seed{}_run{}.png".format(str(rnd_seed), str(i))
    plotPhasePlane(theta_h0_vals, theta_dot_h0_vals, xs_qp_trueest, state_data, xs_post_qp, ebs, num_episodes, savename)

  # record violations
  print("seed: {}, num of violations: {}".format(rnd_seed, str(num_violations)))
  return num_violations

########################run function##########################################

def run_segway_nn_training(rnd_seed, num_episodes, num_tests, save_dir, run_quant_evaluation=False, run_qual_evaluation=False):
  from core.controllers import FilterController
  seed(rnd_seed)

  seg_est, seg_true, _, _, pd = initializeSystem()
  safety_est, safety_true, flt_est, flt_true = initializeSafetyFilter(seg_est, seg_true, pd)

  alpha = 10
  comp_safety = lambda r: alpha * r
  
  d_drift_in_seg = 8
  d_act_in_seg = 8
  d_hidden_seg= 200
  d_out_seg = 1
  res_model_seg = KerasResidualScalarAffineModel(d_drift_in_seg, d_act_in_seg, d_hidden_seg, 1, d_out_seg)
  safety_learned = LearnedSegwaySafetyAAR_NN(safety_est, res_model_seg)

  # Episodic Parameters
  weights = linspace(0, 1, num_episodes)

  # Controller Setup
  phi_0 = lambda x, t: safety_est.drift( x, t ) + comp_safety( safety_est.eval( x, t ) )
  phi_1 = lambda x, t: safety_est.act( x, t )
  flt_baseline = FilterController( seg_est, phi_0, phi_1, pd )
  flt_learned = FilterController( seg_est, phi_0, phi_1, pd )

  # Data Storage Setup
  state_data = [zeros((0, 4))]
  data = safety_learned.init_data(d_drift_in_seg, d_act_in_seg, 1, d_out_seg)

  # Simulation Setup
  freq = 500 # Hz
  tend = 3
  x_0 = array([0, 0.2, 0.2, 0.1])
  ic_prec = 0.25
  ts_qp = linspace(0, tend, tend*freq + 1)

  # initial points Setup
  x_0s = generateInitialPoints(x_0, num_episodes, ic_prec)

  # initial points for testing
  x_0s_test = generateInitialPoints(x_0, num_tests, ic_prec)

  print('x_0s:', x_0s)
  print('x_0s_test:', x_0s_test)
 
  # Episodic Learning
  # Iterate through each episode
  for i in range(num_episodes):
    print("Episode:", i+1)
    # Controller Combination
    flt_combined = CombinedController( flt_baseline, flt_learned, array([1-weights[i], weights[i]]) )
    
    # Simulation
    x_0 = x_0s[i,:]
    print("x_0", x_0)
    start_time = time.time()
    sim_data = seg_true.simulate(x_0, flt_combined, ts_qp)
    end_time = time.time()
    print("Finished simulation with average control cycle time (s): ", (end_time - start_time)/(tend*freq))

    # Data Handling
    xs, us = sim_data
    data_episode = safety_learned.process_episode(xs, us, ts_qp)

    state_data = [np.concatenate((old, new)) for old, new in zip(state_data, [xs])]
    print(state_data[0].shape)
    data = [np.concatenate((old, new)) for old, new in zip(data, data_episode)]
    
    print("Input mean",safety_learned.res_model.input_mean)

    res_model_seg = KerasResidualScalarAffineModel(d_drift_in_seg, d_act_in_seg, d_hidden_seg, 1, d_out_seg)
    safety_learned = LearnedSegwaySafetyAAR_NN(safety_est, res_model_seg)
    
    safety_learned.res_model.input_mean = np.zeros((8,))
    safety_learned.res_model.input_std = np.ones((8,))

    #fit residual model on data
    safety_learned.fit(data, 1, num_epochs=10, validation_split=0.1)

    # Controller Update
    phi_0_learned = lambda x, t: safety_learned.drift( x, t ) + comp_safety( safety_learned.eval( x, t ) )
    phi_1_learned = lambda x, t: safety_learned.act( x, t )
    flt_learned = FilterController( seg_est, phi_0_learned, phi_1_learned, pd )

  data = None  
  num_violations = 0
  if run_quant_evaluation:
    figure_quant_dir = save_dir + "quant/" 
    if not os.path.isdir(figure_quant_dir):
      os.mkdir(figure_quant_dir)

    num_violations = run_full_evaluation(seg_est, seg_true, flt_est, flt_true, pd, state_data, 
                                       safety_learned, safety_est, safety_true, comp_safety, 
                                       x_0s_test, num_tests, figure_quant_dir)
  if run_qual_evaluation:
    figure_qual_dir = save_dir + "qual/"
    if not os.path.isdir(figure_qual_dir):
      os.mkdir(figure_qual_dir)

    run_qualitative_evaluation(seg_est, seg_true, flt_est, flt_true, pd, safety_learned, comp_safety,
                        safety_true, figure_qual_dir)
  return num_violations


if __name__=='__main__':
  #rnd_seed_list = [123]
  rnd_seed_list = [ 123, 234, 345, 456, 567, 678, 789, 890, 901, 12]
  # Episodic Learning Setup

  #experiment_name = "reproduce_seg_nn_all_seeds"
  experiment_name = "debug_again"

  parent_path = "/scratch/gpfs/arkumar/ProBF/"
  parent_path = os.path.join(parent_path, experiment_name)

  if not os.path.isdir(parent_path):
    os.mkdir(parent_path)
    os.mkdir( os.path.join(parent_path, "exps") )
    os.mkdir( os.path.join(parent_path, "models") )

  figure_path = os.path.join(parent_path, "exps/segway_modular_nn/")
  model_path = os.path.join(parent_path, "models/segway_modular_nn/")

  if not os.path.isdir(figure_path):
    os.mkdir(figure_path)

  if not os.path.isdir(model_path):
    os.mkdir(model_path)

  num_violations_list = []
  num_episodes = 5
  num_tests = 10
  print_logger = None
  for rnd_seed in rnd_seed_list:
    dirs = figure_path + str(rnd_seed) + "/"

    if not os.path.isdir(dirs):
      os.mkdir(dirs) 
  
    print_logger = PrintLogger(os.path.join(dirs, 'log.txt'))
    sys.stdout = print_logger
    sys.stderr = print_logger

    num_violations = run_segway_nn_training(rnd_seed, num_episodes, num_tests, dirs, run_quant_evaluation=True, run_qual_evaluation=False)
    num_violations_list.append(num_violations)

  print_logger.reset(os.path.join(figure_path, 'log.txt'))
  print_logger.reset(os.path.join(figure_path, 'log.txt')) 
  print("num_violations_list: ", num_violations_list)
