from numpy import linspace
from numpy.random import seed
import numpy as np
from numpy import zeros, array
import pickle
import os
import sys

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from utils.print_logger import PrintLogger

from src.quadrotor.utils import initializeSystem,  simulateSafetyFilter
from src.quadrotor.keras.utils import initializeSafetyFilter
from src.quadrotor.handlers import CombinedController
from src.quadrotor.keras.handlers import KerasResidualScalarAffineModel, LearnedQuadSafety_NN

from src.plotting.plotting import plotQuadStates, plotQuadTrajectory
from src.utils import findSafetyData, findLearnedSafetyData_nn, generateQuadPoints
from core.controllers import FilterController

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def run_full_evaluation(ex_quad, ex_quad_true, flt_est, flt_true, fb_lin, state_data, safety_learned, safety_est, 
                        safety_true, comparison_safety, x_0s_test, num_tests, num_episodes, save_dir):                       
  # test for 10 different random points
  num_violations = 0

  phi_0_learned = lambda x, t: safety_learned.drift( x, t ) + comparison_safety( safety_learned.eval( x, t ) )
  phi_1_learned = lambda x, t: safety_learned.act( x, t )
  flt_learned = FilterController( ex_quad, phi_0_learned, phi_1_learned, fb_lin )

  for i in range(num_tests):
    # Learned Controller Simulation
    # Use Learned Controller
    print("Test", i)

    x_0 = x_0s_test[i,:]
    _, qp_truetrue_data, qp_trueest_data, ts_qp = simulateSafetyFilter(x_0, ex_quad_true, ex_quad, flt_true, flt_est)

    hs_qp_truetrue, _, _, _ = findSafetyData(safety_true, qp_truetrue_data, ts_qp)
    hs_qp_trueest, _, _, _ = findSafetyData(safety_true, qp_trueest_data, ts_qp)

    xs_qp_trueest, us_qp_trueest = qp_trueest_data
    xs_qp_truetrue, us_qp_truetrue = qp_truetrue_data

    freq = 200 # Hz
    tend = 12

    ts_post_qp = linspace(0, tend, tend*freq + 1)

    qp_data_post = ex_quad_true.simulate(x_0, flt_learned, ts_post_qp)
    xs_post_qp, us_post_qp = qp_data_post

    savename = save_dir+"residual_predict_seed{}_run{}.pdf".format(str(rnd_seed),str(i))
    drifts_learned_post_qp, acts_learned_post_qp, hdots_learned_post_qp, hs_post_qp, hdots_post_num = findLearnedSafetyData_nn(safety_learned, qp_data_post, ts_post_qp)
   
    # check violation of safety
    if np.any(hs_post_qp < 0.0):
      num_violations += 1
    
    _, drifts_post_qp, acts_post_qp, hdots_post_qp = findSafetyData(safety_est, qp_data_post, ts_post_qp)
    _, drifts_true_post_qp, acts_true_post_qp, hdots_true_post_qp = findSafetyData(safety_true, qp_data_post, ts_post_qp)

    # Plotting
    savename = save_dir+"learned_controller_seed{}_run{}.png".format(str(rnd_seed),str(i))
    plotQuadStates(ts_qp, ts_post_qp, xs_qp_trueest, xs_qp_truetrue, xs_post_qp, us_qp_trueest, us_qp_truetrue, us_post_qp, hs_qp_trueest, hs_qp_truetrue, hs_post_qp, hdots_post_qp, hdots_true_post_qp, hdots_learned_post_qp , drifts_post_qp, drifts_true_post_qp, drifts_learned_post_qp, acts_post_qp, acts_true_post_qp, acts_learned_post_qp, savename)
    
    # Trajectory Plotting
    savename = save_dir+"learned_traj_seed{}_run{}.png".format(str(rnd_seed), str(i))
    pickle.dump(xs_post_qp, open(savename[0:-4]+".p", "wb"))
    plotQuadTrajectory(state_data, num_episodes, xs_post_qp, xs_qp_trueest, xs_qp_truetrue, safety_true.x_e, safety_true.y_e, safety_true.rad, savename, title_label='LCBF-NN')

  # record violations
  print("seed: {}, num of violations: {}".format(rnd_seed, str(num_violations)))
  return num_violations

def run_quadrotor_nn_training(rnd_seed, num_episodes, num_tests, save_dir):
  from core.controllers import FilterController

  fileh = open(save_dir+"viol.txt","w",buffering=5)
  seed(rnd_seed)

  ex_quad, ex_quad_true, ex_quad_output, ex_quad_true_output, fb_lin = initializeSystem()
  safety_est, safety_true, flt_est, flt_true = initializeSafetyFilter(ex_quad, ex_quad_true, ex_quad_output, ex_quad_true_output, fb_lin)

  alpha = 10
  comparison_safety = lambda r: alpha * r
  
  d_drift_in_seg = 14
  d_act_in_seg = 14
  d_hidden_seg= 200
  d_out_seg = 1
  us_scale = array([1.0, 1.0])
  res_model_seg = KerasResidualScalarAffineModel(d_drift_in_seg, d_act_in_seg, d_hidden_seg, 2, d_out_seg, us_scale)
  safety_learned = LearnedQuadSafety_NN(safety_est, res_model_seg)

  # Episodic Parameters
  weights = linspace(0, 1, num_episodes)

  # Controller Setup
  phi_0 = lambda x, t: safety_est.drift( x, t ) + comparison_safety( safety_est.eval( x, t ) )
  phi_1 = lambda x, t: safety_est.act( x, t )
  flt_baseline = FilterController( ex_quad, phi_0, phi_1, fb_lin)
  flt_learned = FilterController( ex_quad, phi_0, phi_1, fb_lin)

  # Data Storage Setup
  state_data = [zeros((0, 8))]
  data = safety_learned.init_data(d_drift_in_seg, d_act_in_seg, 2, d_out_seg)

  # Simulation Setup
  freq = 200 # Hz
  tend = 12
  m = ex_quad.quad.params[0]
  g = ex_quad.quad.params[2]  
  x_0 = array([2.0, 2.0, 0, 0, 0, 0, m * g, 0])
    
  ic_prec = 0.1
  ts_qp = linspace(0, tend, tend*freq + 1)

  # initial points Setup
  x_0s = generateQuadPoints(x_0, num_episodes, ic_prec)

  # initial points for testing
  x_0s_test = generateQuadPoints(x_0, num_tests, ic_prec)

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
    sim_data = ex_quad_true.simulate(x_0, flt_combined, ts_qp)

    # Data Handling
    xs, us = sim_data
    data_episode = safety_learned.process_episode(xs, us, ts_qp)

    state_data = [np.concatenate((old, new)) for old, new in zip(state_data, [xs])]
    print(state_data[0].shape)
    data = [np.concatenate((old, new)) for old, new in zip(data, data_episode)]
  
    print("Input mean",safety_learned.res_model.input_mean)

    res_model_seg = KerasResidualScalarAffineModel(d_drift_in_seg, d_act_in_seg, d_hidden_seg, 2, d_out_seg, us_scale)
    safety_learned = LearnedQuadSafety_NN(safety_est, res_model_seg)
    
    safety_learned.res_model.input_mean = np.zeros((14,))
    safety_learned.res_model.input_std = np.ones((14,))
    safety_learned.res_model.us_scale = 1.0

    #fit residual model on data
    safety_learned.fit(data, 32, num_epochs=10, validation_split=0.1)

    # Controller Update
    phi_0_learned = lambda x, t: safety_learned.drift( x, t ) + comparison_safety( safety_learned.eval( x, t ) )
    phi_1_learned = lambda x, t: safety_learned.act( x, t )
    flt_learned = FilterController( ex_quad, phi_0_learned, phi_1_learned, fb_lin )
    
  num_violations = run_full_evaluation(ex_quad, ex_quad_true, flt_est, flt_true, fb_lin, state_data, 
                                       safety_learned, safety_est, safety_true, comparison_safety,
                                         x_0s_test, num_tests, num_episodes, save_dir)
  
  return num_violations


rnd_seed_list = [123]
#rnd_seed_list = [ 123, 234, 345, 456, 567, 678, 789, 890, 901, 12]
# Episodic Learning Setup

experiment_name = "reproduce_quad_nn"

parent_path = "/scratch/gpfs/arkumar/ProBF/"
parent_path = os.path.join(parent_path, experiment_name)

if not os.path.isdir(parent_path):
  os.mkdir(parent_path)
  os.mkdir( os.path.join(parent_path, "exps") )
  os.mkdir( os.path.join(parent_path, "models") )

figure_path = os.path.join(parent_path, "exps/quad_modular_nn/")
model_path = os.path.join(parent_path, "models/quad_modular_nn/")

if not os.path.isdir(figure_path):
  os.mkdir(figure_path)

if not os.path.isdir(model_path):
  os.mkdir(model_path)

num_violations_list = []
num_episodes = 7
num_tests = 3
print_logger = PrintLogger(os.path.join(figure_path, 'log.txt'))
sys.stdout = print_logger
sys.stderr = print_logger
for rnd_seed in rnd_seed_list:
  dirs = figure_path + str(rnd_seed) + "/"

  if not os.path.isdir(dirs):
      os.mkdir(dirs) 
  
  print_logger.reset(os.path.join(dirs, 'log.txt'))
  
  num_violations = run_quadrotor_nn_training(rnd_seed, num_episodes, num_tests, dirs)
  num_violations_list.append(num_violations)

print_logger.reset(os.path.join(figure_path, 'log.txt'))
print("num_violations_list: ", num_violations_list)