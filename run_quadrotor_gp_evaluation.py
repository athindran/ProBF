import os
import time
import numpy as np
import sys
import pickle
from numpy import array, linspace, zeros
from numpy.random import seed

import torch
import gpytorch

from src.quadrotor.utils import initializeSystem, simulateSafetyFilter
from src.quadrotor.torch.utils import initializeSafetyFilter
from src.quadrotor.handlers import CombinedController
from src.quadrotor.torch.handlers import LearnedQuadSafety_gpy, ExactGPModel
from src.plotting.plotting import plotQuadStates, plotQuadTrajectory
from src.utils import findSafetyData, findLearnedSafetyData_gp, downsample, standardize, generateQuadPoints

from core.controllers import FilterController

from utils.print_logger import PrintLogger

#device = 'cpu'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device", device)


def run_full_evaluation(rnd_seed, ex_quad, ex_quad_true, flt_est, flt_true, fb_lin, state_data, 
                        safety_learned, safety_est, safety_true, comparison_safety, 
                        x_0s_test, num_tests, num_episodes, sigma, save_dir):
  from core.controllers import FilterControllerVar2
  
  # Test for 10 different random points
  num_violations = 0

  phi_0_learned = lambda x, t: safety_learned.drift_act_learned( x, t ) 
  flt_learned = FilterControllerVar2( ex_quad, phi_0_learned, fb_lin, sigma=sigma)
  
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

    savename = save_dir+"residual_predict_seed{}_run{}.png".format(str(rnd_seed), str(i))
    drifts_learned_post_qp, acts_learned_post_qp, hdots_learned_post_qp, hs_post_qp, hdots_post_num = findLearnedSafetyData_gp(safety_learned, qp_data_post, ts_post_qp)
   
    # check violation of safety
    if np.any(hs_post_qp < -0.05):
      num_violations += 1
    
    _, drifts_post_qp, acts_post_qp, hdots_post_qp = findSafetyData(safety_est, qp_data_post, ts_post_qp)
    _, drifts_true_post_qp, acts_true_post_qp, hdots_true_post_qp = findSafetyData(safety_true, qp_data_post, ts_post_qp)

    # Plotting
    savename = save_dir+"learned_controller_seed{}_run{}.png".format(str(rnd_seed), str(i))
    plotQuadStates(ts_qp, ts_post_qp, xs_qp_trueest, xs_qp_truetrue, xs_post_qp, us_qp_trueest, us_qp_truetrue, us_post_qp, hs_qp_trueest, hs_qp_truetrue, hs_post_qp, hdots_post_qp, hdots_true_post_qp, hdots_learned_post_qp , drifts_post_qp, drifts_true_post_qp, drifts_learned_post_qp, acts_post_qp, acts_true_post_qp, acts_learned_post_qp, savename)
    
    # Phase Plane Plotting
    savename = save_dir+"learned_traj_seed{}_sigma{}_run{}.png".format(str(rnd_seed), str(sigma), str(i))
    pickle.dump(xs_post_qp, open(save_dir+"learned_traj_seed{}_sigma{}_run{}.p".format(str(rnd_seed), str(sigma), str(i)),"wb"))
    plotQuadTrajectory(state_data, num_episodes, xs_post_qp, xs_qp_trueest, xs_qp_truetrue, safety_true.x_e, safety_true.y_e, safety_true.rad, savename)

  # record violations
  print("seed: {}, num of violations: {}".format(rnd_seed, str(num_violations)))
  return num_violations


########################run function##########################################
def run_quadrotor_gp_training(rnd_seed, num_episodes, model_dir, figure_dir, num_tests=10, sigma_train=0.0, sigma_val=[1.0],
                             run_quant_evaluation=True, run_qual_evaluation=False):
  from core.controllers import FilterControllerVar2
  seed(rnd_seed)

  ex_quad, ex_quad_true, ex_quad_output, ex_quad_true_output, fb_lin = initializeSystem()
  safety_est, safety_true, flt_est, flt_true = initializeSafetyFilter(ex_quad, ex_quad_true, ex_quad_output, ex_quad_true_output, fb_lin)

  alpha = 10
  comparison_safety = lambda r: alpha * r
  
  d_drift_in_seg = 14
  d_act_in_seg = 14
  d_out_seg = 1
  us_scale = array([1.0, 1.0])

  alpha = 10
  comparison_safety = lambda r: alpha * r
  
  safety_learned = LearnedQuadSafety_gpy(safety_est, device=device)

  # Episodic Parameters
  weights = linspace(0, 1, num_episodes)

  # Controller Setup
  phi_0 = lambda x, t: safety_est.drift( x, t ) + comparison_safety( safety_est.eval( x, t ) )
  phi_1 = lambda x, t: safety_est.act( x, t )
  flt_baseline = FilterController( ex_quad, phi_0, phi_1, fb_lin)
  flt_learned = FilterController( ex_quad, phi_0, phi_1, fb_lin)
  
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
    
    start_time = time.time()
    sim_data = ex_quad_true.simulate(x_0, flt_combined, ts_qp)
    end_time = time.time()
    print("Finished simulation with average control cycle time (s): ", (end_time - start_time)/(tend*freq))
    
    # Data Handling
    xs, us = sim_data
    data_episode = safety_learned.process_episode(xs, us, ts_qp)

    state_data = [np.concatenate((old, new)) for old, new in zip(state_data, [xs])]
    print(state_data[0].shape)
    data = [np.concatenate((old, new)) for old, new in zip(data, data_episode)]
    
    drift_inputs_long, act_inputs_long, us_long, residuals_long = data

    downsample_rate = 5
    drift_inputs, _, us, residuals = downsample([drift_inputs_long, act_inputs_long, us_long, residuals_long], downsample_rate)
    
    normalized_data, preprocess_mean, preprocess_std = standardize(drift_inputs)
    
    us_scale = array([[1, 1]])
    
    input_data = np.concatenate((us, normalized_data),axis=1)
    
    ndata = input_data.shape[0]
    print("Number of data points: ", ndata)
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise = 0.01
    input_data_tensor = torch.from_numpy(input_data).float()
    residuals_tensor = torch.from_numpy(residuals.ravel()).float()
    
    residual_model = ExactGPModel(input_data_tensor, residuals_tensor, likelihood)

    if i >=5:
        adam_lr = 0.006
        training_iter = 300
    else:
        adam_lr = 0.009
        training_iter = 300

    # load to gpu if possible
    if device!="cpu":
        input_data_tensor = input_data_tensor.to(device)
        residuals_tensor = residuals_tensor.to(device)
        residual_model.k11 = residual_model.k11.to(device)
        residual_model.k12 = residual_model.k12.to(device)
        residual_model.k2 = residual_model.k2.to(device)
        residual_model = residual_model.to(device)
        likelihood = likelihood.to(device)
    
    residual_model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': residual_model.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=adam_lr)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, residual_model)
    
    with gpytorch.settings.max_cg_iterations(10000), gpytorch.settings.cholesky_jitter(1e-4):
        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = residual_model(input_data_tensor)
            #print(output)
            # Calc loss and backprop gradients
            gpytorch.settings.max_cg_iterations(10000)
            loss = -mll(output, residuals_tensor)
            loss.backward()
            if(i%5==0):
                print("Loss",loss)      
                print('Iter %d/%d - Loss: %.3f   noise: %.3f' % (
                    i + 1, training_iter, loss.item(),
                    residual_model.likelihood.noise.item()
                ))
            optimizer.step()
    

    safety_learned = LearnedQuadSafety_gpy(safety_est, device=device)
    safety_learned.residual_model = residual_model
    safety_learned.us_scale = us_scale
    safety_learned.Kinv = torch.pinverse( residual_model.covar_module( input_data_tensor ).evaluate() 
                                         + residual_model.likelihood.noise.item()*torch.eye( input_data_tensor.shape[0] ).to(device) )  
    safety_learned.alpha = torch.matmul(safety_learned.Kinv, torch.from_numpy(residuals).float().to(device) )
    safety_learned.input_data_tensor = input_data_tensor
    safety_learned.preprocess_mean = torch.from_numpy( preprocess_mean[0] )
    safety_learned.preprocess_std = torch.from_numpy( preprocess_std[0] )
    safety_learned.comparison_safety = comparison_safety
    
    # Controller Update
    phi_0_learned = lambda x, t: safety_learned.drift_act_learned( x, t ) 
    #phi_1_learned = lambda x, t: safety_learned.act_learned( x, t )
    flt_learned = FilterControllerVar2( ex_quad, phi_0_learned, fb_lin, sigma=sigma_train)

  num_violations = []
  if run_quant_evaluation:
    figure_quant_dir = figure_dir + "quant/" 
    if not os.path.isdir(figure_quant_dir):
      os.mkdir(figure_quant_dir)
    
    #num_violations_a = run_full_evaluation(seg_est, seg_true, flt_est, flt_true, pd, state_data, 
    #                                       safety_learned, safety_est, safety_true, 
    #                                       x_0s_test, num_tests, 0, figure_dir)
    #print("viol-0: ", num_violations_a)
    #num_violations_b = run_full_evaluation(seg_est, seg_true, flt_est, flt_true, pd, state_data, 
    #                                       safety_learned, safety_est, safety_true, 
    #                                       x_0s_test, num_tests, 0.5, figure_dir)
    #print("viol-0.5: ", num_violations_b)
    
    for sigma_v in sigma_val:
      num_violations_c = run_full_evaluation(rnd_seed, ex_quad, ex_quad_true, flt_est, flt_true, fb_lin, state_data, 
                        safety_learned, safety_est, safety_true, comparison_safety, 
                        x_0s_test, num_tests, num_episodes, sigma_v, figure_quant_dir)
      num_violations.append( num_violations_c )
    print("viol: ", num_violations)
  
  #if run_qual_evaluation:
  #  figure_qual_dir = figure_dir + "qual/" 
  #  if not os.path.isdir(figure_qual_dir):
  #    os.mkdir(figure_qual_dir)
  #  run_qualitative_evaluation(seg_est, seg_true, flt_est, flt_true, pd, safety_learned, safety_true, figure_qual_dir)
  
  return num_violations


def run_testing():
  rnd_seed_list = [123]  
  #rnd_seed_list = [ 123, 234, 345, 456, 567, 678, 789, 890, 901, 12 ]
  # Episodic Learning Setup
  num_violations_list = []
  num_episodes = 5

  #experiment_name = "runall_quant_reproduce_allseeds"
  experiment_name = "reproduce_quad_gp"

  parent_path = "/scratch/gpfs/arkumar/ProBF/"
  parent_path = os.path.join(parent_path, experiment_name)

  if not os.path.isdir(parent_path):
    os.mkdir(parent_path)
    os.mkdir( os.path.join(parent_path, "exps") )
    os.mkdir( os.path.join(parent_path, "models") )

  figure_path = os.path.join(parent_path, "exps/quad_modular_gp/")
  model_path = os.path.join(parent_path, "models/quad_modular_gp/")

  if not os.path.isdir(figure_path):
    os.mkdir(figure_path)

  if not os.path.isdir(model_path):
    os.mkdir(model_path)

  print_logger = None
  for rnd_seed in rnd_seed_list:
    figure_dirs = figure_path + str(rnd_seed) + "/"
    model_dirs = model_path + str(rnd_seed) + "/"
    if not os.path.isdir(figure_dirs):
      os.mkdir(figure_dirs)

    if not os.path.isdir(model_dirs):    
      os.mkdir(model_dirs)

    print_logger = PrintLogger(os.path.join(figure_dirs, 'log.txt'))
    sys.stdout = print_logger
    sys.stderr = print_logger 

    num_violations_c = run_quadrotor_gp_training(rnd_seed, num_episodes, model_dirs, figure_dirs, 
                                                num_tests=2, sigma_train=0.04, sigma_val=[1.0], run_quant_evaluation=True, run_qual_evaluation=False)
    print("No. of violations", num_violations_c)
    num_violations_list.append(num_violations_c)

  print_logger.reset(os.path.join(figure_path, 'log.txt'))
  print_logger.reset(os.path.join(figure_path, 'log.txt')) 
  print("num_violations_list: ", num_violations_list)

if __name__=='__main__':
  #run_validation()
  run_testing()