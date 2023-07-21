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

import torch
import gpytorch

from src.quadrotor.controllers.filter_controller_qcqp import FilterControllerQCQP
from src.quadrotor.controllers.filter_controller import FilterController
from src.quadrotor.utils import initializeSystemAndController, simulateSafetyFilter
from src.quadrotor.torch.utils import initializeSafetyFilter
from src.utils import generateQuadPoints, findSafetyData, findLearnedQuadSafetyData_gp, downsample, standardize
from utils.print_logger import PrintLogger
from src.quadrotor.handlers import CombinedController
from src.quadrotor.torch.handlers import LearnedQuadSafety_gpy, ExactGPModel
from src.plotting.plotting import plotQuadStatesv2, make_animation, plotQuadTrajectory

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print("Device", device)

def run_qualitative_evaluation(quad, quad_true, flt_est, flt_true, sqp_true, safety_learned, safety_est, 
                        safety_true, x_d, figure_dir):
    # Phase Plane Plotting
    
    freq = 200 # Hz 
    tend = 14
    ts_post_qp = linspace(0, tend, tend*freq + 1)
    
    x_0s_test = np.zeros((2, 6))
    x_0s_test[0, :] = array([0, -1, 0., 0., 0., 0.])
    x_0s_test[1, :] = array([0.5, -0.5, 0., 0., 0., 0.])
    
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

      for z_index, delta in enumerate([0, 0.5, 1.0, 2.0]):
        flt_learned = FilterControllerQCQP( safety_learned, sqp_true, delta=delta)
        
        start_time = time.time()
        qp_data_post = quad_true.simulate(x_0_test, flt_learned, ts_post_qp)
        end_time = time.time()
        print('Average control cycle time: ', (end_time-start_time)/(tend*freq))

        xs_post_qp, _ = qp_data_post   
        
        # Final Result
        plot(xs_post_qp[:, 0], xs_post_qp[:, 1], 'b', linewidth=1.5, label='ProBF-GP  ' + '$\\delta=$' + str(delta), alpha=float((z_index+1))/4.0)

        pickle.dump( xs_post_qp, open( figure_dir + "/learned_pp_run{}_delta{}.pkl".format(str(i), str(delta)) , 'wb') )  
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

  flt_learned = FilterControllerQCQP( safety_learned, sqp_true, delta=1.0)
  
  trueest_violations = 0
  truetrue_violations = 0
  for i in range(num_tests):
    # Learned Controller Simulation
    # Use Learned Controller
    print("Test", i)

    x_0_test = x_0s_test[i,:]
    _, qp_truetrue_data, qp_trueest_data, ts_qp = simulateSafetyFilter(x_0=x_0_test, quad_true=quad_true, quad=quad, flt_true=flt_true, flt_est=flt_est)

    hs_qp_truetrue, _, _, hdots_qp_truetrue = findSafetyData(safety_true, qp_truetrue_data, ts_qp)
    hs_qp_trueest, _, _, hdots_qp_trueest = findSafetyData(safety_est, qp_trueest_data, ts_qp)

    xs_qp_trueest, us_qp_trueest = qp_trueest_data
    xs_qp_truetrue, us_qp_truetrue = qp_truetrue_data

    freq = 200 # Hz
    tend = 14

    ts_post_qp = linspace(0, tend, tend*freq + 1)

    qp_data_post = quad_true.simulate(x_0_test, flt_learned, ts_post_qp)
    xs_post_qp, us_post_qp = qp_data_post

    savename = save_dir+"residual_predict_seed{}_run{}.pdf".format(str(rnd_seed),str(i))
    _, _, hdots_learned_post_qp, hs_post_qp, _ = findLearnedQuadSafetyData_gp(safety_learned, qp_data_post, ts_post_qp)
   
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
    plotQuadStatesv2(axes2, ts_qp, xs_qp_trueest, us_qp_trueest, hs_qp_trueest, hdots_qp_trueest, label='TrueEst', clr='r')
    plotQuadStatesv2(axes2, ts_qp, xs_qp_truetrue, us_qp_truetrue, hs_qp_truetrue, hdots_qp_truetrue, label='TrueTrue', clr='g')
    plotQuadStatesv2(axes2, ts_qp, xs_post_qp, us_post_qp, hs_post_qp, hdots_learned_post_qp, label='ProBF-GP', clr='b')
    fig2.savefig(savename)

    # Trajectory Plotting
    savename = save_dir+"learned_traj_seed{}_run{}.png".format(str(rnd_seed), str(i))
    pickle.dump(xs_post_qp, open(savename[0:-4]+".p", "wb"))
    plotQuadTrajectory(state_data, num_episodes, xs_post_qp=xs_post_qp, xs_qp_trueest=xs_qp_trueest, xs_qp_truetrue=xs_qp_truetrue, 
                       obstacle_position=safety_true.obstacle_position, rad_square=safety_true.obstacle_radius2, x_d=sqp_true.affine_dynamics_position.x_d,
                       savename=savename, title_label='ProBF-GP')

  # record violations
  print("seed: {}, num of violations: {}".format(rnd_seed, str(num_violations)))
  print("Trueest violations", trueest_violations)
  print("Truetrue violations", truetrue_violations)
  return num_violations

def run_quadrotor_gp_training(rnd_seed, num_episodes, num_tests, save_dir, run_quant_evaluation=True, run_qual_evaluation=False):
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
    d_out_seg = 1
    us_scale = array([1.0, 1.0])
    
    # initial points Setup
    x_0s = generateQuadPoints(x_0, num_episodes, ic_prec)

    # initial points for testing
    x_0s_test = generateQuadPoints(x_0, num_tests, ic_prec)

    print('x_0s:', x_0s)
    print('x_0s_test:', x_0s_test)

    safety_learned = LearnedQuadSafety_gpy(safety_est, device=device)

    # Episodic Parameters
    weights = linspace(0, 1, num_episodes)

    # Controller Setup
    flt_baseline = FilterController( safety_est, sqp_true)
    flt_learned = FilterController( safety_est, sqp_true )

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
  
        drift_inputs_long, act_inputs_long, us_long, residuals_long = data

        downsample_rate = 8
        drift_inputs, _, us, residuals = downsample([drift_inputs_long, act_inputs_long, us_long, residuals_long], downsample_rate)
    
        normalized_data, preprocess_mean, preprocess_std = standardize(drift_inputs)
    
        us_scale = array([[1, 1]])
    
        input_data = np.concatenate((us, normalized_data),axis=1)

        #residuals = residuals.ravel() + 0.01*np.random.randn(residuals.size,)

        ndata = input_data.shape[0]
        print("Number of data points: ", ndata)
    
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = 0.01
        input_data_tensor = torch.from_numpy(input_data).float()
        residuals_tensor = torch.from_numpy(residuals.ravel()).float()
    
        residual_model = ExactGPModel(input_data_tensor, residuals_tensor, likelihood)

        if i >=5:
          adam_lr = 0.006
          training_iter = 200
        else:
          adam_lr = 0.009
          training_iter = 200

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
    
        #:
        with gpytorch.settings.max_cg_iterations(10000), gpytorch.settings.cholesky_jitter(1e-4), gpytorch.settings.max_preconditioner_size(15):
          for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = residual_model(input_data_tensor)
            #print(output)
            # Calc loss and backprop gradients
            #gpytorch.settings.max_cg_iterations(10000)
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
    
        # Controller Update
        flt_learned = FilterControllerQCQP( safety_learned, sqp_true, delta=0.0)
    
    num_violations = None
    if run_quant_evaluation:
      figure_quant_dir = save_dir + "quant/" 
      if not os.path.isdir(figure_quant_dir):
        os.mkdir(figure_quant_dir)
      num_violations = run_full_evaluation(rnd_seed, quad=quad, quad_true=quad_true, flt_est=flt_est, flt_true=flt_true, sqp_true=sqp_true, state_data=state_data, 
                                       safety_learned=safety_learned, safety_est=safety_est, safety_true=safety_true,
                                         x_0s_test=x_0s_test, num_tests=num_tests, num_episodes=num_episodes, save_dir=figure_quant_dir)

    if run_qual_evaluation:
      figure_qual_dir = save_dir + "qual/" 
      if not os.path.isdir(figure_qual_dir):
        os.mkdir(figure_qual_dir)
      run_qualitative_evaluation(quad=quad, quad_true=quad_true, flt_est= flt_est, flt_true=flt_true, sqp_true=sqp_true, 
                                 safety_learned=safety_learned, safety_true=safety_true, safety_est=safety_est, x_d=x_d, figure_dir=figure_qual_dir)

    flt_learned = FilterControllerQCQP( safety_learned, sqp_true, delta=0.5)
    return num_violations, flt_learned

def test_quadrotor_cbf(rnd_seed, work_dir, flt_learned=None):
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
    
    safety_est, safety_true, flt_est, flt_true = initializeSafetyFilter(quad, quad_true, sqp_true, obstacle_position=obstacle_position, obstacle_rad2=obstacle_rad2
                                                                        ,cbf_gamma=cbf_gamma, cbf_beta=cbf_beta)
    truetrue_violations = 0
    trueest_violations = 0

    fig1 = plt.figure(figsize=(5, 4))
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

        if(flt_learned is not None):
          qp_learned_data = quad_true.simulate(x_0_test, flt_learned, ts_qp)
          xs_learned, _ = qp_learned_data
          if j==0:
            ax1.plot(xs_learned[:, 0], xs_learned[:, 1], 'b', linewidth=1, label='ProBF-GP')
          else:
            ax1.plot(xs_learned[:, 0], xs_learned[:, 1], 'b', linewidth=1)   

        hs_qp_truetrue, _, _, hdots_qp_truetrue = findSafetyData(safety_true, qp_truetrue_data, ts_qp)
        hs_qp_trueest, _, _, hdots_qp_trueest = findSafetyData(safety_est, qp_trueest_data, ts_qp)

        if np.any(hs_qp_trueest<0):
            trueest_violations += 1
        
        if np.any(hs_qp_truetrue<0):
            truetrue_violations += 1

        if(j==0):
            ax1.plot(xs_qp_nocbf[:, 0], xs_qp_nocbf[:, 1], 'k--', linewidth=1, label='No CBF')
            ax1.plot(xs_qp_truetrue[:, 0], xs_qp_truetrue[:, 1], 'b', linewidth=1, label='True model')
            ax1.plot(xs_qp_trueest[:, 0], xs_qp_trueest[:, 1], 'g', linewidth=1, label='Nominal model')
        else:
            ax1.plot(xs_qp_nocbf[:, 0], xs_qp_nocbf[:, 1], 'k--', linewidth=1)
            ax1.plot(xs_qp_truetrue[:, 0], xs_qp_truetrue[:, 1], 'b', linewidth=1)
            ax1.plot(xs_qp_trueest[:, 0], xs_qp_trueest[:, 1], 'g', linewidth=1)

        fig2, axes2 = plt.subplots(2, 3, figsize=(13,8))
        plotQuadStatesv2(axes2, ts_qp, xs_qp_trueest, us_qp_trueest, hs_qp_trueest, hdots_qp_trueest, label='Nominal model', clr='g')
        plotQuadStatesv2(axes2, ts_qp, xs_qp_truetrue, us_qp_truetrue, hs_qp_truetrue, hdots_qp_truetrue, label='True model', clr='b')
        fig2.savefig(os.path.join(work_dir, str(rnd_seed) + '_' + 'run' + str(j) + 'quadrotor_states.png'))
        plt.close()

        if(j==0):
            make_animation(xs_qp_truetrue, x_d, obstacle_position, obstacle_rad2, fig_folder=os.path.join(work_dir,'animation/'))

    circle = Circle((obstacle_position[0], obstacle_position[1]), np.sqrt(obstacle_rad2), color="y")
    ax1.add_patch(circle)
    ax1.plot(x_d[0, :], x_d[1, :], 'k*', label='Desired')
    ax1.set_xticks([-2.0, obstacle_position[0], x_d[0, 0], 13.0])
    ax1.set_yticks([-2.0, obstacle_position[1], x_d[1, 0], 13.0])
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
  #rnd_seed_list = [345, 123, 678, 567]
  rnd_seed_list = [ 123, 234, 345, 456, 678 ]
  #rnd_seed_list = [123]

  # Episodic Learning Setup
  experiment_name = "check_probfGP"

  parent_path = "/scratch/gpfs/arkumar/ProBF/"
  parent_path = os.path.join(parent_path, experiment_name)
  
  baseline_dir = os.path.join(parent_path, "baseline")
  if not os.path.isdir(parent_path):
    os.mkdir(parent_path)
    os.mkdir(baseline_dir)
    os.mkdir( os.path.join(parent_path, "exps") )
    os.mkdir( os.path.join(parent_path, "models") )
 
  
  figure_path = os.path.join(parent_path, "exps/quad_modular_gp/")
  model_path = os.path.join(parent_path, "models/quad_modular_gp/")

  if not os.path.isdir(figure_path):
    os.mkdir(figure_path)

  if not os.path.isdir(model_path):
    os.mkdir(model_path)

  #test_quadrotor_cbf(56, baseline_dir)
  
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

    num_violations, flt_learned = run_quadrotor_gp_training(rnd_seed, num_episodes, num_tests, dirs, run_quant_evaluation=True, run_qual_evaluation=False)
    num_violations_list.append(num_violations)

  print_logger.reset(os.path.join(figure_path, 'log.txt'))
  print_logger.reset(os.path.join(figure_path, 'log.txt')) 
  #test_quadrotor_cbf(479, baseline_dir, flt_learned)
  print("num_violations_list: ", num_violations_list)
  #test_quadrotor_cbf(56, baseline_dir, flt_learned)