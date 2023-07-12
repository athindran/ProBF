import os
import time
import numpy as np
from numpy import array, linspace, ones, size, sqrt, zeros
from numpy.random import seed

import torch
import gpytorch

from core.controllers import Controller, FilterController, FilterControllerVar

from utils.SegwaySupport import initializeSystem, initializeSafetyFilter, simulateSafetyFilter, LearnedSegwaySafetyAAR
from utils.AuxFunc import findSafetyData, findLearnedSafetyData_gp, downsample, standardize, generateInitialPoints
from utils.Plotting import plotTestStates, plotPhasePlane, plotLearnedCBF, plotPredictions

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print("Device", device)

class ExactGPModel(gpytorch.models.ExactGP):
    """GPytorch model with explicit modeling of kernel"""
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
            
        # Define kernels and covariance function of GP
        active_dimsu = np.array([0])
        ku = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel(active_dims=active_dimsu))

        active_dimsv = np.array([1, 2, 3, 4, 6, 8])
        ka = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(active_dims=active_dimsv, ard_num_dims=6))
        self.k1 = ku*ka

        kb = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(active_dims=active_dimsv, ard_num_dims=6))
        self.k2 = kb
        self.covar_module = self.k1 + self.k2

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Combined Controller
class CombinedController(Controller):
    """Controller combination"""
    def __init__(self, controller_1, controller_2, weights):
        self.controller_1 = controller_1
        self.controller_2 = controller_2
        self.weights = weights
        
    def eval(self, x, t):
        u_1 = self.controller_1.process( self.controller_1.eval( x, t ) )
        u_2 = self.controller_2.process( self.controller_2.eval( x, t ) )
        return self.weights[ 0 ] * u_1 + self.weights[ 1 ] * u_2

def evaluateTrainedModel(seg_est, seg_true, flt_est, flt_true, pd, state_data, safety_learned, safety_est, safety_true, comparison_safety, x_0s_test, num_tests=10, sigma=0.0, figure_dir="./"):                       
  """
    Evaluate trained model and plot various comparisons
  """
  num_violations = 0

  # QP simulation comapre trues and estimate for plots
  _, qp_truetrue_data, qp_trueest_data, ts_qp = simulateSafetyFilter(seg_true, seg_est, flt_true, flt_est)
  #hs_qp_estest, drifts_qp_estest, acts_qp_estest, hdots_qp_estest = findSafetyData(safety_est, qp_estest_data, ts_qp)
  hs_qp_truetrue, _, _, _ = findSafetyData(safety_true, qp_truetrue_data, ts_qp)
  hs_qp_trueest, _, _, _ = findSafetyData(safety_true, qp_trueest_data, ts_qp)

  #xs_qp_estest, us_qp_estest = qp_estest_data
  xs_qp_trueest, us_qp_trueest = qp_trueest_data
  xs_qp_truetrue, us_qp_truetrue = qp_truetrue_data

  phi_0_learned = lambda x, t: safety_learned.drift_learned( x, t )   
  phi_1_learned = lambda x, t: safety_learned.act_learned( x, t )
  flt_learned = FilterControllerVar( seg_est, phi_0_learned, phi_1_learned, pd, sigma)

  freq = 500 # Hz 
  tend = 3
  ts_post_qp = linspace(0, tend, tend*freq + 1)

  # Test for 10 different random points  
  for i in range(num_tests):
    # Learned Controller Simulation
    # Use Learned Controller
    x_0 = x_0s_test[i,:]
    qp_data_post = seg_true.simulate(x_0, flt_learned, ts_post_qp)
    xs_post_qp, us_post_qp = qp_data_post

    data_episode = safety_learned.process_episode(xs_post_qp, us_post_qp, ts_post_qp)
    
    # Plot of residual predictions from GP
    savename = figure_dir + "/residual_predict_seed{}_test{}.png".format(str(rnd_seed), str(i))
    plotPredictions(safety_learned, data_episode, savename, device=device)

    drifts_learned_post_qp, acts_learned_post_qp, hdots_learned_post_qp, hs_post_qp, _ = findLearnedSafetyData_gp(safety_learned, qp_data_post, ts_post_qp)
   
    # check violation of safety
    if np.any(hs_post_qp < 0):
      num_violations += 1
    
    _, drifts_post_qp, acts_post_qp, hdots_post_qp = findSafetyData(safety_est, qp_data_post, ts_post_qp)
    _, drifts_true_post_qp, acts_true_post_qp, hdots_true_post_qp = findSafetyData(safety_true, qp_data_post, ts_post_qp)
    
    theta_bound_u = ( safety_true.theta_e + safety_true.angle_max ) * ones( size( ts_post_qp ) )
    theta_bound_l = ( safety_true.theta_e - safety_true.angle_max ) * ones( size( ts_post_qp ) )

    # Plotting
    savename = figure_dir+"/learned_filter_seed{}_run{}_sigma{}.png".format(str(rnd_seed), str(i), str(sigma))
    plotTestStates(ts_qp, ts_post_qp, xs_qp_trueest, xs_qp_truetrue, xs_post_qp, us_qp_trueest, us_qp_truetrue, 
                    us_post_qp, hs_qp_trueest, hs_qp_truetrue, hs_post_qp, hdots_post_qp, hdots_true_post_qp, hdots_learned_post_qp , 
                        drifts_post_qp, drifts_true_post_qp, drifts_learned_post_qp, acts_post_qp, acts_true_post_qp, acts_learned_post_qp, 
                            theta_bound_u, theta_bound_l, savename)

    # h plotting
    ebs = int(len(state_data[0])/num_episodes)

    xs_1 = state_data[0][0*ebs:1*ebs]
    xs_2 = state_data[0][1*ebs:2*ebs]
    xs_3 = state_data[0][2*ebs:3*ebs]
    xs_4 = state_data[0][3*ebs:4*ebs]
    xs_5 = state_data[0][4*ebs:5*ebs]

    hs_1 = array([safety_learned.eval(x,t) for x, t in zip(xs_1, ts_post_qp)])
    hs_2 = array([safety_learned.eval(x,t) for x, t in zip(xs_2, ts_post_qp)])
    hs_3 = array([safety_learned.eval(x,t) for x, t in zip(xs_3, ts_post_qp)])
    hs_4 = array([safety_learned.eval(x,t) for x, t in zip(xs_4, ts_post_qp)])
    hs_5 = array([safety_learned.eval(x,t) for x, t in zip(xs_5, ts_post_qp)])

    hs_all = np.hstack((hs_1,hs_2,hs_3,hs_4,hs_5))

    # Learned CBF safety filter
    savename = figure_dir + "/learned_h_seed{}_run{}_sigma{}.png".format(str(rnd_seed), str(i), str(sigma))
    plotLearnedCBF(ts_qp, hs_qp_trueest, hs_all, ts_post_qp, hs_post_qp, ebs, num_episodes, savename)
    
    # Phase Plane Plotting
    epsilon = 1e-6
    theta_h0_vals = linspace(safety_true.theta_e-safety_true.angle_max+epsilon, safety_true.theta_e + safety_true.angle_max - epsilon, 1000)
    theta_dot_h0_vals = array([sqrt((safety_true.angle_max ** 2 - (theta - safety_true.theta_e) ** 2) /safety_true.coeff) for theta in theta_h0_vals])
    ebs = int(len(state_data[0])/num_episodes)
    
    savename = figure_dir + "/learned_pp_seed{}_run{}_sigma{}.png".format(str(rnd_seed), str(i), str(sigma))
    plotPhasePlane(theta_h0_vals, theta_dot_h0_vals, xs_qp_trueest, state_data, xs_post_qp, ebs, num_episodes, savename)

  # record violations
  print("seed: {}, num of violations: {}".format(rnd_seed, str(num_violations)))
  return num_violations
    
def run_experiment(rnd_seed, num_episodes, num_tests, figure_dir, model_dir):
  from core.controllers import FilterControllerVar
  seed(rnd_seed)
  torch.manual_seed(rnd_seed)

  # Estimated and true segway dynamics, PD controller
  seg_est, seg_true, _, _, pd = initializeSystem()
  safety_est, safety_true, flt_est, flt_true = initializeSafetyFilter(seg_est, seg_true, pd)
  
  alpha = 10
  comparison_safety = lambda r: alpha * r
  safety_learned = LearnedSegwaySafetyAAR(safety_est, device=device)
  #--------------------- Set Episodic Parameters -------------------#
  weights = linspace(0, 1, num_episodes)
  # change controller to pure learned controller

  # Initialize learned safety filter with no GP  
  phi_0 = lambda x, t: safety_learned.drift_estimate( x, t ) + comparison_safety( safety_learned.eval( x, t ) )
  phi_1 = lambda x, t: safety_learned.act_estimate( x, t )
  flt_baseline = flt_est
  flt_learned = FilterController( seg_est, phi_0, phi_1, pd )

  # Data Storage Setup
  d_drift_in_seg = 8
  d_act_in_seg = 8
  d_out_seg = 1
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

  print('Initial states for training:', x_0s)
  print('Initial states for testing:', x_0s_test)

  ustd_list = []
  input_data_list = []
  residual_true_list = []
  residual_pred_list = []
  residual_pred_lower_list = []
  residual_pred_upper_list = []

  # Iterate through each episode
  for i in range(num_episodes):
      print("Episode:", i+1)
      # Controller Combination
      flt_combined = CombinedController( flt_baseline, flt_learned, array([1 - weights[i], weights[i]]) )
      x_0 = x_0s[i,:]
      print("Initial state in Episode: ", x_0)
      start_time = time.time()
      sim_data = seg_true.simulate(x_0, flt_combined, ts_qp)
      end_time = time.time()
      print("Finished simulation with average control cycle time (s): ", (end_time - start_time)/(tend*freq))
      
      # Data Handling
      xso, uso = sim_data
      data_episode = safety_learned.process_episode(xso, uso, ts_qp)
      data_episode = data_episode[0:-1]
      # Concatenate logs from multiple episodes for GP fitting
      state_data = [np.concatenate((old, new)) for old, new in zip(state_data, [xso])]
      data = [np.concatenate((old, new)) for old, new in zip(data, data_episode)]

      drift_inputs, act_inputs, usc, residualsc = data
      
      # FInd the predictions after each round and compare with true residuals
      if i > 0:
          #predictions on current episode data
          # first normalize with previous round mean and std
          normalized_data_test = (drift_inputs - preprocess_mean)/preprocess_std
          input_data_test = np.concatenate(((usc - usmean)/usstd, normalized_data_test), axis=1)
          
          #store input data under previous round transformation
          input_data_list.append( input_data_test )
          input_data_test_tensor = torch.from_numpy(input_data_test).float().to(device)
          # Get into evaluation (predictive posterior) mode
          residual_model.eval()
          likelihood.eval()
          #gpytorch.settings.fast_pred_var()
          
          with torch.no_grad(), gpytorch.settings.fast_computations(solves=False):
              respred_test = likelihood( safety_learned.residual_model( input_data_test_tensor ) )

          lower, upper = respred_test.confidence_region()

          if device!="cpu":
            lower = lower.cpu().detach().numpy()
            upper = upper.cpu().detach().numpy()
            mean = respred_test.mean.cpu().detach().numpy()
            #var = respred_test.variance.cpu().detach().numpy()
          else:
            lower = lower.detach().numpy()
            upper = upper.detach().numpy()
            mean = respred_test.mean.detach().numpy()
            #var = respred_test.variance.detach().numpy()
          
          residual_true_list.append( residualsc )
          # this is from a*u + b
          #residual_pred_compare_list.append( respredsc )
          
          # from GP prediction directly
          residual_pred_list.append( mean )
          residual_pred_lower_list.append( lower )
          residual_pred_upper_list.append( upper )
      
      # Prepare data for GP fitting
      downsample_rate = 5
      drift_inputs, act_inputs, us, residuals = downsample([drift_inputs, act_inputs, usc, residualsc], downsample_rate)
      
      # Normalization on input and u
      normalized_data, preprocess_mean, preprocess_std = standardize(drift_inputs)
      usstd = np.std(us)
      # NO shift in u
      usmean = 0
      input_data = np.concatenate(((us - usmean)/usstd, normalized_data), axis=1)    

      ndata = input_data.shape[0]
      print("Number of data points: ", ndata)

      #residuals = residuals + 0.01*np.random.randn(residuals.size, )
      if i > 0:
          ustd_list.append(usstd)

      likelihood = gpytorch.likelihoods.GaussianLikelihood()
      likelihood.noise = 0.01
      input_data_tensor = torch.from_numpy(input_data).float().to(device)
      residuals_tensor = torch.from_numpy(residuals.ravel()).float().to(device)
      
      residual_model = ExactGPModel(input_data_tensor, residuals_tensor, likelihood)

      if i == 0:
          # save random initialization model      
          torch.save(residual_model.state_dict(), model_dir + "residual_model_iter_{}.pth".format(str(0)))
          adam_lr = 0.03
          training_iter = 200
      elif i >= 10:
          state_dict = torch.load(model_dir + "residual_model_iter_{}.pth".format(str(0)))
          residual_model.load_state_dict(state_dict)
          adam_lr = 0.04
          training_iter = 0
      else:
          #load previous episode trained model
          #state_dict = torch.load("residual_model_iter_{}.pth".format(str(i)))
          # load random initialization
          state_dict = torch.load(model_dir + "residual_model_iter_{}.pth".format(str(0)))
          residual_model.load_state_dict(state_dict)
          adam_lr = 0.01
          training_iter = 300
    
      # load to gpu if possible
      if device!='cpu':
        input_data_tensor = input_data_tensor.to( device )
        residuals_tensor = residuals_tensor.to( device )
        residual_model.k1 = residual_model.k1.to( device )
        residual_model.k2 = residual_model.k2.to( device )
        residual_model = residual_model.to( device )
        likelihood = likelihood.to( device )

      # Find optimal model hyperparameters
      residual_model.train()
      likelihood.train()
          
      # Use the adam optimizer
      optimizer = torch.optim.Adam([
          {'params': residual_model.parameters()},  # Includes GaussianLikelihood parameters
          ], lr=adam_lr)

      # "Loss" for GPs - the marginal log likelihood
      mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, residual_model)


      # print hyperparams before training
      print("kernel lengthscale for a(x)",residual_model.covar_module.kernels[0].kernels[1].base_kernel.lengthscale)
      print("kernel scale for a(x)",residual_model.covar_module.kernels[0].kernels[1].outputscale.item())
      print("kernel scale for u",residual_model.covar_module.kernels[0].kernels[0].outputscale.item())
      print("kernel lengthscale for b(x)",residual_model.covar_module.kernels[1].base_kernel.lengthscale)
      print("kernel scale for b(x)",residual_model.covar_module.kernels[1].outputscale.item())
      
      with gpytorch.settings.max_cg_iterations(3000), gpytorch.settings.cholesky_jitter(1e-4):
          for j in range(training_iter):
              # Zero gradients from previous iteration
              optimizer.zero_grad()
              # Output from model
              output = residual_model( input_data_tensor )
              # Calc loss and backprop gradients
              gpytorch.settings.max_cg_iterations(3000)
              loss = -mll(output, residuals_tensor)
              loss.backward()
              if(j%5==0):
                  print("Loss",loss)      
                  print('Iter %d/%d - Loss: %.3f   noise: %.3f' % (
                      j + 1, training_iter, loss.item(),
                      residual_model.likelihood.noise.item()
                  ))
              optimizer.step()
      
      # print hyperparams after training
      print("kernel lengthscale for a(x)", residual_model.covar_module.kernels[0].kernels[1].base_kernel.lengthscale)
      print("kernel scale for a(x)", residual_model.covar_module.kernels[0].kernels[1].outputscale.item())
      print("kernel scale for u", residual_model.covar_module.kernels[0].kernels[0].outputscale.item())
      print("kernel lengthscale for b(x)", residual_model.covar_module.kernels[1].base_kernel.lengthscale)
      print("kernel scale for b(x)", residual_model.covar_module.kernels[1].outputscale.item())
      # save the current gp model with hyperparams
      torch.save(residual_model.state_dict(), model_dir + "residual_model_iter_{}.pth".format(str(i+1)))

      safety_learned = LearnedSegwaySafetyAAR( safety_est, device=device)
      
      safety_learned.residual_model = residual_model
      safety_learned.likelihood = likelihood
      safety_learned.usstd = usstd

      # Evaluate covariance matrix with the data  
      safety_learned.Kinv = torch.pinverse( residual_model.covar_module( input_data_tensor ).evaluate() + residual_model.likelihood.noise.item()*torch.eye( input_data_tensor.shape[0] ).to(device) )  
      safety_learned.alpha = torch.matmul(safety_learned.Kinv, torch.from_numpy(residuals).float().to(device) )
      safety_learned.input_data_tensor = input_data_tensor
      safety_learned.preprocess_mean = torch.from_numpy( preprocess_mean[0] )
      safety_learned.preprocess_std = torch.from_numpy( preprocess_std[0] )
      safety_learned.comparison_safety = comparison_safety
      
      # Controller Update
      phi_0_learned = safety_learned.drift_learned 
      phi_1_learned = safety_learned.act_learned
      flt_learned = FilterControllerVar( seg_est, phi_0_learned, phi_1_learned, pd, sigma=0.0)

  print(residual_model.covar_module.kernels[0].kernels[1].outputscale)
  print(residual_model.covar_module.kernels[1].outputscale)
  print(residual_model.covar_module.kernels[0].kernels[1].base_kernel.lengthscale)
  print(residual_model.covar_module.kernels[1].base_kernel.lengthscale)

  num_violations_a = evaluateTrainedModel(seg_est, seg_true, flt_est, flt_true, pd, state_data, safety_learned, safety_est, safety_true, comparison_safety, x_0s_test, num_tests,0, figure_dir)
  print("viol-0: ", num_violations_a)
  num_violations_b = evaluateTrainedModel(seg_est, seg_true, flt_est, flt_true, pd, state_data, safety_learned, safety_est, safety_true, comparison_safety, x_0s_test, num_tests,0.5, figure_dir)
  print("viol-0.5: ", num_violations_b)
  num_violations_c = evaluateTrainedModel(seg_est, seg_true, flt_est, flt_true, pd, state_data, safety_learned, safety_est, safety_true, comparison_safety, x_0s_test, num_tests,1.0, figure_dir)
  print("viol-1.0: ", num_violations_c)

  return num_violations_c


#rnd_seed_list = [123]
rnd_seed_list = [ 123, 234, 345, 456, 567, 678, 789, 890, 901, 12]
#rnd_seed_list = [345]
# Episodic Learning Setup
num_violations_list = []
num_episodes = 5
num_tests = 10

parent_path = "/scratch/gpfs/arkumar/ProBF/experiments/segway_modular_gp/"
model_path = "/scratch/gpfs/arkumar/ProBF/model/segway_modular_gp/"

if not os.path.isdir(parent_path):
    os.mkdir(parent_path)

if not os.path.isdir(model_path):
    os.mkdir(model_path)

for rnd_seed in rnd_seed_list:
  parent_dirs = parent_path + str(rnd_seed) + "/"
  model_dirs = model_path + str(rnd_seed) + "/"
  if not os.path.isdir(parent_dirs):
      os.mkdir(parent_dirs)

  if not os.path.isdir(model_dirs):    
      os.mkdir(model_dirs)
  num_violations_c = run_experiment(rnd_seed, num_episodes, num_tests, parent_dirs, model_dirs)
  num_violations_list.append(num_violations_c)

print("num_violations_list: ", num_violations_list)
