from core.dynamics import AffineDynamics, ConfigurationDynamics, LearnedDynamics, PDDynamics, ScalarDynamics
from core.systems import Segway
from core.controllers import Controller, FBLinController, LQRController, FilterController, PDController, QPController, FilterControllerVar
from core.util import differentiate
from matplotlib.pyplot import figure, legend, plot, xlabel, ylabel, fill_between
import numpy as np
from numpy import array, concatenate, dot, identity, linspace, ones, savetxt, size, sqrt, zeros
from numpy.random import uniform, seed
from numpy.random import permutation
from numpy import clip
import os
import torch
import gpytorch
import time
from core.dynamics import LearnedAffineDynamics

from utils.SegwaySupport import initializeSystem, initializeSafetyFilter, simulateSafetyFilter, SafetyAngleAngleRate
from utils.AuxFunc import findSafetyData, findLearnedSafetyData_gp, postProcessEpisode, downsample, standardize, generateInitialPoints
from utils.Plotting import plotTestStates, plotTrainStates, plotTrainMetaData, plotPhasePlane, plotLearnedCBF

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
print(device)    # %%

def plotPredictions(safety_learned, data_episode, savename):
    """
    Plots the comparison of residual predictions and actual residuals.

    Inputs:
        safety_learned: Learned safety filter
        data_episode: Trajectory data from each episode
    """
    residual_model = safety_learned.residual_model
    likelihood = safety_learned.likelihood
    
    drift_inputs, _, us, residuals, _, _, _, _, _, _  = data_episode
    npoints = drift_inputs.shape[0]
    test_inputs = (torch.from_numpy( drift_inputs ) - torch.reshape(safety_learned.preprocess_mean, (-1, 8)).repeat(npoints, 1) )
    test_inputs = torch.divide(test_inputs, torch.reshape(safety_learned.preprocess_std, (-1, 8)).repeat(npoints, 1) )
    test_inputs = torch.cat((torch.from_numpy(us/safety_learned.usstd), test_inputs), axis=1)
    test_inputs = test_inputs.float()

    residual_model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_computations(solves=False):
        residual_pred = likelihood(safety_learned.residual_model(test_inputs))
    lower, upper = residual_pred.confidence_region()
    lower = lower.cpu()
    upper = upper.cpu()
    mean = residual_pred.mean.detach().cpu().numpy()
    var = residual_pred.variance.detach().cpu().numpy()
    f = figure()
    plot(mean*safety_learned.residual_std + safety_learned.residual_mean)
    fill_between(np.arange(mean.size), lower.detach().numpy(), upper.detach().numpy(), color='blue', alpha=0.2)
    plot(residuals)
    xlabel('Time')
    ylabel('CBF residuals')
    legend(["Prediction", "Actual"])
    f.savefig(savename, bbox_inches='tight')
    
# Learned Segway Angle-Angle Rate Safety
class LearnedSegwaySafetyAAR(LearnedAffineDynamics):
    """
    Learned Segway Angle-Angle Rate Safety
        Interface to use GP for residual dynamics
    """
    def __init__(self, segway_est):
        """
          Initialize with estimate of segway dynamics
        """
        self.dynamics = segway_est
        self.residual_model = None
        self.input_data = []
        self.preprocess_mean = torch.zeros((8,))
        self.preprocess_std = 1
        self.residual_std = 1
        self.residual_mean = 0
        self.usstd = 1
              
    def process_drift(self, x, t):
        dhdx = self.dynamics.dhdx( x, t )
        return concatenate([x, dhdx])

    def process_act(self, x, t):
        dhdx = self.dynamics.dhdx( x, t )
        return concatenate([x, dhdx])

    def process_drift_torch(self, x_torch, t):
        dhdx_torch = self.dynamics.dhdx_torch( x_torch, t )
        return torch.cat([x_torch, dhdx_torch])

    def process_act_torch(self, x_torch, t):
        dhdx_torch = self.dynamics.dhdx_torch( x_torch, t )
        return concatenate([x_torch, dhdx_torch])
    
    def eval(self, x, t):
        return self.dynamics.eval(x, t)
    
    def drift_estimate(self, x, t):
        return self.dynamics.drift(x, t)

    def act_estimate(self, x, t):
        return self.dynamics.act(x, t)

    def drift_learned(self, x, t):
        """
          Find mean and variance of control-independent dynamics b after residual modeling.
        """
        xtorch = torch.from_numpy( x )
        xfull = torch.cat((torch.Tensor([1.0]), torch.divide(self.process_drift_torch(xtorch, t) - self.preprocess_mean, self.preprocess_std)))
        xfull = torch.reshape(xfull, (-1, 9)).float()

        cross1 = self.residual_model.k1(xfull, self.input_data_tensor)
        cross1 = cross1.evaluate()
        cross1 = cross1*(1/self.usstd)
        cross2 = self.residual_model.k2(xfull, self.input_data_tensor)
        cross2 = cross2.evaluate().float()
        bmean = torch.matmul(cross2, self.alpha)
        
        mean = bmean*self.residual_std
        variance = (self.residual_model.k2(xfull, xfull)).evaluate() - torch.matmul( torch.matmul(cross2, self.Kinv), cross2.T )
        varab = -torch.matmul( torch.matmul(cross1, self.Kinv), cross2.T)
        return [self.dynamics.drift(x, t) + mean.detach().numpy().ravel() + self.residual_mean + self.comparison_safety(self.eval(x, t)), variance.detach().numpy().ravel(), varab.detach().numpy().ravel()]
    
    def act_learned(self, x, t):
        """
          Find mean and variance of control-dependent dynamics a after residual modeling.
        """
        xtorch = torch.from_numpy( x )
        xfull = torch.cat((torch.Tensor([1.0]), torch.divide(self.process_drift_torch(xtorch, t) - self.preprocess_mean, self.preprocess_std)))
        xfull = torch.reshape(xfull, (-1, 9)).float()

        cross = self.residual_model.k1(xfull, self.input_data_tensor).evaluate()/self.usstd
        mean = torch.matmul(cross, self.alpha)
        variancequad = self.residual_model.k1(xfull, xfull).evaluate()/(self.usstd)**2 - torch.matmul( torch.matmul(cross, self.Kinv), cross.T)
        return self.dynamics.act(x, t) + mean.detach().numpy().ravel(), variancequad.detach().numpy().ravel()
    
    def process_episode(self, xs, us, ts, window=3):
        """
            Data pre-processing step to generate plots
        """
        #------------------------- truncating data -----------------------#
        # for tstart in range(len(us)):
        #    if np.all(np.abs(np.array(us[tstart:]))<6e-3):
        #      break
        
        tend = len(us)
        endpoint = tend

        half_window = (window - 1) // 2
        xs = xs[:len(us)]
        ts = ts[:len(us)]
        
        drift_inputs = array([self.process_drift(x, t) for x, t in zip(xs, ts)])
        act_inputs = array([self.process_act(x, t) for x, t in zip(xs, ts)])

        reps = array([self.dynamics.eval(x, t) for x, t in zip(xs, ts)])
        rep_dots = differentiate(reps, ts)
        rep_dot_noms = array([self.dynamics.eval_dot(x, u, t) for x, u, t in zip(xs, us, ts)])
        
        apreds = zeros(ts.size, )
        bpreds = zeros(ts.size, )
        apredsvar = zeros(ts.size, )
        bpredsvar = zeros(ts.size, )
        respreds = zeros(ts.size, )

        j = 0
        if self.residual_model is not None:
          for x,u,t in zip(xs,us,ts):
            meanb, varb, _ = self.drift_learned(x,t)
            meana, vara = self.act_learned(x,t)
            apreds[j] = meana - self.dynamics.act(x, t)
            bpreds[j] = meanb - self.comparison_safety( self.eval(x,t) ) - self.dynamics.drift(x, t)
            apredsvar[j] = vara
            bpredsvar[j] = varb
            respreds[j] = apreds[j]*u + bpreds[j]
            j = j+1
        
        drift_inputs = drift_inputs[half_window:-half_window]
        act_inputs = act_inputs[half_window:-half_window]
        rep_dot_noms = rep_dot_noms[half_window:-half_window]
        
        apreds = apreds[half_window:-half_window]
        apredsvar = apredsvar[half_window:-half_window]
        bpreds = bpreds[half_window:-half_window]
        respreds = respreds[half_window:-half_window]
        bpredsvar = bpredsvar[half_window:-half_window]
        us = us[0:-2*half_window]
        
        apreds = apreds[0:endpoint]
        apredsvar = apredsvar[0:endpoint]
        bpreds = bpreds[0:endpoint]
        bpredsvar = bpredsvar[0:endpoint]
        respreds = respreds[0:endpoint]
        us = us[0:endpoint]
        
        drift_inputs = drift_inputs[0:endpoint]
        act_inputs = act_inputs[0:endpoint]
        rep_dot_noms = rep_dot_noms[0:endpoint]
        rep_dots = rep_dots[0:endpoint]
        
        residuals = rep_dots - rep_dot_noms
        
        return drift_inputs, act_inputs, us, residuals, apreds, bpreds, apredsvar, bpredsvar, respreds, endpoint
    
    def actvar(self, x, t):
        xfull = np.concatenate(([1],np.divide(self.process_act(x, t)-self.preprocess_mean,self.preprocess_std)))
        cross = self.k1(xfull, self.input_data)
        mean = np.dot(cross, self.alpha)*self.residual_std
        variancequad = self.k1(xfull, xfull) - np.dot( np.dot( cross, self.Kinv ), cross)
        variancequad = variancequad.cpu().numpy()
        sigma1 = 0
        return [self.dynamics.act(x, t) + mean, -sigma1*np.sqrt(variancequad)]
    
    def init_data(self, d_drift_in, d_act_in, m, d_out):
        return [zeros((0, d_drift_in)), zeros((0, d_act_in)), zeros((0, m)), zeros(0), zeros(0), zeros(0), zeros(0), zeros(0), zeros(0)]
    

class ExactGPModel(gpytorch.models.ExactGP):
    """GPytorch model with explicit modeling of kernel"""
    def __init__(self, train_x, train_y, likelihood, k1, k2, covar_module):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.k1 = k1
        self.k2 = k2
        self.covar_module = covar_module

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

def evaluateTrainedModel(seg_est, seg_true, flt_est, flt_true, pd, state_data, safety_learned, safety_est, safety_true, comparison_safety, x_0s_test, num_tests=10, sigma=0.0, save_dir="./"):                       
  """
    Evaluate trained model and plot various comparisons
  """
  num_violations = 0

  # QP simulation comapre trues and estimate for plots
  qp_estest_data, qp_truetrue_data, qp_trueest_data, ts_qp = simulateSafetyFilter(seg_true, seg_est, flt_true, flt_est)
  hs_qp_estest, drifts_qp_estest, acts_qp_estest, hdots_qp_estest = findSafetyData(safety_est, qp_estest_data, ts_qp)
  hs_qp_truetrue, drifts_qp_truetrue, acts_qp_truetrue, hdots_qp_truetrue = findSafetyData(safety_true, qp_truetrue_data, ts_qp)
  hs_qp_trueest, drifts_qp_trueest, acts_qp_trueest, hdots_qp_trueest = findSafetyData(safety_true, qp_trueest_data, ts_qp)

  xs_qp_estest, us_qp_estest = qp_estest_data
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
    savename = save_dir + "/residual_predict_seed{}_test{}.png".format(str(rnd_seed), str(i))
    plotPredictions(safety_learned, data_episode, savename)

    drifts_learned_post_qp, acts_learned_post_qp, hdots_learned_post_qp, hs_post_qp, hdots_post_num = findLearnedSafetyData_gp(safety_learned, qp_data_post, ts_post_qp)
   
    # check violation of safety
    if np.any(hs_post_qp < 0):
      num_violations += 1
    
    _, drifts_post_qp, acts_post_qp, hdots_post_qp = findSafetyData(safety_est, qp_data_post, ts_post_qp)
    _, drifts_true_post_qp, acts_true_post_qp, hdots_true_post_qp = findSafetyData(safety_true, qp_data_post, ts_post_qp)
    
    theta_bound_u = ( safety_true.theta_e + safety_true.angle_max ) * ones( size( ts_post_qp ) )
    theta_bound_l = ( safety_true.theta_e - safety_true.angle_max ) * ones( size( ts_post_qp ) )

    # Plotting
    savename = save_dir+"/learned_filter_seed{}_run{}_sigma{}.png".format(str(rnd_seed), str(i), str(sigma))
    plotTestStates(ts_qp, ts_post_qp, xs_qp_trueest, xs_qp_truetrue, xs_post_qp, us_qp_trueest, us_qp_truetrue, 
                    us_post_qp, hs_qp_trueest, hs_qp_truetrue, hs_post_qp, hdots_post_qp, hdots_true_post_qp, hdots_learned_post_qp , 
                        drifts_post_qp, drifts_true_post_qp, drifts_learned_post_qp, acts_post_qp, acts_true_post_qp, acts_learned_post_qp, 
                            theta_bound_u, theta_bound_l, savename)

    # h plotting
    ebs = int(len(state_data[0])/num_episodes)

    xs_baseline = xs_qp_trueest
    xs_1 = state_data[0][0*ebs:1*ebs]
    xs_2 = state_data[0][1*ebs:2*ebs]
    xs_3 = state_data[0][2*ebs:3*ebs]
    xs_4 = state_data[0][3*ebs:4*ebs]
    xs_5 = state_data[0][4*ebs:5*ebs]
    xs_learned = xs_post_qp

    hs_1 = array([safety_learned.eval(x,t) for x, t in zip(xs_1, ts_post_qp)])
    hs_2 = array([safety_learned.eval(x,t) for x, t in zip(xs_2, ts_post_qp)])
    hs_3 = array([safety_learned.eval(x,t) for x, t in zip(xs_3, ts_post_qp)])
    hs_4 = array([safety_learned.eval(x,t) for x, t in zip(xs_4, ts_post_qp)])
    hs_5 = array([safety_learned.eval(x,t) for x, t in zip(xs_5, ts_post_qp)])

    hs_all = np.hstack((hs_1,hs_2,hs_3,hs_4,hs_5))

    # Learned CBF safety filter
    savename = save_dir+"/learned_h_seed{}_run{}_sigma{}.png".format(str(rnd_seed), str(i), str(sigma))
    plotLearnedCBF(ts_qp, hs_qp_trueest, hs_all, ts_post_qp, hs_post_qp, ebs, num_episodes, savename)
    
    # Phase Plane Plotting
    epsilon = 1e-6
    theta_h0_vals = linspace(safety_true.theta_e-safety_true.angle_max+epsilon, safety_true.theta_e + safety_true.angle_max - epsilon, 1000)
    theta_dot_h0_vals = array([sqrt((safety_true.angle_max ** 2 - (theta - safety_true.theta_e) ** 2) /safety_true.coeff) for theta in theta_h0_vals])
    ebs = int(len(state_data[0])/num_episodes)
    
    savename = save_dir+"/learned_pp_seed{}_run{}_sigma{}.png".format(str(rnd_seed), str(i), str(sigma))
    plotPhasePlane(theta_h0_vals, theta_dot_h0_vals, xs_qp_trueest, state_data, xs_post_qp, ebs, num_episodes, savename)

  # record violations
  print("seed: {}, num of violations: {}".format(rnd_seed, str(num_violations)))
  return num_violations
    
def run_experiment(rnd_seed, num_episodes, num_tests, save_dir):
  from core.controllers import FilterControllerVar
  seed(rnd_seed)
  torch.manual_seed(rnd_seed)

  # Estimated and true segway dynamics, PD controller
  seg_est, seg_true, sego_est, sego_true, pd = initializeSystem()
  safety_est, safety_true, flt_est, flt_true = initializeSafetyFilter(seg_est, seg_true, pd)
  
  alpha = 10
  comparison_safety = lambda r: alpha * r
  safety_learned = LearnedSegwaySafetyAAR(safety_est)
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

  # Define kernels and covariance function of GP
  active_dimsu = np.array([0])
  ku = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel(active_dims=active_dimsu))

  active_dimsv = np.array([1, 2, 3, 4, 6, 8])
  ka = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(active_dims=active_dimsv, ard_num_dims=6))
  k1 = ku*ka

  kb = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(active_dims=active_dimsv, ard_num_dims=6))
  k2 = kb
  covar_module = k1 + k2
  
  best = []
  btrue = []
  blearn = []
  aest = []
  atrue = []
  alearn = []
  bvar = []
  avar = []
  resexps = []
  ustd_list = []
  input_data_list = []
  residual_true_list = []
  residual_pred_list = []
  residual_pred_lower_list = []
  residual_pred_upper_list = []
  residual_pred_compare_list = []

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
      endpoint = data_episode[-1]
      data_episode = data_episode[0:-1]

      # Postprocessing data from episode for analysis and plotting
      if(i>0):
        drift_est, drift_true, drift_learned, act_est, act_true, act_learned, drift_var, act_var, res_expected = postProcessEpisode(xso, uso, ts_qp, safety_est, safety_true, safety_learned, endpoint)
        atrue.append(act_true)
        aest.append(act_est)
        alearn.append(act_learned)
        avar.append(act_var)  
      
        btrue.append(drift_true)
        best.append(drift_est)
        blearn.append(drift_learned)
        bvar.append(drift_var)   
        resexps.append(res_expected)

      # Concatenate logs from multiple episodes for GP fitting
      state_data = [np.concatenate((old, new)) for old, new in zip(state_data, [xso])]
      data = [np.concatenate((old, new)) for old, new in zip(data, data_episode)]

      drift_inputs, act_inputs, usc, residualsc, _, _, _, _, respredsc = data
      
      # FInd the predictions after each round and compare with true residuals
      if i > 0:
          #predictions on current episode data
          # first normalize with previous round mean and std
          normalized_data_test = (drift_inputs - preprocess_mean)/preprocess_std
          input_data_test = np.concatenate(((usc - usmean)/usstd, normalized_data_test),axis=1)
          #store input data under previous round transformation
          input_data_list.append( input_data_test )
          input_data_test_tensor = torch.from_numpy(input_data_test).float().to(device)
          # Get into evaluation (predictive posterior) mode
          residual_model.eval()
          likelihood.eval()
          #gpytorch.settings.fast_pred_var()
          
          #with torch.no_grad(), gpytorch.settings.max_cg_iterations(3000):
          with torch.no_grad(), gpytorch.settings.fast_computations(solves=False):
              #safety_learned.residual_model.to(device)
              respred_test = likelihood(safety_learned.residual_model(input_data_test_tensor.cpu()))

          lower, upper = respred_test.confidence_region()

          if device!="cpu":
            lower = lower.cpu().detach().numpy()
            upper = upper.cpu().detach().numpy()
            mean = respred_test.mean.cpu().detach().numpy()
            var = respred_test.variance.cpu().detach().numpy()
          else:
            lower = lower.detach().numpy()
            upper = upper.detach().numpy()
            mean = respred_test.mean.detach().numpy()
            var = respred_test.variance.detach().numpy()
          
          residual_true_list.append( residualsc )
          # this is from a*u + b
          residual_pred_compare_list.append( respredsc )
          # from GP prediction directly
          residual_pred_list.append( mean )
          residual_pred_lower_list.append( lower )
          residual_pred_upper_list.append( upper )
      
      # Prepare data for GP fitting
      downsample_rate = 5
      drift_inputs, act_inputs, us, residuals,respreds = downsample(drift_inputs, act_inputs, usc, residualsc, respredsc, downsample_rate)
      
      # Normalization on input and u
      normalized_data, preprocess_mean, preprocess_std = standardize(drift_inputs)
      usstd = np.std(us)
      usmean = 0
      input_data = np.concatenate(((us - usmean)/usstd, normalized_data), axis=1)    

      ndata = input_data.shape[0]
      nfeat = input_data.shape[1]
      print("Number of data points: ", ndata)

      residuals = residuals + 0.01*np.random.randn(residuals.size, )
      if i > 0:
          ustd_list.append(usstd)

      likelihood = gpytorch.likelihoods.GaussianLikelihood()
      likelihood.noise = 0.01
      input_data_tensor = torch.from_numpy(input_data).float().to(device)
      residuals_tensor = torch.from_numpy(residuals.ravel()).float().to(device)
      
      residual_model = ExactGPModel(input_data_tensor, residuals_tensor, likelihood, k1, k2, covar_module)

      if i == 0:
          # save random initialization model      
          torch.save(residual_model.state_dict(),"residual_model_iter_{}.pth".format(str(0)))
          adam_lr = 0.03
          training_iter = 200
      elif i >= 10:
          state_dict = torch.load("residual_model_iter_{}.pth".format(str(0)))
          residual_model.load_state_dict(state_dict)
          adam_lr = 0.04
          training_iter = 0
      else:
          #load previous episode trained model
          #state_dict = torch.load("residual_model_iter_{}.pth".format(str(i)))
          # load random initialization
          state_dict = torch.load("residual_model_iter_{}.pth".format(str(0)))
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
      torch.save(residual_model.state_dict(),"residual_model_iter_{}.pth".format(str(i+1)))
      
      """
      for k in range(i):
          residual_model_iter = ExactGPModel(input_data_tensor, residuals_tensor,likelihood, k1, k2, covar_module)
          state_dict = torch.load("residual_model_iter_{}.pth".format(str(k+1)))
          residual_model_iter.load_state_dict(state_dict)
          # load to gpu if possible
          if device!="cpu":
            input_data_tensor = input_data_tensor.to(device)
            residuals_tensor = residuals_tensor.to(device)
            residual_model_iter.k1 = residual_model_iter.k1.to(device)
            residual_model_iter.k2 = residual_model_iter.k2.to(device)
            residual_model_iter = residual_model_iter.to(device)
            likelihood = likelihood.to(device)
          
          mll_iter = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, residual_model_iter)
          # calculate output from previous saved models
          output_iter = residual_model_iter(input_data_tensor)
          mll_curr = mll_iter(output_iter, residuals_tensor)
          print("MLL using iter " + str(k+1) + " is: " + str(mll_curr.cpu().item()))
      """

      safety_learned = LearnedSegwaySafetyAAR( safety_est )
      
      safety_learned.residual_model = residual_model.cpu()
      safety_learned.likelihood = likelihood.cpu()
      safety_learned.usstd = usstd

      # Evaluate covariance matrix with the data  
      safety_learned.Kinv = torch.inverse( residual_model.covar_module( input_data_tensor ).evaluate() 
                                                + residual_model.likelihood.noise.item()*torch.eye( input_data_tensor.shape[0] ) )  
      safety_learned.alpha = torch.matmul(safety_learned.Kinv, torch.from_numpy(residuals).float() )
      safety_learned.input_data_tensor = input_data_tensor
      safety_learned.preprocess_mean = torch.from_numpy( preprocess_mean[0] )
      safety_learned.preprocess_std = torch.from_numpy( preprocess_std[0] )
      safety_learned.comparison_safety = comparison_safety
      
      # Controller Update
      phi_0_learned = lambda x, t: safety_learned.drift_learned( x, t ) 
      phi_1_learned = lambda x, t: safety_learned.act_learned( x, t )
      flt_learned = FilterControllerVar( seg_est, phi_0_learned, phi_1_learned, pd, sigma=0.0)

  print(residual_model.covar_module.kernels[0].kernels[1].outputscale)
  print(residual_model.covar_module.kernels[1].outputscale)
  print(residual_model.covar_module.kernels[0].kernels[1].base_kernel.lengthscale)
  print(residual_model.covar_module.kernels[1].base_kernel.lengthscale)

  # PLotting training data for analysis 
  if num_episodes > 1:
    ebs = int(len(state_data[0])/num_episodes)
    ebs_res = int(len(residual_pred_list[-1])/num_episodes)
    #plotTrainStates(input_data_list, ebs_res, num_episodes, save_dir, rnd_seed)
  else:
    ebs = len(state_data[0])
    ebs_res = 0
    
  plotTrainMetaData(alearn, atrue, aest, blearn, btrue, best, avar, bvar, ustd_list, residual_true_list, residual_pred_list,
                      residual_pred_lower_list, residual_pred_upper_list,residual_pred_compare_list, num_episodes, ebs,
                    save_dir, rnd_seed)

  num_violations_a = evaluateTrainedModel(seg_est, seg_true, flt_est, flt_true, pd, state_data, safety_learned, safety_est, safety_true, comparison_safety, x_0s_test, num_tests,0, save_dir)
  print("viol-0: ", num_violations_a)
  num_violations_b = evaluateTrainedModel(seg_est, seg_true, flt_est, flt_true, pd, state_data, safety_learned, safety_est, safety_true, comparison_safety, x_0s_test, num_tests,0.5, save_dir)
  print("viol-0.5: ", num_violations_b)
  num_violations_c = evaluateTrainedModel(seg_est, seg_true, flt_est, flt_true, pd, state_data, safety_learned, safety_est, safety_true, comparison_safety, x_0s_test, num_tests,1.0, save_dir)
  print("viol-1.0: ", num_violations_c)

  return num_violations_c

rnd_seed_list = [123]
#rnd_seed_list = [ 123, 234, 345, 456, 567, 678, 789, 890, 901, 12]
#rnd_seed_list = [345]
# Episodic Learning Setup
num_violations_list = []
num_episodes = 5
num_tests = 10

parent_path = "./experiments/segway_modular_gp/"

if not os.path.isdir(parent_path):
    os.mkdir(parent_path)

for rnd_seed in rnd_seed_list:
  dirs = parent_path + str(rnd_seed) + "/"
  if not os.path.isdir(dirs):
      os.mkdir(dirs)
  num_violations_c = run_experiment(rnd_seed, num_episodes, num_tests, dirs)
  num_violations_list.append(num_violations_c)

print("num_violations_list: ", num_violations_list)
