from core.dynamics import AffineDynamics, ConfigurationDynamics, LearnedDynamics, PDDynamics, ScalarDynamics
from core.systems import Segway
from core.controllers import Controller, FBLinController, LQRController, FilterController, PDController, QPController
from core.util import differentiate
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.pyplot import cla, clf, Circle, figure, grid, legend, plot, savefig, show, subplot, title, xlabel, ylabel
from numpy import array, concatenate, dot, identity, linspace, ones, savetxt, size, sqrt, zeros
from numpy.random import uniform,seed
import numpy as np
import pdb
import pickle
import os

from QuadSupport import initializeSystem, initializeSafetyFilter, simulateSafetyFilter, SafetyCoordinate
from Plotting import plotQuadStates, plotTrainStates, plotTrainMetaData, plotPhasePlane, plotLearnedCBF, plotQuadTrajectory
from AuxFunc import findSafetyData, findLearnedSafetyData, postProcessEpisode, shuffle_downsample, standardize, generateQuadPoints

from core.dynamics import LearnedAffineDynamics
import torch
import gpytorch

#device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print(device)

class LearnedQuadSafety(LearnedAffineDynamics):
    def __init__(self, quad_safety):
        self.dynamics = quad_safety
        self.res_model = None
        self.input_data = []
        self.inputmean = np.zeros((8,))
        self.inputstd = 1
        self.resstd = 1
        self.usstd = 1
              
    def process_drift(self, x, t):
        dhdx = self.dynamics.dhdx( x, t )
        return concatenate([x, dhdx])

    def process_act(self, x, t):
        dhdx = self.dynamics.dhdx( x, t )
        return concatenate([x, dhdx])
    
    def eval(self, x, t):
        return self.dynamics.eval(x, t)
    
    def driftinit(self, x, t):
        # Drift term with no residual mean and variance
        return self.dynamics.drift(x, t)

    def actinit(self, x, t):
        # Actuation term with no residual mean and variance
        return self.dynamics.act(x, t)
    
    def driftfix(self, x, t):
        # Drift term with fixed variance
        return [self.dynamics.drift(x, t)+comp_safety(self.eval(x,t))],[0.5],[0.0]

    def actdum(self, x, t):
        # Actuation term with fixed variance
        return [self.dynamics.act(x, t)],[70.0]

    def driftn(self, x, t):
        # Drift term with residual mean and variance
        xfull = np.concatenate(([1.0,1.0],np.divide(self.process_drift(x, t)-self.inputmean,self.inputstd)))
        xfull = np.reshape(xfull,(-1,18))
        xfull_tensor = torch.from_numpy(xfull).float()
        
        #print("Sizes",xfull_tensor.shape,self.input_data.shape)

        cross11 = self.res_model.k11(xfull_tensor,self.input_data).numpy()/self.usstd[0,0]
        cross12 = self.res_model.k12(xfull_tensor,self.input_data).numpy()/self.usstd[0,1]
        cross2 = self.res_model.k2(xfull_tensor,self.input_data).numpy()
        
        bmean = np.dot(cross2,self.alpha)
        #amean1 = np.dot(cross11,self.alpha)
        #amean2 = np.dot(cross12,self.alpha)
        
        mean = bmean*self.resstd
        variance = self.res_model.k2(xfull_tensor,xfull_tensor).numpy()-np.dot(np.dot(cross2,self.Kinv),cross2.T)
        varab = np.array([-np.dot(np.dot(cross11,self.Kinv),cross2.T),-np.dot(np.dot(cross12,self.Kinv),cross2.T)])
        return [self.dynamics.drift(x, t)+mean.ravel()+self.resmean+self.comp_safety(self.eval(x,t)),variance.ravel(), varab.ravel()]
        #return self.dynamics.drift(x, t)+mean.ravel()+self.resmean+comp_safety(self.eval(x,t)) 
    
    def actn(self, x, t):
        # Actuation term with residual mean and variance
        xfull = np.concatenate(([1.0,1.0],np.divide(self.process_act(x, t)-self.inputmean,self.inputstd)))
        xfull = np.reshape(xfull,(-1,18))
        xfull_tensor = torch.from_numpy(xfull).float()
        
        cross1 = self.res_model.k11(xfull_tensor,self.input_data).numpy()/self.usstd[0,0]
        cross2 = self.res_model.k12(xfull_tensor,self.input_data).numpy()/self.usstd[0,1]
        
        mean = np.array([np.dot(cross1,self.alpha)*self.resstd, np.dot(cross2,self.alpha)*self.resstd])
        variancequad1 = self.res_model.k11(xfull_tensor,xfull_tensor).numpy()/(self.usstd[0,0])**2-np.dot(np.dot(cross1,self.Kinv),cross1.T)
        variancequad2 = self.res_model.k12(xfull_tensor,xfull_tensor).numpy()/(self.usstd[0,1])**2-np.dot(np.dot(cross2,self.Kinv),cross2.T)
        
        variancequad = np.array([variancequad1[0,0],variancequad2[0,0]])
        return self.dynamics.act(x, t)+mean.ravel(),variancequad.ravel()
        #return self.dynamics.act(x, t)+mean.ravel()
    
    def process_episode(self, xs, us, ts, window=9):
        half_window = (window - 1) // 2
        xs = xs[:len(us)]
        ts = ts[:len(us)]
        
        drift_inputs = array([self.process_drift(x, t) for x, t in zip(xs, ts)])
        act_inputs = array([self.process_act(x, t) for x, t in zip(xs, ts)])

        reps = array([self.dynamics.eval(x, t) for x, t in zip(xs, ts)])
        rep_dots = differentiate(reps, ts)
        rep_ddots = differentiate(rep_dots,ts[1:-1])
        rep_dddots = differentiate(rep_ddots,ts[2:-2])
        rep_ddddots = differentiate(rep_dddots,ts[3:-3])
        
        rep_dot_noms = array([self.dynamics.eval_dot(x, u, t) for x, u, t in zip(xs, us, ts)])
        j = 0
        
        drift_inputs = drift_inputs[half_window:-half_window]
        act_inputs = act_inputs[half_window:-half_window]
        rep_dot_noms = rep_dot_noms[half_window:-half_window]
        
        us = us[0:-2*half_window,:]
        
        residuals = rep_ddddots - rep_dot_noms
        
        return drift_inputs, act_inputs, us, residuals
    
    
    def init_data(self, d_drift_in, d_act_in, m, d_out):
        return [zeros((0, d_drift_in)), zeros((0, d_act_in)), zeros((0, m)), zeros(0)]
        
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,k11,k12,k2,covar_module):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.k11 = k11
        self.k12 = k12
        self.k2 = k2
        self.covar_module = covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Combined Controller
class CombinedController(Controller):
    def __init__(self, controller_1, controller_2, weights):
        self.controller_1 = controller_1
        self.controller_2 = controller_2
        self.weights = weights
        
    def eval(self, x, t):
        u_1 = self.controller_1.process( self.controller_1.eval( x, t ) )
        u_2 = self.controller_2.process( self.controller_2.eval( x, t ) )
        return self.weights[ 0 ] * u_1 + self.weights[ 1 ] * u_2


# %%
def standardize(data_train):
    """
    Standardize a dataset to have zero mean and unit standard deviation.
    :param data_train: 2-D Numpy array. Training data.
    :param data_test: 2-D Numpy array. Test data.
    :return: (train_set, test_set, mean, std), The standardized dataset and
      their mean and standard deviation before processing.
    """
    std = np.std(data_train, 0, keepdims=True)
    std[std == 0] = 1
    mean = np.mean(data_train, 0, keepdims=True)
    data_train_standardized = (data_train - mean) / std
    output = [data_train_standardized]
    output.append(mean)
    output.append(std)
    return output 


def evaluateTrainedModel(ex_quad, ex_quad_true, flt_est, flt_true, fb_lin, state_data, safety_learned, safety_est, safety_true, comp_safety, x_0s_test, num_tests, num_episodes, sigma, save_dir):
  from core.controllers import FilterControllerVar2
  # test for 10 different random points
  num_violations = 0

  for i in range(num_tests):
    # Learned Controller Simulation
    # Use Learned Controller
    print("Test", i)
    x_0 = x_0s_test[i,:]
    qp_estest_data, qp_truetrue_data, qp_trueest_data, ts_qp = simulateSafetyFilter(x_0, ex_quad_true, ex_quad, flt_true, flt_est)
    hs_qp_estest, drifts_qp_estest, acts_qp_estest, hdots_qp_estest = findSafetyData(safety_est, qp_estest_data, ts_qp)
    hs_qp_truetrue, drifts_qp_truetrue, acts_qp_truetrue, hdots_qp_truetrue = findSafetyData(safety_true, qp_truetrue_data, ts_qp)
    hs_qp_trueest, drifts_qp_trueest, acts_qp_trueest, hdots_qp_trueest = findSafetyData(safety_true, qp_trueest_data, ts_qp)

    xs_qp_estest, us_qp_estest = qp_estest_data
    xs_qp_trueest, us_qp_trueest = qp_trueest_data
    xs_qp_truetrue, us_qp_truetrue = qp_truetrue_data
    
    phi_0_learned = lambda x, t: safety_learned.driftn( x, t ) 
    phi_1_learned = lambda x, t: safety_learned.actn( x, t )
    flt_learned = FilterControllerVar2( ex_quad, phi_0_learned, phi_1_learned, fb_lin, sigma=sigma)

    freq = 200 # Hz
    tend = 12

    ts_post_qp = linspace(0, tend, tend*freq + 1)

    qp_data_post = ex_quad_true.simulate(x_0, flt_learned, ts_post_qp)
    xs_post_qp, us_post_qp = qp_data_post

    data_episode = safety_learned.process_episode(xs_post_qp, us_post_qp, ts_post_qp)
    savename = save_dir+"residual_predict_seed{}_run{}.pdf".format(str(rnd_seed),str(i))
    drifts_learned_post_qp, acts_learned_post_qp, hdots_learned_post_qp, hs_post_qp, hdots_post_num = findLearnedSafetyData(safety_learned, qp_data_post, ts_post_qp)
   
    # check violation of safety
    if np.any(hs_post_qp < -0.05):
      num_violations += 1
    
    _, drifts_post_qp, acts_post_qp, hdots_post_qp = findSafetyData(safety_est, qp_data_post, ts_post_qp)
    _, drifts_true_post_qp, acts_true_post_qp, hdots_true_post_qp = findSafetyData(safety_true, qp_data_post, ts_post_qp)

    # Plotting
    savename = save_dir+"learned_controller_seed{}_run{}.pdf".format(str(rnd_seed),str(i))
    plotQuadStates(ts_qp, ts_post_qp, xs_qp_trueest, xs_qp_truetrue, xs_post_qp, us_qp_trueest, us_qp_truetrue, us_post_qp, hs_qp_trueest, hs_qp_truetrue, hs_post_qp, hdots_post_qp, hdots_true_post_qp, hdots_learned_post_qp , drifts_post_qp, drifts_true_post_qp, drifts_learned_post_qp, acts_post_qp, acts_true_post_qp, acts_learned_post_qp, savename)
    
    # Phase Plane Plotting
    savename = save_dir+"learned_traj_seed{}_sigma{}_run{}.pdf".format(str(rnd_seed), str(sigma), str(i))
    pickle.dump(xs_post_qp, open(save_dir+"learned_traj_seed{}_sigma{}_run{}.p".format(str(rnd_seed), str(sigma), str(i)),"wb"))
    plotQuadTrajectory(state_data, num_episodes, xs_post_qp, xs_qp_trueest, xs_qp_truetrue, safety_true.x_e, safety_true.y_e, safety_true.rad, savename)

  # record violations
  print("seed: {}, num of violations: {}".format(rnd_seed, str(num_violations)))
  return num_violations

########################run function##########################################

def run_experiment(rnd_seed, num_episodes, num_tests,save_dir):
  from core.controllers import FilterControllerVar2
  seed(rnd_seed)
  fileh = open(save_dir+"viol.txt", "w", buffering=5)  

  ex_quad, ex_quad_true, ex_quad_output, ex_quad_true_output, fb_lin = initializeSystem()
  safety_est, safety_true, flt_est, flt_true = initializeSafetyFilter(ex_quad, ex_quad_true, ex_quad_output, ex_quad_true_output, fb_lin)

  alpha = 10
  comp_safety = lambda r: alpha * r
  
  safety_learned = LearnedQuadSafety(safety_est)

  # Episodic Parameters
  weights = linspace(0, 1, num_episodes)

  # Controller Setup
  phi_0 = lambda x, t: safety_est.drift( x, t ) + comp_safety( safety_est.eval( x, t ) )
  phi_1 = lambda x, t: safety_est.act( x, t )
  flt_baseline = FilterController( ex_quad, phi_0, phi_1, fb_lin)
  flt_learned = FilterController( ex_quad, phi_0, phi_1, fb_lin)

  
  # Data Storage Setup
  d_drift_in_seg = 16
  d_act_in_seg = 16
  d_out_seg = 1
  
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
 
  # Kernel setup
  active_dimsu1 = np.array([0])
  active_dimsu2 = np.array([1])
  active_dimsv = np.array([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])

  ku1 = gpytorch.kernels.LinearKernel(active_dims=active_dimsu1)
  ku2 = gpytorch.kernels.LinearKernel(active_dims=active_dimsu2)

  ka1 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(active_dims=active_dimsv,ard_num_dims=16))
  ka2 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(active_dims=active_dimsv,ard_num_dims=16))
  k11 = ku1*ka1
  k12 = ku2*ka2
  k1 = k11+k12

  kb = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(active_dims=active_dimsv,ard_num_dims=16))
  k2 = kb

  covar_module = k1+k2
  
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
    
    drift_inputs, act_inputs, usc, residualsc = data

    downsample_factor = 5
    drift_inputs, act_inputs, us, residuals, _ = shuffle_downsample(drift_inputs, act_inputs, usc, residualsc, residualsc, downsample_factor)
    
    normalized_data,inputmean,inputstd = standardize(drift_inputs)
    
    usmean = 0
    usstd = array([[1,1]])
    resmean = 0
    resstd = 1
    
    input_data = np.concatenate(((us-usmean)/usstd[0,0],normalized_data),axis=1)
    residuals = np.reshape((residuals-resmean)/resstd,(residuals.size,-1))+0.01*np.random.randn(residuals.size,1)

    torch.manual_seed(12)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise = 0.01
    input_data_tensor = torch.from_numpy(input_data).float()
    residuals_tensor = torch.from_numpy(residuals.ravel()).float()
    
    covar_module = k1+k2
    res_model = ExactGPModel(input_data_tensor, residuals_tensor, likelihood, k11, k12, k2, covar_module)

    # load to gpu if possible
    input_data_tensor = input_data_tensor.to(device)
    residuals_tensor = residuals_tensor.to(device)
    res_model.k11 = res_model.k11.to(device)
    res_model.k12 = res_model.k12.to(device)
    res_model.k2 = res_model.k2.to(device)
    res_model = res_model.to(device)
    likelihood = likelihood.to(device)
    
    res_model.train()
    likelihood.train()

    if i >=5:
        adam_lr = 0.006
        training_iter = 300
    else:
        adam_lr = 0.009
        training_iter = 300
    
    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': res_model.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=adam_lr)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, res_model)
    
    with gpytorch.settings.max_cg_iterations(10000), gpytorch.settings.cholesky_jitter(1e-4):
        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = res_model(input_data_tensor)
            #print(output)
            # Calc loss and backprop gradients
            gpytorch.settings.max_cg_iterations(10000)
            loss = -mll(output, residuals_tensor)
            loss.backward()
            if(i%5==0):
                print("Loss",loss)      
                print('Iter %d/%d - Loss: %.3f   noise: %.3f' % (
                    i + 1, training_iter, loss.item(),
                    res_model.likelihood.noise.item()
                ))
            optimizer.step()
    

    safety_learned = LearnedQuadSafety(safety_est)
    
    safety_learned.res_model = res_model.cpu()
    safety_learned.resstd = resstd
    safety_learned.resmean = resmean
    safety_learned.usstd = usstd
    safety_learned.usmean = usmean
    input_data = input_data_tensor.cpu()
    safety_learned.Kinv = np.linalg.pinv(res_model.covar_module(input_data).numpy()+0.001*np.eye(input_data.shape[0]))

    safety_learned.alpha = np.dot(safety_learned.Kinv,residuals)

    safety_learned.input_data = input_data
    safety_learned.inputmean = inputmean[0]
    safety_learned.inputstd = inputstd[0]
    safety_learned.comp_safety = comp_safety 
    
    # Controller Update
    phi_0_learned = lambda x, t: safety_learned.driftn( x, t ) 
    phi_1_learned = lambda x, t: safety_learned.actn( x, t )
    flt_learned = FilterControllerVar2( ex_quad, phi_0_learned, phi_1_learned, fb_lin, sigma=0.04)
    
  num_violations = evaluateTrainedModel(ex_quad, ex_quad_true, flt_est, flt_true, fb_lin, state_data, safety_learned, safety_est, safety_true, comp_safety, x_0s_test, num_tests, num_episodes, 0.2, save_dir)
  print("num_violations 0.2", num_violations, file = fileh)
  num_violations = evaluateTrainedModel(ex_quad, ex_quad_true, flt_est, flt_true, fb_lin, state_data, safety_learned, safety_est, safety_true, comp_safety, x_0s_test, num_tests, num_episodes, 0.5, save_dir)
  print("num_violations 0.5", num_violations, file = fileh)
  num_violations = evaluateTrainedModel(ex_quad, ex_quad_true, flt_est, flt_true, fb_lin, state_data, safety_learned, safety_est, safety_true, comp_safety, x_0s_test, num_tests, num_episodes, 0.8, save_dir)
  print("num_violations 0.8", num_violations, file = fileh)

  fileh.close()  
  return num_violations

def evaluateTrue():
    ex_quad, ex_quad_true, ex_quad_output, ex_quad_true_output, fb_lin = initializeSystem()
    safety_est, safety_true, flt_est, flt_true = initializeSafetyFilter(ex_quad, ex_quad_true, ex_quad_output, ex_quad_true_output, fb_lin)
  
    seed(345)
    m = ex_quad.quad.params[0]
    g = ex_quad.quad.params[2]  
    x_0 = array([2.0, 2.0, 0, 0, 0, 0, m * g, 0])
    ic_prec = 0.1
    x_0s_test = generateQuadPoints(x_0, 10, ic_prec) 
    n_viol1 = 0
    n_viol2 = 0
    
    for i in range(10):
        # Learned Controller Simulation
        # Use Learned Controller
        print("Test", i)
        x_0 = x_0s_test[i,:]
        print(x_0)
        qp_estest_data, qp_truetrue_data, qp_trueest_data, ts_qp = simulateSafetyFilter(x_0, ex_quad_true, ex_quad, flt_true, flt_est)
        
        xs_qp_estest, us_qp_estest = qp_estest_data
        xs_qp_trueest, us_qp_trueest = qp_trueest_data
        xs_qp_truetrue, us_qp_truetrue = qp_truetrue_data
        hs_qp_truetrue, drifts_qp_truetrue, acts_qp_truetrue, hdots_qp_truetrue = findSafetyData(safety_true, qp_truetrue_data, ts_qp)
        hs_qp_trueest, drifts_qp_trueest, acts_qp_trueest, hdots_qp_trueest = findSafetyData(safety_true, qp_trueest_data, ts_qp)
        if np.any(hs_qp_truetrue < -0.05):
          n_viol1 += 1
        if np.any(hs_qp_trueest < -0.05):
          n_viol2 += 1
        
        savename = "./quad_modular_gp/true"+str(i)+".pdf"
        
        f = figure(figsize=(10, 8))
        plot(xs_qp_trueest[:, 0], xs_qp_trueest[:, 1], 'g', label='True-Est-QP')
        plot(xs_qp_truetrue[:, 0], xs_qp_truetrue[:, 1], 'c', label='True-True-QP')
        circle = Circle((safety_true.x_e, safety_true.y_e),0.9*np.sqrt(safety_true.rad),color="y")
        fig = plt.gcf()
        ax = fig.gca()
        ax.add_patch(circle)

        grid()
        legend()
        xlabel('$x$', fontsize=16)
        ylabel('$y$', fontsize=16)
        plt.xlim([-4,8])
        plt.ylim([-3,6])
        f.savefig(savename, bbox_inches='tight')
    print("Violations", n_viol1, n_viol2)
    
rnd_seed_list = [123, 234, 345, 456, 567, 678, 789, 890, 901, 12]
#rnd_seed_list = [234]
# Episodic Learning Setup
num_violations_list = []
num_episodes = 10
num_tests = 10
for rnd_seed in rnd_seed_list:
  dirs = "./experiments/quad_modular_gp/"+str(rnd_seed)+"/"
  if not os.path.isdir(dirs):
      os.mkdir(dirs)
  num_violations = run_experiment(rnd_seed, num_episodes, num_tests, dirs)
  num_violations_list.append(num_violations)

print("num_violations_list: ", num_violations_list)
