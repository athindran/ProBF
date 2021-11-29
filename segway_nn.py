from core.dynamics import AffineDynamics, ConfigurationDynamics, LearnedDynamics, PDDynamics, ScalarDynamics
from core.systems import Segway
from core.controllers import Controller, FBLinController, LQRController, FilterController, PDController, QPController
from core.util import differentiate
import matplotlib
from matplotlib.pyplot import cla, clf, figure, grid, legend, plot, savefig, show, subplot, title, xlabel, ylabel
from numpy import array, concatenate, dot, identity, linspace, ones, savetxt, size, sqrt, zeros
from numpy.random import uniform,seed
import tensorflow as tf
import numpy as np
import pdb
import pickle
import os
from tensorflow.python.client import device_lib

from SegwaySupport import initializeSystem, initializeSafetyFilter, simulateSafetyFilter, SafetyAngleAngleRate
from Plotting import plotTestStates, plotTrainStates, plotTrainMetaData, plotPhasePlane, plotLearnedCBF
from AuxFunc import findSafetyData, findLearnedSafetyData_nn, postProcessEpisode, shuffle_downsample, standardize, generateInitialPoints

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from core.dynamics import LearnedAffineDynamics
from core.learning.keras import KerasResidualAffineModel
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Add, Dense, Dot, Input, Reshape, Lambda

class LearnedSegwaySafetyAAR_NN(LearnedAffineDynamics):
    def __init__(self, segway_safety_aar, scalar_res_aff_model):
        self.dynamics = segway_safety_aar
        self.res_model = scalar_res_aff_model
              
    def process_drift(self, x, t):
        
        dhdx = self.dynamics.dhdx( x, t )
        
        return concatenate([x, dhdx])

    def process_act(self, x, t):
        
        dhdx = self.dynamics.dhdx( x, t )
        
        return concatenate([x, dhdx])     
    
    def init_data(self, d_drift_in, d_act_in, m, d_out):
        return [zeros((0, d_drift_in)), zeros((0, d_act_in)), zeros((0, m)), zeros(0)]

# Keras Residual Scalar Affine Model Definition
class KerasResidualScalarAffineModel(KerasResidualAffineModel):
    def __init__(self, d_drift_in, d_act_in, d_hidden, m, d_out, us_std, optimizer='sgd', loss='mean_absolute_error'):
        drift_model = Sequential()
        drift_model.add(Dense(d_hidden, input_shape=(d_drift_in,), activation='relu'))
        drift_model.add(Dense(d_out))
        self.drift_model = drift_model
        self.us_std = us_std

        drift_inputs = Input((d_drift_in,))
        drift_residuals = self.drift_model(drift_inputs)

        act_model = Sequential()
        act_model.add(Dense(d_hidden, input_shape=(d_act_in,), activation='relu'))
        act_model.add(Dense(d_out * m))
        act_model.add(Reshape((d_out, m)))
        self.act_model = act_model

        act_inputs = Input((d_act_in,))
        act_residuals = self.act_model(act_inputs)

        us = Input((m,))
        residuals = Add()([drift_residuals, Dot([2, 1])([act_residuals, Lambda(lambda x: x/self.us_std)(us) ])])
        model = Model([drift_inputs, act_inputs, us], residuals)
        model.compile(optimizer, loss)
        self.model = model
        self.input_mean = None
        self.input_std = None

    def eval_drift(self, drift_input):
        #print(drift_input,self.input_mean,self.input_std)
        #print(array([(drift_input-self.input_mean)/self.input_std]))
        return self.drift_model.predict(array([(drift_input-self.input_mean)/self.input_std]))[0][0]

    def eval_act(self, act_input):
        return self.act_model.predict(array([(act_input-self.input_mean)/self.input_std]))[0][0]/self.us_std
    
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


def evaluateTrainedModel(seg_est, seg_true, flt_est, flt_true, pd, state_data, safety_learned, safety_est, safety_true, comp_safety, x_0s_test, num_tests, save_dir):                       
  # test for 10 different random points
  num_violations = 0
  qp_estest_data, qp_truetrue_data, qp_trueest_data, ts_qp = simulateSafetyFilter(seg_true, seg_est, flt_true, flt_est)
  hs_qp_estest, drifts_qp_estest, acts_qp_estest, hdots_qp_estest = findSafetyData(safety_est, qp_estest_data, ts_qp)
  hs_qp_truetrue, drifts_qp_truetrue, acts_qp_truetrue, hdots_qp_truetrue = findSafetyData(safety_true, qp_truetrue_data, ts_qp)
  hs_qp_trueest, drifts_qp_trueest, acts_qp_trueest, hdots_qp_trueest = findSafetyData(safety_true, qp_trueest_data, ts_qp)

  xs_qp_estest, us_qp_estest = qp_estest_data
  xs_qp_trueest, us_qp_trueest = qp_trueest_data
  xs_qp_truetrue, us_qp_truetrue = qp_truetrue_data
    
  for i in range(num_tests):
    # Learned Controller Simulation
    # Use Learned Controller
    phi_0_learned = lambda x, t: safety_learned.drift( x, t ) + comp_safety( safety_learned.eval( x, t ) )
    phi_1_learned = lambda x, t: safety_learned.act( x, t )
    flt_learned = FilterController( seg_est, phi_0_learned, phi_1_learned, pd )

    freq = 500 # Hz
    tend = 3

    x_0 = x_0s_test[i,:]
    ts_post_qp = linspace(0, tend, tend*freq + 1)

    qp_data_post = seg_true.simulate(x_0, flt_learned, ts_post_qp)
    xs_post_qp, us_post_qp = qp_data_post

    data_episode = safety_learned.process_episode(xs_post_qp, us_post_qp, ts_post_qp)
    savename = save_dir+"residual_predict_seed{}_run{}.pdf".format(str(rnd_seed),str(i))
    drifts_learned_post_qp, acts_learned_post_qp, hdots_learned_post_qp, hs_post_qp, hdots_post_num = findLearnedSafetyData_nn(safety_learned, qp_data_post, ts_post_qp)
   
    # check violation of safety
    if np.any(hs_post_qp < 0):
      num_violations += 1
    
    _, drifts_post_qp, acts_post_qp, hdots_post_qp = findSafetyData(safety_est, qp_data_post, ts_post_qp)
    _, drifts_true_post_qp, acts_true_post_qp, hdots_true_post_qp = findSafetyData(safety_true, qp_data_post, ts_post_qp)
    
    theta_bound_u = ( safety_true.theta_e + safety_true.angle_max ) * ones( size( ts_post_qp ) )
    theta_bound_l = ( safety_true.theta_e - safety_true.angle_max ) * ones( size( ts_post_qp ) )

    # Plotting
    savename = save_dir+"learned_controller_seed{}_run{}.pdf".format(str(rnd_seed),str(i))
    plotTestStates(ts_qp, ts_post_qp, xs_qp_trueest, xs_qp_truetrue, xs_post_qp, us_qp_trueest, us_qp_truetrue, us_post_qp, hs_qp_trueest, hs_qp_truetrue, hs_post_qp, hdots_post_qp, hdots_true_post_qp, hdots_learned_post_qp , drifts_post_qp, drifts_true_post_qp, drifts_learned_post_qp, acts_post_qp, acts_true_post_qp, acts_learned_post_qp, theta_bound_u, theta_bound_l, savename)

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

    # # LEARNED CONTROLLER
    savename = save_dir+"learned_h_seed{}_run{}.pdf".format(str(rnd_seed), str(i))
    plotLearnedCBF(ts_qp, hs_qp_trueest, hs_all, ts_post_qp, hs_post_qp, ebs, num_episodes, savename)
    
    # Phase Plane Plotting
    
    epsilon=1e-6
    theta_h0_vals = linspace(safety_true.theta_e-safety_true.angle_max+epsilon, safety_true.theta_e + safety_true.angle_max - epsilon, 1000)
    theta_dot_h0_vals = array([sqrt((safety_true.angle_max ** 2 - (theta - safety_true.theta_e) ** 2) /safety_true.coeff) for theta in theta_h0_vals])
    ebs = int(len(state_data[0])/num_episodes)
    
    savename = save_dir+"learned_pp_seed{}_run{}.pdf".format(str(rnd_seed), str(i))
    plotPhasePlane(theta_h0_vals, theta_dot_h0_vals, xs_qp_trueest, state_data, xs_post_qp, ebs, num_episodes, savename)

  # record violations
  print("seed: {}, num of violations: {}".format(rnd_seed, str(num_violations)))
  return num_violations

########################run function##########################################

def run_experiment(rnd_seed, num_episodes, num_tests,save_dir):
  from core.controllers import FilterController
  seed(rnd_seed)

  seg_est, seg_true, sego_est, sego_true, pd = initializeSystem()
  safety_est, safety_true, flt_est, flt_true = initializeSafetyFilter(seg_est, seg_true, pd)

  alpha = 10
  comp_safety = lambda r: alpha * r
  
  d_drift_in_seg = 8
  d_act_in_seg = 8
  d_hidden_seg= 200
  d_out_seg = 1
  us_std = 0.05
  res_model_seg = KerasResidualScalarAffineModel(d_drift_in_seg, d_act_in_seg, d_hidden_seg, 1, d_out_seg, us_std)
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
    sim_data = seg_true.simulate(x_0, flt_combined, ts_qp)

    # Data Handling
    xs, us = sim_data
    data_episode = safety_learned.process_episode(xs, us, ts_qp)

    state_data = [np.concatenate((old, new)) for old, new in zip(state_data, [xs])]
    print(state_data[0].shape)
    data = [np.concatenate((old, new)) for old, new in zip(data, data_episode)]
    
    drift_inputs, act_inputs, usc, residualsc = data

    print("Input mean",safety_learned.res_model.input_mean)

    res_model_seg = KerasResidualScalarAffineModel(d_drift_in_seg, d_act_in_seg, d_hidden_seg, 1, d_out_seg, us_std)
    safety_learned = LearnedSegwaySafetyAAR_NN(safety_est, res_model_seg)
    
    safety_learned.res_model.input_mean = np.zeros((8,))
    safety_learned.res_model.input_std = np.ones((8,))
    safety_learned.res_model.usstd = 1.0

    #fit residual model on data
    safety_learned.fit(data,1,num_epochs=10,validation_split=0.1)

    # Controller Update
    phi_0_learned = lambda x, t: safety_learned.drift( x, t ) + comp_safety( safety_learned.eval( x, t ) )
    phi_1_learned = lambda x, t: safety_learned.act( x, t )
    flt_learned = FilterController( seg_est, phi_0_learned, phi_1_learned, pd )
    
  num_violations = evaluateTrainedModel(seg_est, seg_true, flt_est, flt_true, pd, state_data, safety_learned, safety_est, safety_true, comp_safety, x_0s_test, num_tests, save_dir)
  return num_violations

rnd_seed_list = [123, 234, 345, 456, 567, 678, 789, 890, 901, 12]

# Episodic Learning Setup
num_violations_list = []
num_episodes = 2
num_tests = 1
for rnd_seed in rnd_seed_list:
  dirs = "./segway_modular_nn/"+str(rnd_seed)+"/"
  os.mkdir(dirs)
  num_violations = run_experiment(rnd_seed, num_episodes, num_tests, dirs)
  num_violations_list.append(num_violations)

print("num_violations_list: ", num_violations_list)