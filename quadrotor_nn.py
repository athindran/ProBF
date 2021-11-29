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
#from tensorflow.python.client import device_lib

from utils.QuadSupport import initializeSystem, initializeSafetyFilter, simulateSafetyFilter, SafetyCoordinate
from utils.Plotting import plotQuadStates, plotTrainStates, plotTrainMetaData, plotPhasePlane, plotLearnedCBF, plotQuadTrajectoryNN
from utils.AuxFunc import findSafetyData, findLearnedSafetyData_nn, postProcessEpisode, shuffle_downsample, standardize, generateQuadPoints

#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from core.dynamics import LearnedAffineDynamics
from core.learning.keras import KerasResidualAffineModel
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Add, Dense, Dot, Input, Reshape, Lambda

class LearnedQuadSafetyAAR_NN(LearnedAffineDynamics):
    def __init__(self, quad_safety, scalar_res_aff_model):
        self.dynamics = quad_safety
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
        #print("Shape", d_out*m)
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


def evaluateTrainedModel(ex_quad, ex_quad_true, flt_est, flt_true, fb_lin, state_data, safety_learned, safety_est, safety_true, comp_safety, x_0s_test, num_tests, num_episodes, save_dir):                       
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
    
    phi_0_learned = lambda x, t: safety_learned.drift( x, t ) + comp_safety( safety_learned.eval( x, t ) )
    phi_1_learned = lambda x, t: safety_learned.act( x, t )
    flt_learned = FilterController( ex_quad, phi_0_learned, phi_1_learned, fb_lin )

    freq = 200 # Hz
    tend = 12

    ts_post_qp = linspace(0, tend, tend*freq + 1)

    qp_data_post = ex_quad_true.simulate(x_0, flt_learned, ts_post_qp)
    xs_post_qp, us_post_qp = qp_data_post

    data_episode = safety_learned.process_episode(xs_post_qp, us_post_qp, ts_post_qp)
    savename = save_dir+"residual_predict_seed{}_run{}.pdf".format(str(rnd_seed),str(i))
    drifts_learned_post_qp, acts_learned_post_qp, hdots_learned_post_qp, hs_post_qp, hdots_post_num = findLearnedSafetyData_nn(safety_learned, qp_data_post, ts_post_qp)
   
    # check violation of safety
    if np.any(hs_post_qp < -0.05):
      num_violations += 1
    
    _, drifts_post_qp, acts_post_qp, hdots_post_qp = findSafetyData(safety_est, qp_data_post, ts_post_qp)
    _, drifts_true_post_qp, acts_true_post_qp, hdots_true_post_qp = findSafetyData(safety_true, qp_data_post, ts_post_qp)

    # Plotting
    savename = save_dir+"learned_controller_seed{}_run{}.pdf".format(str(rnd_seed),str(i))
    plotQuadStates(ts_qp, ts_post_qp, xs_qp_trueest, xs_qp_truetrue, xs_post_qp, us_qp_trueest, us_qp_truetrue, us_post_qp, hs_qp_trueest, hs_qp_truetrue, hs_post_qp, hdots_post_qp, hdots_true_post_qp, hdots_learned_post_qp , drifts_post_qp, drifts_true_post_qp, drifts_learned_post_qp, acts_post_qp, acts_true_post_qp, acts_learned_post_qp, savename)
    
    # Phase Plane Plotting
    savename = save_dir+"learned_traj_seed{}_run{}.pdf".format(str(rnd_seed), str(i))
    pickle.dump(xs_post_qp, open(savename[0:-3]+".p", "wb"))
    plotQuadTrajectoryNN(state_data, num_episodes, xs_post_qp, xs_qp_trueest, xs_qp_truetrue, safety_true.x_e, safety_true.y_e, safety_true.rad, savename)

  # record violations
  print("seed: {}, num of violations: {}".format(rnd_seed, str(num_violations)))
  return num_violations

########################run function##########################################

def run_experiment(rnd_seed, num_episodes, num_tests,save_dir):
  from core.controllers import FilterController

  fileh = open(save_dir+"viol.txt","w",buffering=5)
  seed(rnd_seed)

  ex_quad, ex_quad_true, ex_quad_output, ex_quad_true_output, fb_lin = initializeSystem()
  safety_est, safety_true, flt_est, flt_true = initializeSafetyFilter(ex_quad, ex_quad_true, ex_quad_output, ex_quad_true_output, fb_lin)

  alpha = 10
  comp_safety = lambda r: alpha * r
  
  d_drift_in_seg = 16
  d_act_in_seg = 16
  d_hidden_seg= 200
  d_out_seg = 1
  us_std = 1.0
  res_model_seg = KerasResidualScalarAffineModel(d_drift_in_seg, d_act_in_seg, d_hidden_seg, 2, d_out_seg, us_std)
  safety_learned = LearnedQuadSafetyAAR_NN(safety_est, res_model_seg)

  # Episodic Parameters
  weights = linspace(0, 1, num_episodes)

  # Controller Setup
  phi_0 = lambda x, t: safety_est.drift( x, t ) + comp_safety( safety_est.eval( x, t ) )
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
    
    drift_inputs, act_inputs, usc, residualsc = data

    print("Input mean",safety_learned.res_model.input_mean)

    res_model_seg = KerasResidualScalarAffineModel(d_drift_in_seg, d_act_in_seg, d_hidden_seg, 2, d_out_seg, us_std)
    safety_learned = LearnedQuadSafetyAAR_NN(safety_est, res_model_seg)
    
    safety_learned.res_model.input_mean = np.zeros((16,))
    safety_learned.res_model.input_std = np.ones((16,))
    safety_learned.res_model.usstd = 1.0

    #fit residual model on data
    safety_learned.fit(data,1,num_epochs=10,validation_split=0.1)

    # Controller Update
    phi_0_learned = lambda x, t: safety_learned.drift( x, t ) + comp_safety( safety_learned.eval( x, t ) )
    phi_1_learned = lambda x, t: safety_learned.act( x, t )
    flt_learned = FilterController( ex_quad, phi_0_learned, phi_1_learned, fb_lin )
    
  num_violations = evaluateTrainedModel(ex_quad, ex_quad_true, flt_est, flt_true, fb_lin, state_data, safety_learned, safety_est, safety_true, comp_safety, x_0s_test, num_tests, num_episodes, save_dir)
  print("Violations",num_violations, file=fileh)
  fileh.close()
  
  return num_violations

rnd_seed_list = [123, 234, 345, 456, 567, 678, 789, 890, 901, 12]
#rnd_seed_list = [234]
# Episodic Learning Setup
num_violations_list = []
num_episodes = 10
num_tests = 10
for rnd_seed in rnd_seed_list:
  dirs = "./experiments/quad_modular_nn/"+str(rnd_seed)+"/"
  #os.mkdir(dirs)
  if not os.path.isdir(dirs):
     os.mkdir(dirs)  
  num_violations = run_experiment(rnd_seed, num_episodes, num_tests, dirs)
  num_violations_list.append(num_violations)

print("num_violations_list: ", num_violations_list)
