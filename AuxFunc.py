from core.dynamics import AffineDynamics, ConfigurationDynamics, LearnedDynamics, PDDynamics, ScalarDynamics
from core.systems import Segway
from core.controllers import Controller, FBLinController, LQRController, FilterController,PDController, QPController, FilterControllerVar
from core.util import differentiate
from matplotlib.pyplot import cla, clf, figure, grid, legend, plot, savefig, show, subplot, title, xlabel, ylabel, fill_between
import numpy as np
from numpy import array, concatenate, dot, identity, linspace, ones, savetxt, size, sqrt, zeros
from numpy.random import uniform,seed
from numpy.random import permutation
from numpy import clip
import os

def shuffle_downsample(drift_inputs, act_inputs, us, residuals, respreds, factor): 
    # turn off shuffling now
    perm = range(len(residuals))
    perm = perm[0::factor]
    return drift_inputs[perm], act_inputs[perm], us[perm], residuals[perm], respreds[perm]

def generateInitialPoints(x_0, num_episodes, ic_prec):
  random_0 = np.ones((num_episodes,1))
  random_1 = uniform(1-ic_prec, 1+ic_prec, num_episodes)[:,np.newaxis]
  random_2 = uniform(1-ic_prec, 1+ic_prec, num_episodes)[:,np.newaxis]
  random_3 = uniform(1-ic_prec, 1+ic_prec, num_episodes)[:,np.newaxis]
  weights_all = np.concatenate((random_0,random_1,random_2,random_3), axis=1)
  x_0s = x_0 * weights_all
  return x_0s  

def generateQuadPoints(x_0, num_episodes, ic_prec):
    random_0 = uniform(1-ic_prec, 1+ic_prec, num_episodes)[:,np.newaxis]
    random_1 = uniform(1-ic_prec, 1+ic_prec, num_episodes)[:,np.newaxis]
    random_2 = np.ones((num_episodes,1))
    random_3 = np.ones((num_episodes,1))
    random_4 = np.ones((num_episodes,1))
    random_5 = np.ones((num_episodes,1))
    random_6 = np.ones((num_episodes,1))
    random_7 = np.ones((num_episodes,1))

    weights_all = np.concatenate((random_0,random_1,random_2,random_3,random_4,random_5,random_6,random_7), axis=1)
    x_0s = x_0 * weights_all
    return x_0s
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

def findSafetyData(safety_filt, sim_data, ts_qp):    
    # Angle-Angle Rate Safety QP Plotting
    xs_data, us_data = sim_data
    
    hs_data = array([safety_filt.eval(x,t) for x, t in zip(xs_data, ts_qp)])
    drifts_data = array([safety_filt.drift(x,t) for x, t in zip(xs_data, ts_qp)])
    acts_data = array([safety_filt.act(x,t) for x, t in zip(xs_data, ts_qp)])
    hdots_data = array([safety_filt.drift(x,t) + dot(safety_filt.act(x,t), u) for x, u, t in zip(xs_data[:-1], us_data, ts_qp[:-1])])

    return hs_data, drifts_data, acts_data, hdots_data

def findLearnedSafetyData_nn(safety_learned, sim_data, ts_post_qp):
    xs_post_qp, us_post_qp = sim_data
    drifts_learned_post_qp = array([safety_learned.drift(x,t) for x, t in zip(xs_post_qp, ts_post_qp)])
    acts_learned_post_qp = array([safety_learned.act(x,t) for x, t in zip(xs_post_qp, ts_post_qp)])
    hdots_learned_post_qp = array([safety_learned.drift(x,t) + dot(safety_learned.act(x,t),u) for x, u, t in zip(xs_post_qp[:-1], us_post_qp, ts_post_qp[:-1])])

    # Learned Controller Plotting
    hs_post_qp = array([safety_learned.eval(x,t) for x, t in zip(xs_post_qp, ts_post_qp)])
    hdots_post_num = differentiate( hs_post_qp, ts_post_qp)
    
    return drifts_learned_post_qp, acts_learned_post_qp, hdots_learned_post_qp, hs_post_qp, hdots_post_num

def findLearnedSafetyData(safety_learned, sim_data, ts_post_qp):
    xs_post_qp, us_post_qp = sim_data
    drifts_learned_post_qp = array([safety_learned.driftn(x,t)[0] - safety_learned.comp_safety(safety_learned.eval(x,t)) for x, t in zip(xs_post_qp, ts_post_qp)])
    acts_learned_post_qp = array([safety_learned.actn(x,t)[0] for x, t in zip(xs_post_qp, ts_post_qp)])
    hdots_learned_post_qp = array([safety_learned.driftn(x,t)[0] - safety_learned.comp_safety(safety_learned.eval(x,t)) + dot(safety_learned.actn(x,t)[0],u) for x, u, t in zip(xs_post_qp[:-1], us_post_qp, ts_post_qp[:-1])])

    # Learned Controller Plotting

    hs_post_qp = array([safety_learned.eval(x,t) for x, t in zip(xs_post_qp, ts_post_qp)])
    hdots_post_num = differentiate( hs_post_qp, ts_post_qp)
    
    return drifts_learned_post_qp, acts_learned_post_qp, hdots_learned_post_qp, hs_post_qp, hdots_post_num

def postProcessEpisode(xs_curr,us,ts_qp,safety_est,safety_true,safety_learned,startpoint):
    # evaluate dynamics for f(x) and g(x)
    drift_est = array([safety_est.drift(x,t) for x, t in zip(xs_curr, ts_qp)])
    drift_true = array([safety_true.drift(x,t) for x, t in zip(xs_curr, ts_qp)])
    drift_learned = array([safety_learned.driftn(x,t)[0]-safety_learned.comp_safety(safety_learned.eval(x,t)) for x, t in zip(xs_curr, ts_qp)])
    #drift_dynamics_diff = drift_dynamics_est - drift_dynamics_true
    drift_var = array([safety_learned.driftn(x,t)[1] for x, t in zip(xs_curr, ts_qp)])
    
    act_est = np.squeeze(array([safety_est.act(x,t) for x, t in zip(xs_curr, ts_qp)]),axis=-1)
    act_true = np.squeeze(array([safety_true.act(x,t) for x, t in zip(xs_curr, ts_qp)]),axis=-1)
    act_learned = np.squeeze(array([safety_learned.actn(x,t)[0] for x, t in zip(xs_curr, ts_qp)]),axis=-1)
    
    act_var = np.squeeze(array([safety_learned.actn(x,t)[1] for x, t in zip(xs_curr, ts_qp)]),axis=-1)
    #act_dynamics_diff = act_dynamics_est - act_dynamics_true
    res_expected = array([(act_true-act_est)*u+drift_true-drift_est for u in us])
    
    drift_est = drift_est[0:startpoint]
    drift_true = drift_true[0:startpoint]
    drift_learned = drift_learned[0:startpoint]
    drift_var = drift_var[0:startpoint]
    
    act_est = act_est[0:startpoint]
    act_true = act_true[0:startpoint]
    act_learned = act_learned[0:startpoint]
    act_var = act_var[0:startpoint]
    
    res_expected = res_expected[0:startpoint]
    return drift_est,drift_true,drift_learned,act_est,act_true,act_learned, drift_var, act_var, res_expected
