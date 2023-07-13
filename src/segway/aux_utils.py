from core.util import differentiate

import numpy as np
from numpy import array, dot
from numpy.random import uniform

def downsample(arraylist, factor): 
    """
        Downsample trajectory data for GP training.

        Inputs:
            drift_inputs: Features corresponding to control-independent dynamics
            act_inputs: Feature corresponding to control-dependent dynamics
            us: Controls used to generate trajectory
            residuals: Residuals generated from simulation
        
        Outputs:
            Downsampled version of all inputs
    """
    perm = range( len(arraylist[-1]) )
    perm = perm[0::factor]

    out_arraylist = []

    for array in arraylist:
        out_arraylist.append( array[perm] )

    return out_arraylist

def generateInitialPoints(x_0, num_episodes, ic_prec):
  """
      Generate initial points from an uniform distribution for 4D Segway.

      Inputs:
          x_0: Center of region
          num_episodes: Episodes to train over
          ic_prec: Precision
      
      Outputs:
          x_0s: Set of initial points corresponding to one for each episode.
  """
  random_0 = np.ones((num_episodes,1))
  random_1 = uniform(1-ic_prec, 1+ic_prec, num_episodes)[:,np.newaxis]
  random_2 = uniform(1-ic_prec, 1+ic_prec, num_episodes)[:,np.newaxis]
  random_3 = uniform(1-ic_prec, 1+ic_prec, num_episodes)[:,np.newaxis]
  weights_all = np.concatenate((random_0,random_1,random_2,random_3), axis=1)
  x_0s = x_0 * weights_all
  return x_0s  

def generateQuadPoints(x_0, num_episodes, ic_prec):
    """
      Generate initial points from an uniform distributionfor 8D Quadrotor.

      Inputs:
          x_0: Center of region
          num_episodes: Episodes to train over
          ic_prec: Precision
      
      Outputs:
          x_0s: Set of initial points corresponding to one for each episode.
    """
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


def standardize(data_train):
    """
      Standardize a dataset to have zero mean and unit standard deviation.
      
      Inputs:
        param data_train: 2-D Numpy array. Training data.
        param data_test: 2-D Numpy array. Test data.
      
      Outputs:
        (train_set, test_set, mean, std), The standardized dataset and their mean and standard deviation before processing.
    """
    std = np.std( data_train, 0, keepdims=True )
    ## If it is constant, no division by any number
    std[std == 0] = 1
    mean = np.mean( data_train, 0, keepdims=True )
    data_train_standardized = (data_train - mean) / std
    
    return data_train_standardized, mean, std

def findSafetyData(safety_filt, sim_data, ts_qp):    
    """
      Post-process the trajectory and evaluate CBF and Lie derivatives.

      Inputs:
        safety_filt: Safety filter
        sim_data: Simulation data of trajectories
        ts_qp: Array of time samples of trajectories

      Outputs:
        hs_data: CBF evaluations along trajectories
        drifts_data: CBF Lie derivatives along control-independent dynamics
        acts_data: CBF Lie derivatives along control-dependent dynamics
        hdots_data: CBF time derivatives
    """
    # Angle-Angle Rate Safety QP Plotting
    xs_data, us_data = sim_data
    
    hs_data = array([safety_filt.eval(x,t) for x, t in zip(xs_data, ts_qp)])
    drifts_data = array([safety_filt.drift(x,t) for x, t in zip(xs_data, ts_qp)])
    acts_data = array([safety_filt.act(x,t) for x, t in zip(xs_data, ts_qp)])
    hdots_data = array([safety_filt.drift(x,t) + dot(safety_filt.act(x,t), u) for x, u, t in zip(xs_data[:-1], us_data, ts_qp[:-1])])

    return hs_data, drifts_data, acts_data, hdots_data

def findLearnedSafetyData_nn(safety_learned, sim_data, ts_post_qp):
    """
      Post-process the trajectory and evaluate CBF and Lie derivatives with neural network modeling.

      Inputs:
        safety_learned: Learned Safety filter
        sim_data: Simulation data of trajectories
        ts_post_qp: Array of time samples of trajectories

      Outputs:
        drifts_learned_post_qp: CBF Lie derivatives along control-independent dynamics based on predictions
        acts_learned_post_qp: CBF Lie derivatives along control-dependent dynamics based on predictions
        hdots_learned_post_qp: CBF time derivatives based on predictions
        hs_post_qp: CBF evaluations
        hdots_post_num: Numerical Time derivatives of CBF
    """
    xs_post_qp, us_post_qp = sim_data
    drifts_learned_post_qp = array([safety_learned.drift(x,t) for x, t in zip(xs_post_qp, ts_post_qp)])
    acts_learned_post_qp = array([safety_learned.act(x,t) for x, t in zip(xs_post_qp, ts_post_qp)])
    hdots_learned_post_qp = array([safety_learned.drift(x,t) + dot(safety_learned.act(x,t),u) for x, u, t in zip(xs_post_qp[:-1], us_post_qp, ts_post_qp[:-1])])

    # Learned Controller Plotting
    hs_post_qp = array([safety_learned.eval(x,t) for x, t in zip(xs_post_qp, ts_post_qp)])
    hdots_post_num = differentiate( hs_post_qp, ts_post_qp)
    
    return drifts_learned_post_qp, acts_learned_post_qp, hdots_learned_post_qp, hs_post_qp, hdots_post_num

def findLearnedSafetyData_gp(safety_learned, sim_data, ts_post_qp):
    """  
      Post-process the trajectory and evaluate CBF and Lie derivatives with GP modeling.

      Inputs:
        safety_learned: Learned Safety filter
        sim_data: Simulation data of trajectories
        ts_post_qp: Array of time samples of trajectories

      Outputs:
        drifts_learned_post_qp: CBF Lie derivatives along control-independent dynamics based on predictions
        acts_learned_post_qp: CBF Lie derivatives along control-dependent dynamics based on predictions
        hdots_learned_post_qp: CBF time derivatives based on predictions
        hs_post_qp: CBF evaluations
        hdots_post_num: Numerical Time derivatives of CBF
    """
    xs_post_qp, us_post_qp = sim_data
    drifts_learned_post_qp = array([safety_learned.drift_learned(x,t)[0] - safety_learned.comparison_safety(safety_learned.eval(x,t)) for x, t in zip(xs_post_qp, ts_post_qp)])
    acts_learned_post_qp = array([safety_learned.act_learned(x,t)[0] for x, t in zip(xs_post_qp, ts_post_qp)])
    hdots_learned_post_qp = array([safety_learned.drift_learned(x,t)[0] - safety_learned.comparison_safety(safety_learned.eval(x,t)) + dot(safety_learned.act_learned(x,t)[0],u) for x, u, t in zip(xs_post_qp[:-1], us_post_qp, ts_post_qp[:-1])])

    # Learned Controller Plotting

    hs_post_qp = array([safety_learned.eval(x,t) for x, t in zip(xs_post_qp, ts_post_qp)])
    hdots_post_num = differentiate( hs_post_qp, ts_post_qp)
    
    return drifts_learned_post_qp, acts_learned_post_qp, hdots_learned_post_qp, hs_post_qp, hdots_post_num

def postProcessEpisode(xs_curr, us, ts_qp, safety_est, safety_true, safety_learned, endpoint):
    """
    Post-process simulation data from episode.

    Inputs:
      xs_curr: State trajectory
      us: Controls trajectory
      ts_qp: Simulation times
      safety_est: Safety filter with dynamics estimate
      safety_true: Safety filter with true dynamics
      safety_learned: Safety filter with learned dynamics
      endpoint: Cut-off for post-processing
    """
    # evaluate dynamics for f(x) and g(x)
    drift_est = array([safety_est.drift(x,t) for x, t in zip(xs_curr, ts_qp)])
    drift_true = array([safety_true.drift(x,t) for x, t in zip(xs_curr, ts_qp)])
    drift_learned = array([safety_learned.drift_learned(x,t)[0]-safety_learned.comparison_safety(safety_learned.eval(x,t)) for x, t in zip(xs_curr, ts_qp)])
    #drift_dynamics_diff = drift_dynamics_est - drift_dynamics_true
    drift_var = array([safety_learned.drift_learned(x,t)[1] for x, t in zip(xs_curr, ts_qp)])
    
    act_est = np.squeeze(array([safety_est.act(x,t) for x, t in zip(xs_curr, ts_qp)]),axis=-1)
    act_true = np.squeeze(array([safety_true.act(x,t) for x, t in zip(xs_curr, ts_qp)]),axis=-1)
    act_learned = np.squeeze(array([safety_learned.act_learned(x,t)[0] for x, t in zip(xs_curr, ts_qp)]),axis=-1)
    
    act_var = np.squeeze(array([safety_learned.act_learned(x,t)[1] for x, t in zip(xs_curr, ts_qp)]),axis=-1)
    #act_dynamics_diff = act_dynamics_est - act_dynamics_true
    res_expected = array([(act_true-act_est)*u+drift_true-drift_est for u in us])
    
    drift_est = drift_est[0:endpoint]
    drift_true = drift_true[0:endpoint]
    drift_learned = drift_learned[0:endpoint]
    drift_var = drift_var[0:endpoint]
    
    act_est = act_est[0:endpoint]
    act_true = act_true[0:endpoint]
    act_learned = act_learned[0:endpoint]
    act_var = act_var[0:endpoint]
    
    res_expected = res_expected[0:endpoint]
    return drift_est,drift_true,drift_learned,act_est,act_true,act_learned, drift_var, act_var, res_expected

