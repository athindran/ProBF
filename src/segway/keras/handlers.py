import numpy as np
from numpy import array, zeros, concatenate, dot
from numpy.random import permutation

from core.dynamics import ConfigurationDynamics, PDDynamics, LearnedAffineDynamics, AffineDynamics, ScalarDynamics
from core.controllers import Controller
from core.util import differentiate
from core.dynamics import LearnedAffineDynamics
from core.learning import ResidualAffineModel

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Add, Dense, Dot, Input, Reshape, Lambda

class SegwayOutput(ConfigurationDynamics):
    """
      Class to represent observable output which is the first two dimensions for Segway
    """
    def __init__(self, segway):
        ConfigurationDynamics.__init__(self, segway, 1)
        
    def y(self, q):
        return q[1:] - .1383
    
    def dydq(self, q):
        return array([[0, 1]])
    
    def d2ydq2(self, q):
        return zeros((1, 2, 2))

class SegwayPD(PDDynamics):
    """
      Return proportional and derivative terms for use by the PD controller
    """
    def proportional(self, x, t):
        return x[0:2] - array([0, .1383])
    
    def derivative(self, x, t):
        return x[2:4]
    
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

# Angle-Angle Rate Safety Function Definition
class SafetyAngleAngleRate(AffineDynamics, ScalarDynamics):
    """
      Definition of CBF for Segway and its accompanying Lie derivatives.
    """
    def __init__(self, segway, theta_e, angle_max, coeff):
        self.dynamics = segway
        self.theta_e = theta_e
        self.angle_max = angle_max
        self.coeff = coeff
        
    def eval( self, x, t ):
        """
          Definition of CBF
        """
        theta = x[1]
        theta_dot = x[3]
        return 0.5 * ( self.angle_max ** 2 - self.coeff * ( theta_dot ** 2 ) - ( theta - self.theta_e ) ** 2 )
    
    def dhdx( self, x , t ):
        """
          Derivative of CBF wrt state  
        """
        theta = x[1]
        theta_dot = x[3]
        return array( [ 0, - ( theta - self.theta_e ), 0, - self.coeff * theta_dot ] )
    
    def drift( self, x, t ):
        """
          Lie derivative wrt control-independent dynamics
        """
        return dot( self.dhdx( x, t ), self.dynamics.drift( x, t ) )
        
    def act(self, x, t):
        """
          Lie derivative wrt control-dependent dynamics
        """   
        return dot( self.dhdx( x, t ), self.dynamics.act( x, t ) )
    
class LearnedSegwaySafetyAAR_NN(LearnedAffineDynamics):
    """
    Class to setup the CBF and derivatives
    """
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
class KerasResidualScalarAffineModel(ResidualAffineModel):
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
        prediction = self.drift_model(array([(drift_input-self.input_mean)/self.input_std]), training=False).numpy()
        return prediction[0][0]

    def eval_act(self, act_input):
        prediction = self.act_model(array([(act_input-self.input_mean)/self.input_std]), training=False).numpy()
        return prediction[0][0]/self.us_std
    
    def shuffle(self, drift_inputs, act_inputs, us, residuals):
        perm = permutation(len(residuals))
        return drift_inputs[perm], act_inputs[perm], us[perm], residuals[perm]

    def fit(self, drift_inputs, act_inputs, us, residuals, batch_size=1, num_epochs=1, validation_split=0):
        drift_inputs, act_inputs, us, residuals = self.shuffle(drift_inputs, act_inputs, us, residuals)
        self.model.fit([drift_inputs, act_inputs, us], residuals, batch_size=batch_size, epochs=num_epochs, validation_split=validation_split)