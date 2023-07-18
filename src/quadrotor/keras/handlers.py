from numpy import array, concatenate, zeros, dot
from core.dynamics import LearnedAffineDynamics, AffineDynamics, ScalarDynamics
from core.learning.keras import KerasResidualAffineModel

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Add, Dense, Dot, Input, Reshape, Lambda


class LearnedQuadSafety_NN(LearnedAffineDynamics):
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

class SafetyCoordinate(AffineDynamics, ScalarDynamics):
    """
    Safety function setup: Quadrotor should not get close to a ball
    """
    def __init__(self, ex_quad, x_e, y_e, rad):
        self.dynamics = ex_quad
        self.x_e = x_e
        self.y_e = y_e
        self.rad = rad
        
    def eval( self, x, t ):
        """
        Check the following paper for choice of safety function
        https://hybrid-robotics.berkeley.edu/publications/ACC2016_Safety_Control_Planar_Quadrotor.pdf
        We have to use an extended higher-order CBF as described in this paper
        https://arxiv.org/pdf/2011.10721.pdf
        """
        xpos = x[0]
        ypos = x[1]
        #theta = x[2]
        #xposdd = x[6]
        #yposdd = x[7]
        #s = sin(x[2])*(xpos-self.x_e)+cos(x[2])*(ypos-self.y_e)
        return 0.5*((xpos-self.x_e)**2+(ypos-self.y_e)**2-1.0*self.rad)
    
    def dhdx( self, x , t ):
        # Note that these can be obtained by taking the 4th derivative of CBF
        derivs = self.dynamics.eval(x,t)
        #r = derivs[0:2]
        rd = derivs[2:4]
        rdd = derivs[4:6]
        rddd = derivs[6:8]
        xpos = x[0]
        ypos = x[1]
        #theta = x[2]
        #xpdot = x[3]
        #ypdot = x[4]
        #thetadot = x[5]
        #xposdd = x[6]
        #yposdd = x[7]
        return array( [2*rddd[0],2*rddd[1],3*rdd[0],3*rdd[1],2*rd[0],2*rd[1],
                       (xpos-self.x_e),(ypos-self.y_e) ])
    
    def drift( self, x, t ):
        #print("Drift",dot( self.dhdx( x, t ), self.dynamics.drift( x, t ) ))
        return dot( self.dhdx( x, t ), self.dynamics.drift( x, t ) )
        
    def act(self, x, t):
        #print("Act",dot(self.dhdx( x, t ), self.dynamics.act( x, t ) ))
        return dot( self.dhdx( x, t ), self.dynamics.act( x, t ) )

"""
Safety function setup: Quadrotor should not get close to a ball
"""
class SafetyCoordinateReduced(AffineDynamics, ScalarDynamics):
    def __init__(self, ex_quad, x_e, y_e, rad):
        self.dynamics = ex_quad
        self.x_e = x_e
        self.y_e = y_e
        self.rad = rad
        self.subset = [0, 1, 2, 3, 4, 5]
        
    def eval( self, x, t ):
        """
        Check the following paper for choice of safety function
        https://hybrid-robotics.berkeley.edu/publications/ACC2016_Safety_Control_Planar_Quadrotor.pdf
        We have to use an extended higher-order CBF as described in this paper
        https://arxiv.org/pdf/2011.10721.pdf
        """
        xpos = x[0]
        ypos = x[1]
        return 0.5*((xpos-self.x_e)**2+(ypos-self.y_e)**2-1.0*self.rad)
    
    def dhdx( self, x , t ):
        # Note that these can be obtained by taking the 4th derivative of CBF
        xpos = x[0]
        ypos = x[1]
        return array( [(xpos-self.x_e), (ypos-self.y_e), 0, 0, 0, 0])
    
    def drift( self, x, t ):
        return dot( self.dhdx( x, t ), self.dynamics.quad.drift( x[self.subset], t ) )
        
    def act(self, x, t):
        return dot( self.dhdx( x, t ), self.dynamics.quad.act( x[self.subset], t ) )
    
# Keras Residual Scalar Affine Model Definition
class KerasResidualScalarAffineModel(KerasResidualAffineModel):
    def __init__(self, d_drift_in, d_act_in, d_hidden, m, d_out, us_scale, optimizer='sgd', loss='mean_absolute_error'):
        drift_model = Sequential()
        drift_model.add(Dense(d_hidden, input_shape=(d_drift_in,), activation='relu'))
        drift_model.add(Dense(d_out))
        self.drift_model = drift_model
        self.us_scale = us_scale

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
        residuals = Add()([drift_residuals, Dot([2, 1])([act_residuals, Lambda(lambda x: x/self.us_scale)(us) ])])
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
        return prediction[0][0]
