from numpy import zeros, newaxis, array

from core.dynamics import LearnedAffineDynamics
from core.learning.keras import KerasResidualAffineModel
from core.util import differentiate

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Add, Dense, Dot, Input, Reshape, Lambda


class LearnedQuadSafety_NN(LearnedAffineDynamics):
    def __init__(self, quad_safety, scalar_res_aff_model):
        self.dynamics = quad_safety
        self.res_model = scalar_res_aff_model
              
    def process_drift(self, x, t):
        #dhdx = self.dynamics.dhdx( x, t )
        return x

    def process_act(self, x, t):
        #dhdx = self.dynamics.dhdx( x, t )
        return x
    
    def get_cbf_params(self, x, t):
        phi0_learned = self.res_model.eval_drift( self.process_drift(x, t) )
        phi1_learned = self.res_model.eval_act( self.process_act(x, t) )
        phi0_nominal, phi1_nominal = self.dynamics.get_cbf_params(x, t)
        hfunc = self.dynamics.eval(x, t)
        phi0_learned = -phi0_learned/(hfunc**2 + 1e-12)
        phi1_learned = -phi1_learned/(hfunc**2 + 1e-12)
        return phi0_nominal + phi0_learned, phi1_nominal + phi1_learned
    
    def process_episode(self, xs, us, ts, window=3):
     
        half_window = (window - 1) // 2
        xs = xs[:len(us)]
        ts = ts[:len(us)]

        drift_inputs = array([self.process_drift(x, t) for x, t in zip(xs, ts)])
        act_inputs = array([self.process_act(x, t) for x, t in zip(xs, ts)])

        reps = array([self.dynamics.eval(x, t) for x, t in zip(xs, ts)])
        rep_dots = differentiate(reps, ts)

        rep_dot_noms = array([self.dynamics.eval_dot(x, u, t) for x, u, t in zip(xs, us, ts)])
        
        drift_inputs = drift_inputs[half_window:-half_window]
        act_inputs = act_inputs[half_window:-half_window]
        rep_dot_noms = rep_dot_noms[half_window:-half_window]
        
        residuals = rep_dots - rep_dot_noms
        residuals = residuals[:, newaxis]

        return drift_inputs, act_inputs, us, residuals
    
    def init_data(self, d_drift_in, d_act_in, m, d_out):
        return [zeros((0, d_drift_in)), zeros((0, d_act_in)), zeros((0, m)), zeros((0, d_out))]
    
# Keras Residual Scalar Affine Model Definition
class KerasResidualScalarAffineModel(KerasResidualAffineModel):
    def __init__(self, d_drift_in, d_act_in, d_hidden, m, d_out, us_scale, optimizer='sgd', loss='mean_absolute_error'):
        drift_model = Sequential()
        drift_model.add(Dense(d_hidden, input_shape=(d_drift_in,), activation='relu'))
        drift_model.add(Dense(d_hidden, input_shape=(d_hidden,), activation='relu'))
        drift_model.add(Dense(d_out))
        self.drift_model = drift_model
        self.us_scale = us_scale

        drift_inputs = Input((d_drift_in,))
        drift_residuals = self.drift_model(drift_inputs)

        act_model = Sequential()
        act_model.add(Dense(d_hidden, input_shape=(d_act_in,), activation='relu'))
        act_model.add(Dense(d_hidden, input_shape=(d_hidden,), activation='relu'))
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
        prediction = self.drift_model(drift_input[newaxis, :], training=False).numpy()
        return prediction[0][0]

    def eval_act(self, act_input):
        prediction = self.act_model(act_input[newaxis, :], training=False).numpy()
        return prediction[0][0]
