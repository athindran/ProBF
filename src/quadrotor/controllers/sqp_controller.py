from core.controllers import Controller

from numpy.linalg import inv
from numpy import maximum, sin, cos, arctan, clip, array


class SequentialQPController(Controller):
    """Class for CLF-QP controller"""

    def __init__(self, affine_dynamics_position, Q, orientation_clf):
        """Create an Sequential-QP controller specific to quadrotor.
        """
        Controller.__init__(self, affine_dynamics_position)
        self.affine_dynamics_position = affine_dynamics_position
        self.affine_orientation = orientation_clf
        self.mass = affine_dynamics_position.mass
        self.g = affine_dynamics_position.g
        self.Q = Q
        self.Q_inv = inv( self.Q )
        self.thrust_limit = 12
        self.moment_limit = 1

    def eval(self, x, t):
        
        phi0, phi1 = self.affine_dynamics_position.position_clf_params( x, t )
        den = phi1.T @ self.Q_inv @ phi1 + 1e-8
        num = phi0

        if den!=0:
            lambda_star = maximum( 0 , num / den )
        else:
            lambda_star = 0

        virtual_thrust = -lambda_star*self.Q_inv@phi1
        #return virtual_thrust
        
        #virtual_thrust[1] += self.mass*self.g
        desired_thrust = maximum(virtual_thrust[0]*sin(x[2]) + virtual_thrust[1]*cos(x[2]), 0)
        if virtual_thrust[0]!=0 and virtual_thrust[1]!=0:
            desired_theta = arctan(virtual_thrust[0]/(virtual_thrust[1]+1e-8))
        else:
            desired_theta = 0
        
        #return array([desired_thrust, desired_moment])
        if virtual_thrust[0]!=0 or virtual_thrust[1]!=0:
            self.affine_orientation.theta_d = desired_theta      
            phit0, phit1 = self.affine_orientation.orientation_clf_params(x, t)

            den = phit1.T @ phit1 + 1e-8
            num = phit0
        
            if den!=0:
                lambda_star_t = maximum( 0 , num / den )
            else:
                lambda_star_t = 0

            desired_moment = -lambda_star_t*phit1[0]

            desired_thrust = clip(desired_thrust, 0, self.thrust_limit)
            desired_moment = clip(desired_moment, -1*self.moment_limit, self.moment_limit)
            #print("Final control", desired_thrust, desired_moment)
            return array([desired_thrust, desired_moment])
        else:
            return array([0., 0.])
