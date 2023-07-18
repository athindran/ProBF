from core.controllers import Controller

from numpy import dot, maximum
import numpy as np

class FilterController(Controller):
    #QP solver

    def __init__(self, affine_dynamics, desired_controller):

        Controller.__init__(self, affine_dynamics)
        self.affine_dynamics = affine_dynamics
        self.desired_controller = desired_controller
        self.thrust_limit = 12
        self.moment_limit = 1

    def eval(self, x, t):
        phi_0, phi_1 = self.affine_dynamics.obstacle_cbf_params(x, t)
        ud = self.desired_controller.eval(x, t )
        num = -phi_0 - dot( phi_1,  ud)
        den = dot( phi_1, ( phi_1 ).T)
        if den!=0:
            lambda_star = maximum( 0 , num / den )
        else:
            lambda_star = 0
        
        ufiltered = ud + lambda_star*phi_1.T
        ufiltered[0] = np.clip(ufiltered[0], 0, self.thrust_limit)
        ufiltered[1] = np.clip(ufiltered[1], -1*self.moment_limit, self.moment_limit)
        return ufiltered