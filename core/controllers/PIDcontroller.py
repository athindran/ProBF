from numpy import dot

from .controller import Controller

class PIDController(Controller):
    """Class for proportional-derivative policies."""

    def __init__(self, pid_dynamics, K_p, K_i, K_d):
        """Create a PDController object.

        Policy is u = -K_p * e_p - K_i * e_i, where e_p and e_d are propotional
        and derivative components of error.

        Inputs:
        Proportional-derivative dynamics, pd_dynamics: PDDynamics
        Proportional gain matrix, K_p: numpy array
        Derivative gain matrix, K_d: numpy array
        """

        Controller.__init__(self, pid_dynamics)
        self.K_p = K_p
        self.K_i = K_i
        self.K_d = K_d

    def eval(self, x, t):
        e_p = self.dynamics.proportional(x, t)
        e_i = self.dynamics.integral(x, t)
        e_d = self.dynamics.derivative(x, t)
        #print(e_p,e_i)
        return -dot(self.K_p, e_p) - dot(self.K_i, e_i) - dot(self.K_d,e_d)
