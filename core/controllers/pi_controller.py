from numpy import dot

from .controller import Controller

class PIController(Controller):
    """Class for proportional-derivative policies."""

    def __init__(self, pi_dynamics, K_p, K_i):
        """Create a PDController object.

        Policy is u = -K_p * e_p - K_i * e_i, where e_p and e_d are propotional
        and derivative components of error.

        Inputs:
        Proportional-derivative dynamics, pd_dynamics: PDDynamics
        Proportional gain matrix, K_p: numpy array
        Derivative gain matrix, K_d: numpy array
        """

        Controller.__init__(self, pi_dynamics)
        self.K_p = K_p
        self.K_i = K_i

    def eval(self, x, t):
        e_p = self.dynamics.proportional(x, t)
        e_i = self.dynamics.integral(x, t)
        return -dot(self.K_p, e_p) - dot(self.K_i, e_i)
