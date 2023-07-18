from core.dynamics import RoboticDynamics

from numpy import array, zeros, sin, cos

class PlanarQuadrotor2D(RoboticDynamics):
    def __init__(self, m, J, g=9.81):
        RoboticDynamics.__init__(self, 3, 2)
        self.params = m, J, g

    def drift(self, x, t):
        _, _, _, x_dot, y_dot, theta_dot = x
        return array([x_dot, y_dot, theta_dot, 0, -self.params[2], 0])

    def act(self, x, t):
        x, _, theta, _, _, _ = x
        gx = zeros((6, 2))
        gx[:, 0] = array([ 0, 0, 0, sin(theta)/self.params[0], cos(theta)/self.params[0], 0])
        gx[:, 1] = array([ 0, 0, 0, 0, 0, -1/self.params[1]])

        return gx