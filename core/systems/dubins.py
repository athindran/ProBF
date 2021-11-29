from numpy import array, cos, sin, tanh, dot
from core.dynamics import SystemDynamics

class Dubins(SystemDynamics):
    """Planar Segway system. State is [x, theta, x_dot, theta_dot], where x is
    the position of the Segway base in m, x_dot is the velocity in m / sec,
    theta is the angle of the frame in rad clockwise from upright, and
    theta_dot is the angular rate in rad / sec. The input is [u], where u is
    positive or negative percent of maximum motor voltage.
    Attributes:
    x_ddot drift component, f3: float * float * float -> float
    theta_ddot drift component, f4: float * float * float -> float
    x_ddot actuation component, g3: float -> float
    theta_ddot actuation component, g4: float -> float
    """

    def __init__(self, v_nom=2.0, L=1.0):
        SystemDynamics.__init__(self, 3, 1)
        self.params = v_nom, L
        self.v_nom = v_nom
        self.L = L

    def drift(self, x, t):
        _, _, theta = x
        return array([self.v_nom*cos(theta),self.v_nom*sin(theta),0.0])

    def act(self, x, t):
        _, _, theta = x
        return array([[0],[0],[self.v_nom/self.L]])
    
    def eval_dot(self, x, u, t):
        return self.drift(x, t) + dot(self.act(x, t), u)