from numpy import array, zeros

from core.dynamics import ConfigurationDynamics, PDDynamics
from core.controllers import Controller


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
    
    

