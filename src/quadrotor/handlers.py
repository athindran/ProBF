from core.dynamics import AffineDynamics, ScalarDynamics
from core.controllers import Controller

from numpy import array, dot

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
        #[r,rd,rdd,rddd] = 
        xpos = x[0]
        ypos = x[1]
        #theta = x[2]
        #xpdot = x[3]
        #ypdot = x[4]
        #thetadot = x[5]
        #xposdd = x[6]
        #yposdd = x[7]
        return array( [(xpos-self.x_e), (ypos-self.y_e), 0., 0., 0., 0., 0., 0. ])
    
    def drift( self, x, t ):
        #print("Drift",dot( self.dhdx( x, t ), self.dynamics.drift( x, t ) ))
        return dot( self.dhdx( x, t ), self.dynamics.drift( x, t ) )
        
    def act(self, x, t):
        #print("Act",dot(self.dhdx( x, t ), self.dynamics.act( x, t ) ))
        return dot( self.dhdx( x, t ), self.dynamics.act( x, t ) )
    
# Combined Controller
class CombinedController(Controller):
    def __init__(self, controller_1, controller_2, weights):
        self.controller_1 = controller_1
        self.controller_2 = controller_2
        self.weights = weights
        
    def eval(self, x, t):
        u_1 = self.controller_1.process( self.controller_1.eval( x, t ) )
        u_2 = self.controller_2.process( self.controller_2.eval( x, t ) )
        return self.weights[ 0 ] * u_1 + self.weights[ 1 ] * u_2