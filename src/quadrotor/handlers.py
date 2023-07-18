from core.controllers import Controller
from core.dynamics import AffineDynamics, ScalarDynamics

from numpy import eye, cross, array, dot, zeros, sin, cos, arctan

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
    
class SigmoidHandler():
    def __init__(self, a1, a2, a3):
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3

    def sigmoid(self, z):
        return -self.a1*arctan(self.a2*z + self.a3)
    
    def sigmoid_grad(self, z):
        return -(self.a1*self.a2)/(1+(self.a2*z + self.a3)**2)
    
    def sigmoid_hess(self, z):
        return 2*(self.a1*self.a2*self.a2)*(self.a2*z + self.a3)/((1+(self.a2*z + self.a3)**2)**2)
    
class virtualpositionCLF(AffineDynamics, ScalarDynamics):
    """
    Safety function setup: Quadrotor should not get close to a ball
    """
    def __init__(self, dynamics, ts_qp, x_d, x_dd, freq=100, k=0.1, epsilon=0.1, eta=0.1):
        self.dynamics = dynamics
        self.mass = dynamics.params[0]
        self.g = dynamics.params[2]

        self.x_d = x_d
        self.x_dd = x_dd
        self.ts_qp = ts_qp
        self.freq = freq
        self.k = k
        self.eta = eta
        self.epsilon = epsilon
        
    def eval( self, x, t ):
        xpos = x[0]
        ypos = x[1]
        xposd = x[3]
        yposd = x[4]
        
        t = int(t*self.freq)
        error_v = array([xposd - self.x_dd[0, t], yposd - self.x_dd[1, t]])
        error_x = array([xpos - self.x_d[0, t], ypos - self.x_d[1, t]])
        return 0.5*self.k*dot(error_x, error_x) + 0.5*self.mass*dot(error_v, error_v) + self.epsilon*dot(error_v, error_x)
    
    def dVdx( self, x , t ):
        # Note that these can be obtained by taking the 4th derivative of CBF
        xpos = x[0]
        ypos = x[1]
        #theta = x[2]
        xposd = x[3]
        yposd = x[4]
        t = int(t*self.freq)
        error_v = array([xposd - self.x_dd[0, t], yposd - self.x_dd[1, t]])
        error_x = array([xpos - self.x_d[0, t], ypos - self.x_d[1, t]])
        return array( [self.k*error_x[0] + self.epsilon*error_v[0], self.k*error_x[1] + self.epsilon*error_v[1], 0,
                       self.mass*error_v[0] + self.epsilon*error_x[0], self.mass*error_v[1] + self.epsilon*error_x[1], 0])
    
    def drift( self, x, t ):
        xposd = x[3]
        yposd = x[4]
        thetad = x[5]

        pseudo_drift = array([xposd, yposd, thetad, 0, -self.g, 0])
        return dot( self.dVdx( x, t ), pseudo_drift )
        
    def act(self, x, t):
        pseudo_act = zeros((6, 2))
        pseudo_act[3:5, :] = eye(2)/self.mass
        return dot( self.dVdx( x, t ), pseudo_act )
    
    def position_clf_params(self, x, t):
        xposd = x[3]
        yposd = x[4]
        thetad = x[5]
        pseudo_drift = array([xposd, yposd, thetad, 0, -self.g, 0]) 
        pseudo_act = zeros((6, 2))
        pseudo_act[3:5, :] = eye(2)/self.mass
        phi0 = dot( self.dVdx( x, t ), pseudo_drift ) + self.eta*self.eval(x, t)
        phi1 = dot( self.dVdx( x, t ), pseudo_act )
        return phi0, phi1                                                                             

class orientationCLF(AffineDynamics, ScalarDynamics):
    # CLF for tracking the faster theta dynamics
    def __init__(self, quad, theta_d, ko, epsilono, etao):
        self.dynamics = quad
        self.g = quad.params[2]
        self.J = quad.params[1]
        self.theta_d = theta_d
        self.ko = ko
        self.etao = etao
        self.epsilono = epsilono
        
    def eval( self, x, t ):
        theta = x[2]
        theta_dot = x[5]
        error_tdot = array([theta_dot])
        error_t = array([theta - self.theta_d])
        return 0.5*self.ko*dot(error_t, error_t) + 0.5*self.J*dot(error_tdot, error_tdot) + self.epsilono*dot(error_t, error_tdot)
    
    def dVdx( self, x , t ):
        # Note that these can be obtained by taking the 4th derivative of CBF
        theta = x[2]
        theta_dot = x[5]
        error_tdot = array([theta_dot])
        error_t = array([theta - self.theta_d])
        return array( [0, 0, self.ko*error_t[0] + self.epsilono*error_tdot[0], 0, 0, 
                       self.J*error_tdot[0] + self.epsilono*error_t[0]])
    
    def drift( self, x, t ):
        xposd = x[3]
        yposd = x[4]
        thetad = x[5]

        pseudo_drift = array([xposd, yposd, thetad, 0, -self.g, 0])
        return dot( self.dVdx( x, t ), pseudo_drift )
        
    def act(self, x, t):
        pseudo_act = zeros((6, 1))
        pseudo_act[5, 0] = 1/self.J
        return dot( self.dVdx( x, t ), pseudo_act )
    
    def orientation_clf_params(self, x, t):
        xposd = x[3]
        yposd = x[4]
        thetad = x[5]

        pseudo_drift = array([xposd, yposd, thetad, 0, -self.g, 0])
        pseudo_act = zeros((6, 1))
        pseudo_act[5, 0] = -1/self.J
        phi0 = dot( self.dVdx( x, t ), pseudo_drift ) + self.etao*self.eval(x, t)
        phi1 = dot( self.dVdx( x, t ), pseudo_act )
        return phi0, phi1

class QuadrotorObstacleSafety(AffineDynamics, ScalarDynamics):
    # CBF for avoiding obstacke
    def __init__(self, quad, x_ob, rad2_ob, gamma, beta=1.3):
        self.dynamics = quad 
        self.obstacle_position = x_ob
        self.obstacle_radius2 = rad2_ob
        self.sigmoid_handler = SigmoidHandler(1, 1, 1)
        #self.comparison_safety = comparison_safety
        self.beta = beta
        self.gamma = gamma

    def eval( self, x, t ):
        obstacle_vector = x[0:2] - self.obstacle_position
        orientation_vector = array([sin(x[2]), cos(x[2])])
        gfunc = dot(obstacle_vector, obstacle_vector) - self.beta*self.obstacle_radius2
        s = dot(orientation_vector, obstacle_vector)
        sfunc = self.sigmoid_handler.sigmoid( s )

        gfuncdot = 2*dot(x[3:5], obstacle_vector)
        p = cross(orientation_vector, obstacle_vector)
        v = dot(orientation_vector, x[3:5])
        sfuncdot = self.sigmoid_handler.sigmoid_grad(s)*(p*x[5] + v)
        hfunc = self.gamma*(gfunc - sfunc) + gfuncdot - sfuncdot

        return hfunc
    
    def drift( self, x, t ):
        obstacle_vector = x[0:2] - self.obstacle_position
        orientation_vector = array([sin(x[2]), cos(x[2])])
        s = dot(orientation_vector, obstacle_vector)
        #sfunc = self.sigmoid_handler.sigmoid( s )
        sfuncgrad = self.sigmoid_handler.sigmoid_grad(s)
        sfunchess = self.sigmoid_handler.sigmoid_hess(s)

        p = cross(orientation_vector, obstacle_vector)
        v = dot(orientation_vector, x[3:5])
        w = cos(x[2])*x[3] - sin(x[2])*x[4]

        #sfuncdot = self.sigmoid_handler.sigmoid_grad(s)*(p*x[5] + v)

        gfuncdot = 2*dot(x[3:5], obstacle_vector)
        gfuncdotdot = 2*dot(x[3:5], x[3:5]) - 2*obstacle_vector[1]*self.dynamics.params[2]

        ghatfuncdot = gfuncdot - sfuncgrad*(p*x[5] + v)
        phi0 = ghatfuncdot*self.gamma - sfunchess*(p*x[5]+v)**2
        phi0 += -sfuncgrad*x[5]*(2*w - s*x[5]) + gfuncdotdot
        phi0 += sfuncgrad*cos(x[2])*self.dynamics.params[2]

        return phi0
    
    def act(self, x, t):
        obstacle_vector = x[0:2] - self.obstacle_position
        orientation_vector = array([sin(x[2]), cos(x[2])])
        s = dot(orientation_vector, obstacle_vector)
        sfuncgrad = self.sigmoid_handler.sigmoid_grad(s)
        
        p = cross(orientation_vector, obstacle_vector)

        phi1 = array([(2*s-sfuncgrad)/self.dynamics.params[0], sfuncgrad*p/self.dynamics.params[1]])

        return phi1
    
    
    def obstacle_cbf_params(self, x, t):
        obstacle_vector = x[0:2] - self.obstacle_position
        orientation_vector = array([sin(x[2]), cos(x[2])])
        s = dot(orientation_vector, obstacle_vector)
        #gfunc = dot(obstacle_vector, obstacle_vector) - self.beta*self.obstacle_radius2
        sfuncgrad = self.sigmoid_handler.sigmoid_grad(s)
        sfunchess = self.sigmoid_handler.sigmoid_hess(s)

        p = cross(orientation_vector, obstacle_vector)
        v = dot(orientation_vector, x[3:5])
        w = cos(x[2])*x[3] - sin(x[2])*x[4]

        gfuncdot = 2*dot(x[3:5], obstacle_vector)
        gfuncdotdot = 2*dot(x[3:5], x[3:5]) - 2*obstacle_vector[1]*self.dynamics.params[2]

        hfunc = self.eval(x, t)

        phi1 = array([(2*s-sfuncgrad)/self.dynamics.params[0], sfuncgrad*p/self.dynamics.params[1]])
        phi1= -phi1*1/(hfunc)**2
        ghatfuncdot = gfuncdot - sfuncgrad*(p*x[5] + v)
        phi0 = ghatfuncdot*self.gamma - sfunchess*(p*x[5]+v)**2
        phi0 += -sfuncgrad*x[5]*(2*w - s*x[5]) + gfuncdotdot
        phi0 += sfuncgrad*cos(x[2])*self.dynamics.params[2]
        phi0 = -phi0*1/(hfunc)**2
        phi0 += -self.gamma*hfunc
         
        return -phi0, -phi1          
