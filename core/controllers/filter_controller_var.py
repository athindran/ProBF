from numpy import dot, maximum
from numpy.linalg import solve
from numpy import sign
from scipy.linalg import sqrtm
import cvxpy as cp
import numpy as np
import scipy

from .controller import Controller

class FilterControllerVar(Controller):
    """Class for solving ProBF-QCQP using the convex relaxation."""

    def __init__(self, affine_dynamics, phi_0, phi_1, desired_controller, sigma = 2.0):
        """Create an FBLinController object.

        Policy is u = (act)^-1 * (-drift + aux), where drift and act are
        components of drift vector and actuation matrix corresponding to
        highest-order derivatives of each output coordinate and aux is an
        auxilliary linear controller.

        Inputs:
        Feedback linearizable dynamics, fb_lin_dynamics: FBLinDynamics
        Auxilliary linear controller, linear_controller: LinearController
        """

        Controller.__init__(self, affine_dynamics)
        self.affine_dynamics = affine_dynamics
        self.phi_0 = phi_0
        self.phi_1 = phi_1
        self.desired_controller = desired_controller
        self.sigma = sigma
    
    def eval(self, x, t):
        # Evaluate mean and variance
        phi0, varb, varab = self.phi_0( x, t )
        phi1, vara = self.phi_1( x, t )

        # Evaluate desired controller
        uc = self.desired_controller.process( self.desired_controller.eval(x, t ) )
        u = cp.Variable((3))
        
        # Construct the matrices of the convex program
        sigma = self.sigma
        delta = scipy.linalg.sqrtm(np.array([[vara[0], varab[0], 0],[varab[0], varb[0], 0],[0, 0, 0]]))
        cu = np.array([[0],[0],[1]])
        prob = cp.Problem(cp.Minimize(cp.square(u[0])-2*u[0]*uc[0]),[phi1[0]*u[0]+phi0[0]-sigma*u[2]>=0,cp.norm(delta@u)<=cu.T@u,u[2]>=0,u[1]-1==0])
    
        # Repeatedly solve the program with decreasing values of delta until success.
        try:
          prob.solve()
        except SolverError:
          pass 
    
        ucurr = u
        if prob.status not in ["optimal","optimal_inaccurate"]:  
          while prob.status not in ["optimal","optimal_inaccurate"]:
            sigmahigh = sigma  
            u = cp.Variable((3))
            sigma = sigma/2.0  
            prob = cp.Problem(cp.Minimize(cp.square(u[0])-2*u[0]*uc[0]),[phi1[0]*u[0]+phi0[0]-sigma*u[2]>=0,cp.norm(delta@u)<=cu.T@u,u[2]>=0,u[1]-1==0])
            try:
              prob.solve()
            except SolverError:
              pass
        
          ucurr = u
        self.sigma = sigma
        
        return self.desired_controller.process(np.array([ucurr[0].value])).T
