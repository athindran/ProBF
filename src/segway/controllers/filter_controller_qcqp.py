import cvxpy as cp
import numpy as np
import scipy

from core.controllers import Controller
from cvxpy.error import SolverError

class FilterControllerQCQP(Controller):
    """Class for solving ProBF-QCQP using the convex relaxation."""

    def __init__(self, affine_dynamics, phi_0, desired_controller, delta = 2.0):

        Controller.__init__(self, affine_dynamics)
        self.affine_dynamics = affine_dynamics
        self.phi_0 = phi_0
        #self.phi_1 = phi_1
        self.desired_controller = desired_controller
        self.delta = delta
    
    def eval(self, x, t):
        # Evaluate mean and variance
        phi0, varb, varab, phi1, vara = self.phi_0( x, t )
        #phi1, vara = self.phi_1( x, t )

        # Evaluate desired controller
        uc = self.desired_controller.process( self.desired_controller.eval( x, t ) )
        u = cp.Variable((3))
        
        # Construct the matrices of the convex program
        delta = self.delta
        varm = scipy.linalg.sqrtm(np.array([[vara[0], varab[0], 0],[varab[0], varb[0], 0],[0, 0, 0]]))
        cu = np.array([[0],[0],[1]])
        prob = cp.Problem(cp.Minimize(cp.square(u[0])-2*u[0]*uc[0]),[phi1[0]*u[0]+phi0[0]-delta*u[2]>=0, cp.norm(varm@u)<=cu.T@u, u[2]>=0, u[1]-1==0])
    
        # Repeatedly solve the program with decreasing values of delta until success.
        try:
          prob.solve()
        except SolverError:
          pass 
    
        ucurr = u
        if prob.status not in ["optimal","optimal_inaccurate"]:  
          while prob.status not in ["optimal","optimal_inaccurate"]:
            u = cp.Variable((3))
            delta = delta/2.0  
            prob = cp.Problem(cp.Minimize(cp.square(u[0])-2*u[0]*uc[0]),[phi1[0]*u[0]+phi0[0]-delta*u[2]>=0, cp.norm(varm@u)<=cu.T@u, u[2]>=0, u[1]-1==0])
            try:
              prob.solve()
            except SolverError:
              pass
        
          ucurr = u
        self.delta = delta
        
        return self.desired_controller.process(np.array([ucurr[0].value])).T