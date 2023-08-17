from numpy import dot, maximum
import cvxpy as cp
import numpy as np
import scipy
from cvxpy.error import SolverError
from .controller import Controller

class FilterControllerVar2(Controller):
    """Class for solving the ProBF-QCQP with two controller inputs."""

    def __init__(self, affine_dynamics, phi_0, desired_controller, sigma = 2.0):
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
        self.desired_controller = desired_controller
        self.sigma = sigma
        
    def eval_novar(self, x, t, phi0, phi1, uc):       
        num = - phi0 - dot( phi1, uc )
        den = dot(phi1, phi1 .T)
        if den!=0:
            lambda_star = maximum( 0 , num / (den + 1e-4) )
        else:
            lambda_star = 0
            
        return uc +  lambda_star * phi1.T
    
    def eval(self, x, t):
        #print("Evaluating")
        
        # Evaluate mean and variance
        phi0, varb, varab, phi1, vara = self.phi_0( x, t )
        
        # Obtain desired controller
        uc = self.desired_controller.process( self.desired_controller.eval(x, t ) )
        u = cp.Variable((4))
        sigma = self.sigma
        
        # If sigma is very small, there is no need to explicitly use the variance
        #if(sigma<0.05):
        #return self.eval_novar(x, t, phi0, phi1, uc)
        
        # Constructing the matrices of the convex program
        deltaf = np.array([[vara[0],0,varab[0],0],[0,vara[1],varab[1],0],[varab[0],varab[1],varb[0],0],[0,0,0,0]])
        delta = scipy.linalg.sqrtm(deltaf)
        cu = np.array([[0],[0],[0],[1]])
        
        # Try to solve the convex program. If infeasible, reduce sigma.
        prob = cp.Problem(cp.Minimize(cp.square(u[0])+cp.square(u[1])-2*u[0]*uc[0]-2*u[1]*uc[1]),[phi1[0]*u[0]+phi1[1]*u[1]+phi0[0]-sigma*u[3]>=0,cp.norm(delta@u)<=cu.T@u,u[3]>=0,u[2]-1==0])
        
        try:
          prob.solve()
        except SolverError:
          pass  

        if prob.status not in ["optimal","optimal_inaccurate"]:
          print(prob.status)  
          count = 0
          while count<3 and prob.status not in ["optimal","optimal_inaccurate"]:
            sigmahigh = sigma
            count = count+1
            u = cp.Variable((4))
            sigma = sigma/2.0  
            prob = cp.Problem(cp.Minimize(cp.square(u[0])+cp.square(u[1])-2*u[0]*uc[0]-2*u[1]*uc[1]),[phi1[0]*u[0]+phi1[1]*u[1]+phi0[0]-sigma*u[3]>=0,cp.norm(delta@u)<=cu.T@u,u[3]>=0,u[2]-1==0])
            try:
              prob.solve()
            except SolverError:
              print("Failed")
              pass      
          if prob.status in ["optimal", "optimal_inaccurate"]:
            ucurr = [u[0].value, u[1].value]
          else:
            ucurr = uc
            #ucurr = self.eval_novar(x, t, phi0, phi1, uc)
          print("Sigma reduced to:", sigma)
        else:
          ucurr = [u[0].value, u[1].value]
        
        self.sigma = sigma
        return self.desired_controller.process(np.array([ucurr[0],ucurr[1]])).T