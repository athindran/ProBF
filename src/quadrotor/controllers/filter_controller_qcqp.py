from numpy import dot, maximum
import cvxpy as cp
import numpy as np
import scipy
from cvxpy.error import SolverError
from core.controllers import Controller

class FilterControllerQCQP(Controller):
    """Class for solving the ProBF-QCQP with two controller inputs."""

    def __init__(self, affine_dynamics, desired_controller, sigma = 0.8):
        Controller.__init__(self, affine_dynamics)
        self.affine_dynamics = affine_dynamics
        self.desired_controller = desired_controller
        self.sigma = sigma
        self.thrust_limit = 12
        self.moment_limit = 1
        
    def eval_novar(self, x, t, phi0, phi1, ud):       
        ud = self.desired_controller.eval(x, t )
        num = phi0 + dot( phi1,  ud)
        den = dot( phi1, ( phi1 ).T)
        if den!=0:
            lambda_star = maximum( 0 , num / den )
        else:
            lambda_star = 0
        
        ufiltered = ud - lambda_star*phi1.T
        ufiltered[0] = np.clip(ufiltered[0], 0, self.thrust_limit)
        ufiltered[1] = np.clip(ufiltered[1], -1*self.moment_limit, self.moment_limit)
        return ufiltered
    
    def eval(self, x, t):        
        # Evaluate mean and variance
        phi0, varb, varab, phi1, vara = self.affine_dynamics.get_cbf_params(x, t)
        
        # Obtain desired controller
        ud = self.desired_controller.process( self.desired_controller.eval(x, t ) )
        
        #return self.eval_novar(x, t, phi0, phi1, ud)
        sigma = self.sigma
        
        # If sigma is very small, there is no need to explicitly use the variance
        if(sigma<0.05):
          return self.eval_novar(x, t, phi0, phi1, ud)

        u = cp.Variable((4)) 
        # Constructing the matrices of the convex program
        deltaf = np.array([[vara[0],0,varab[0],0],[0,vara[1],varab[1],0],[varab[0],varab[1],varb[0],0],[0,0,0,0]])
        delta = scipy.linalg.sqrtm(deltaf)
        cu = np.array([[0],[0],[0],[1]])
        ## NOT FIXED FOR CONVENTION CHANGE YET
        # Try to solve the convex program. If infeasible, reduce sigma.
        prob = cp.Problem(cp.Minimize(cp.square(u[0])+cp.square(u[1])-2*u[0]*ud[0]-2*u[1]*ud[1]),
                          [phi1[0]*u[0]+phi1[1]*u[1]+phi0[0]+sigma*u[3]<=0,cp.norm(delta@u)<=cu.T@u,u[3]>=0,u[2]-1==0])
        
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
            prob = cp.Problem(cp.Minimize(cp.square(u[0])+cp.square(u[1])-2*u[0]*ud[0]-2*u[1]*ud[1]),[phi1[0]*u[0]+phi1[1]*u[1]+phi0[0]-sigma*u[3]>=0,cp.norm(delta@u)<=cu.T@u,u[3]>=0,u[2]-1==0])
            try:
              prob.solve()
            except SolverError:
              print("Failed")
              pass      
          if prob.status in ["optimal", "optimal_inaccurate"]:
            ucurr = [u[0].value, u[1].value]
          else:
            ucurr = ud
            #ucurr = self.eval_novar(x, t, phi0, phi1, uc)
          print("Sigma reduced to:", sigma)
        else:
          ucurr = [u[0].value, u[1].value]
        
        self.sigma = sigma

        ufiltered = self.desired_controller.process(np.array([ucurr[0],ucurr[1]])).T
        ufiltered[0] = np.clip(ufiltered[0], 0, self.thrust_limit)
        ufiltered[1] = np.clip(ufiltered[1], -1*self.moment_limit, self.moment_limit)

        return ufiltered