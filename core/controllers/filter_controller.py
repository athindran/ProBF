from numpy import dot, maximum
from numpy.linalg import solve

from .controller import Controller

class FilterController(Controller):
    """Class for linearizing feedback policies."""

    def __init__(self, affine_dynamics, phi_0, phi_1, desired_controller):
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

    def eval(self, x, t):

        num = - self.phi_0( x, t ) - dot( self.phi_1( x, t ), self.desired_controller.process( self.desired_controller.eval(x, t ) ) )
        den = dot( self.phi_1( x, t ), ( self.phi_1( x, t ) ).T)
        if den!=0:
            lambda_star = maximum( 0 , num / den )
        else:
            lambda_star = 0
            
        return self.desired_controller.process( self.desired_controller.eval( x, t ) ) + lambda_star * ( self.phi_1( x, t ) ).T
            