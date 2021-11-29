from .controller import Controller
import numpy as np

class ConstantTVController(Controller):
    """Class for constant action policies."""

    def __init__(self, dynamics, u_const, ts):
        """Create a ConstantController object.

        Inputs:
        Dynamics, dynamics: Dynamics
        Constant action, u_const: numpy array
        """

        Controller.__init__(self, dynamics)
        self.u_const = u_const
        self.ts = ts

    def eval(self, x, t):
        index = np.argwhere(self.ts==t).ravel()
        return self.u_const[index]