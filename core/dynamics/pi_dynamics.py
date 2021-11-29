from .dynamics import Dynamics

class PIDynamics(Dynamics):
    """Abstract class for dynamics with proportional and derivative components.

    Override eval, eval_dot, proportional, derivative.
    """

    def proportional(self, x, t):
        """Compute proportional component.

        Inputs:
        State, x: numpy array
        Time, t: float

        Outputs:
        Proportional component: numpy array
        """

        pass

    def integral(self, x, t):
        """Compute derivative component.

        Inputs:
        State, x: numpy array
        Time, t: float

        Outputs:
        Integral component: numpy array
        """

        pass