from src.quadrotor.controllers.filter_controller import FilterController
from ..handlers import QuadrotorObstacleSafety


def initializeSafetyFilter(quad, quad_true, sqp_true, obstacle_position, obstacle_rad2, cbf_gamma, cbf_beta):
    """
        Initialize safety filters based on true system and system estimate:

        Args:
            quad: Planar quadrotor with parameter estimates 
            quad_true: Planar quadrotor with true parameters 
            sqp_true: Stabilizing controller with true parameters
            obstacle_position: Position of obstacle
            obstacle_rad2: Squared radius of obstacle
            cbf_gamma: Parameter gamma used in CBF
            cbf_beta: Parameter beta used in CBF
    """
    safety_true = QuadrotorObstacleSafety( quad_true, obstacle_position, obstacle_rad2, gamma=cbf_gamma, beta=cbf_beta)
    flt_true = FilterController( safety_true, sqp_true)

    safety_est = QuadrotorObstacleSafety( quad, obstacle_position, obstacle_rad2, gamma=cbf_gamma, beta=cbf_beta)
    flt_est = FilterController( safety_est, sqp_true)

    return safety_est, safety_true, flt_est, flt_true