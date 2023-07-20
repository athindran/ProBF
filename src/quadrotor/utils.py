from src.quadrotor.dynamics.planar_quadrotor  import PlanarQuadrotor2D as PlanarQuadrotor
from src.quadrotor.controllers.sqp_controller import SequentialQPController
from src.quadrotor.handlers import virtualpositionCLF, orientationCLF

from numpy import linspace, diag

from copy import deepcopy as copy


def initializeSystemAndController(x_d, x_dd, freq, ts_qp):# System Definitions
    """
        Initialize true planar quadrotor parameters and estimates
    """
    m = 1.3
    J = 0.7
    m_true = 1.0
    J_true = 0.5

    quad = PlanarQuadrotor(m, J)
    quad_true = PlanarQuadrotor(m_true, J_true)

    vp_clf = virtualpositionCLF(quad_true, ts_qp=ts_qp, x_d=x_d, x_dd=x_dd, freq=freq, k=0.5, epsilon=0.7, eta=0.5)
    Q = diag([0.7, 0.01])
    affine_orientation = orientationCLF(copy(quad_true), 0, 1, 1, 20)
    sqp_true = SequentialQPController(vp_clf, Q, affine_orientation)

    return quad, quad_true, sqp_true


def simulateSafetyFilter(x_0, quad_true, quad, flt_true, flt_est):
    """
    Simulate safety filters on estimate system with true safety filter and safety filter with estimates
    Args:
        x_0: Initial state
        quad_true: Planar quadrotor with true parameters 
        quad: Planar quadrotor with parameter estimates
        flt_true: Safety filter with true parameters
        flt_est: Safety filter with parameter estimates
    """
    # Angle-Angle Rate Safety QP Simulation
    freq = 200 # Hz
    tend = 14
    #x_0 = array([2.0, 2.0, 0, 0, 0, 0, m * g, 0])
    ts_qp = linspace(0, tend, tend*freq + 1)

    # Estimated System - Estimated Safety Filter
    qp_estest_data = quad.simulate(x_0, flt_est, ts_qp)

    # True System - True Safety Filter
    qp_truetrue_data = quad_true.simulate(x_0, flt_true, ts_qp)

    # True System - Estimated Safety Filter
    qp_trueest_data = quad_true.simulate(x_0, flt_est, ts_qp)

    return qp_estest_data, qp_truetrue_data, qp_trueest_data, ts_qp