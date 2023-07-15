from core.controllers import FilterController
from .handlers import SafetyCoordinateReduced


def initializeSafetyFilter(ex_quad, ex_quad_true, ex_quad_output, ex_quad_true_output, fb_lin):
    """
        Initialize safety filters based on true system and system estimate:

        Args:
            ex_quad: Extended planar quadrotor with parameter estimates 
            ex_quad_true: Extended planar quadrotor with true parameters 
            ex_quad_output: Output of estimated quadrotor
            ex_quad_true_output: Output of true quadrotor
            fb_lin: Feedback linearized dynamics
    """
    #x_e = 1.8
    #y_e = 0.6
    #rad = 0.32
    x_e = 1.85
    y_e = 0.6
    rad = 0.28
    
    safety_est = SafetyCoordinateReduced( ex_quad, x_e, y_e, rad)
    safety_true = SafetyCoordinateReduced( ex_quad_true, x_e, y_e, rad)

    # Alpha tuning very critical
    alpha = 10
    comp_safety = lambda r: alpha * r
    phi_0_est = lambda x, t: safety_est.drift( x, t ) + comp_safety( safety_est.eval( x, t ) )
    phi_1_est = lambda x, t: safety_est.act( x, t )
    phi_0_true = lambda x, t: safety_true.drift( x, t ) + comp_safety( safety_true.eval( x, t ) )
    phi_1_true = lambda x, t: safety_true.act( x, t )

    # IMPORTANT: There is a key assumption here. The stabilizing controller knows the true system. It is an oracle.
    # IMPORTANT: BUT THE SAFETY FILTER DOES NOT KNOW THE TRUE SYSTEM
    flt_est = FilterController( ex_quad, phi_0_est, phi_1_est, fb_lin)
    flt_true = FilterController( ex_quad_true, phi_0_true, phi_1_true, fb_lin)
    
    return safety_est, safety_true, flt_est, flt_true