from .handlers import SafetyAngleAngleRate
from core.controllers import FilterController

# Angle-Angle Rate Safety QP Setup
def initializeSafetyFilter(seg_est, seg_true, pd):
    """
    Initialize CBFs for the true and estimated system.
    
    Inputs:
      seg_est - Segway nominal parametric model
      seg_true - Segway true parametric model
      pd - PD controller
    
    Outputs:
      safety_est- CBF for seg_est
      safety_true - CBF for seg_true
      flt_est - CBF-QP filter for seg_est
      flt_True - CBF-QP filter for seg_true
    """
    
    theta_e = 0.1383
    angle_max = 0.2617 
    coeff = 1

    safety_est = SafetyAngleAngleRate( seg_est, theta_e, angle_max, coeff )
    safety_true = SafetyAngleAngleRate( seg_true, theta_e, angle_max, coeff)
    alpha = 10
    comp_safety = lambda r: alpha * r
    phi_0_est = lambda x, t: safety_est.drift( x, t ) + comp_safety( safety_est.eval( x, t ) )
    phi_1_est = lambda x, t: safety_est.act( x, t )
    phi_0_true = lambda x, t: safety_true.drift( x, t ) + comp_safety( safety_true.eval( x, t ) )
    phi_1_true = lambda x, t: safety_true.act( x, t )

    flt_est = FilterController( seg_est, phi_0_est, phi_1_est, pd )
    flt_true = FilterController( seg_true, phi_0_true, phi_1_true, pd)
    
    return safety_est, safety_true, flt_est, flt_true