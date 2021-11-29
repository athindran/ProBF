from numpy import array, concatenate, cos, dot, reshape, sin, zeros

from core.dynamics import RoboticDynamics

class RoboticArm(RoboticDynamics):
    """
    Model used is from this paper.
    https://jacm.scu.ac.ir/article_16136_ad2eb488b37941d441b97f4e3e3ec071.pdf
    Gravity is removed.
    """
    def __init__(self, m_1, m_2, l_1, l_2, J_1, J_2, g=9.81):
        RoboticDynamics.__init__(self, 2, 2)
        self.params = m_1, m_2, l_1, l_2, J_1, J_2
    
    def D(self, q):
        m1, m2, l1, l2, J1, J2 = self.params
        q1, q2 = q
        return array([[(m1+m2)*l1*l1+m2*l2*l2+2*m2*l1*l2*cos(q2)+J1, m2*l2*l2+m2*l1*l2*cos(q2)], [m2*l2*l2+m2*l1*l2*cos(q2), m2 * (l2 ** 2)+J2]])
    
    def C(self, q, q_dot):
        m1, m2, l1, l2, _, _ = self.params
        q1, q2 = q
        q1dot, q2dot = q_dot
        C12 = m2*l1*l2*sin(q2)
        return array([[-2*C12*q2dot, -1*C12*(q1dot)], [C12*q1dot, 0]])
    
    #def U(self, q):
    #    _, m_p, l, g = self.params
    #    _, theta = q
    #    return m_p * g * l * cos(theta)
    
    def G(self, q):
        m1, m2, l1, l2, _, _ = self.params
        q1, q2 = q
        #g = 9.81
        g = 0
        return array([(m1+m2)*l1*cos(q2)*g+m2*l2*cos(q1+q2)*g, m2*l2*cos(q1+q2)*g])
    
    def B(self, q):
        return array([[1,0], [0,1]])
