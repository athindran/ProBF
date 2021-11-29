from numpy import array, cos, sin, tanh
from core.dynamics import RoboticDynamics

class Segway(RoboticDynamics):
    """Planar Segway system. State is [x, theta, x_dot, theta_dot], where x is
    the position of the Segway base in m, x_dot is the velocity in m / sec,
    theta is the angle of the frame in rad clockwise from upright, and
    theta_dot is the angular rate in rad / sec. The input is [u], where u is
    positive or negative percent of maximum motor voltage.
    Attributes:
    x_ddot drift component, f3: float * float * float -> float
    theta_ddot drift component, f4: float * float * float -> float
    x_ddot actuation component, g3: float -> float
    theta_ddot actuation component, g4: float -> float
    """

    def __init__(self, m_b=44.798, m_w=2.485, J_w=0.055936595310797, a_2=-0.023227187592750, c_2=0.166845864363019, B_2=2.899458828344427, R= 0.086985141514373, K=0.141344665167821, r=0.195, g=9.81, f_d=0.076067344020759, f_v=0.002862586216301, V_nom=57):
        
        """Initialize a SegwaySystem object.
        Inputs:
        Mass of frame (kg), m_b: float
        Mass of one wheel (kg), m_w: float
        Inertia of wheel (kg*m^2), J_w: float
        x position of frame (m), a_2: float
        z position of frame (m), c_2: float
        yy inertia of frame (kg*m^2), B_2: float
        Electrical resistance of motors (Ohm), R: float
        Torque constant of motors (N*m/A), K: float
        Radius of wheels (m), r: float
        Gravity constant (m/s^2), g: float
        Dry friction coefficient (N*m), f_d: float
        Viscous friction coefficient (N*m*s), f_v: float
        Nominal battery voltage (V), V_nom: float
        """
        RoboticDynamics.__init__(self, 2, 1)
        self.params = m_b, m_w, J_w, a_2, c_2, B_2, R, K, r, g, f_d, f_v, V_nom
        
        self.f_3 = lambda x_dot, theta, theta_dot: (1/2) * R ** (-1) * (4 * \
        B_2 * J_w + 4 * a_2 ** 2 * J_w * m_b + 4 * c_2 ** 2 * J_w * m_b + 2 * \
        B_2 * m_b * r ** 2 + a_2 ** 2 * m_b ** 2 * r ** 2 + c_2 ** 2 * m_b ** \
        2 * r ** 2 + 4 * B_2 * m_w * r ** 2 + 4 * a_2 ** 2 * m_b * m_w * r ** \
        2 + 4 * c_2 ** 2 * m_b * m_w * r ** 2 + (a_2 ** 2 + (-1) * c_2 ** 2) * \
        m_b ** 2 * r ** 2 * cos(2 * theta) + 2 * a_2 * c_2 * m_b ** 2 * r ** 2 \
        * sin(2 * theta)) ** (-1) * (800 * B_2 * K ** 2 * theta_dot * r + 800 \
        * a_2 ** 2 * K ** 2 * m_b * theta_dot * r + 800 * c_2 ** 2 * K ** 2 * \
        m_b * theta_dot * r + 800 * B_2 * f_v * theta_dot * r * R + 800 * a_2 \
        ** 2 * f_v * m_b * theta_dot * r * R + 800 * c_2 ** 2 * f_v * m_b * \
        theta_dot * r * R + (-800) * B_2 * K ** 2 * x_dot + (-800) * a_2 ** 2 \
        * K ** 2 * m_b * x_dot + (-800) * c_2 ** 2 * K ** 2 * m_b * x_dot + \
        (-800) * B_2 * f_v * R * x_dot + (-800) * a_2 ** 2 * f_v * m_b * R * \
        x_dot + (-800) * c_2 ** 2 * f_v * m_b * R * x_dot + 80 * c_2 * K ** 2 \
        * m_b * theta_dot * r ** 2 * cos(theta) + 80 * c_2 * f_v * m_b * \
        theta_dot * r ** 2 * R * cos(theta) + 4 * a_2 * B_2 * m_b * theta_dot \
        ** 2 * r ** 2 * R * cos(theta) + 4 * a_2 ** 3 * m_b ** 2 * theta_dot \
        ** 2 * r ** 2 * R * cos(theta) + 4 * a_2 * c_2 ** 2 * m_b ** 2 * \
        theta_dot ** 2 * r ** 2 * R * cos(theta) + (-80) * c_2 * K ** 2 * m_b \
        * r * x_dot * cos(theta) + (-80) * c_2 * f_v * m_b * r * R * x_dot * \
        cos(theta) + (-4) * a_2 * c_2 * g * m_b ** 2 * r ** 2 * R * cos(2 * \
        theta) + (-80) * a_2 * K ** 2 * m_b * theta_dot * r ** 2 * sin(theta) \
        + (-80) * a_2 * f_v * m_b * theta_dot * r ** 2 * R * sin(theta) + 4 * \
        B_2 * c_2 * m_b * theta_dot ** 2 * r ** 2 * R * sin(theta) + 4 * a_2 \
        ** 2 * c_2 * m_b ** 2 * theta_dot ** 2 * r ** 2 * R * sin(theta) + 4 * \
        c_2 ** 3 * m_b ** 2 * theta_dot ** 2 * r ** 2 * R * sin(theta) + 80 * \
        a_2 * K ** 2 * m_b * r * x_dot * sin(theta) + 80 * a_2 * f_v * m_b * r \
        * R * x_dot * sin(theta) + 2 * a_2 ** 2 * g * m_b ** 2 * r ** 2 * R * \
        sin(2 * theta) + (-2) * c_2 ** 2 * g * m_b ** 2 * r ** 2 * R * sin(2 * \
        theta) + 4 * f_d * r * R * (10 * (B_2 + (a_2 ** 2 + c_2 ** 2) * m_b) + \
        c_2 * m_b * r * cos(theta) + (-1) * a_2 * m_b * r * sin(theta)) * \
        tanh(500 * r ** (-1) * (2 * theta_dot * r + (-2) * x_dot)) + (-4) * \
        f_d * r * R * (10 * (B_2 + (a_2 ** 2 + c_2 ** 2) * m_b) + c_2 * m_b * \
        r * cos(theta) + (-1) * a_2 * m_b * r * sin(theta)) * tanh(500 * r ** \
        (-1) * ((-2) * theta_dot * r + 2 * x_dot)))

        self.f_4 = lambda x_dot, theta, theta_dot: r ** (-1) * R ** (-1) * (4 * \
        B_2 * J_w + 4 * a_2 ** 2 * J_w * m_b + 4 * c_2 ** 2 * J_w * m_b + 2 *
        B_2 * m_b * r ** 2 + a_2 ** 2 * m_b ** 2 * r ** 2 + c_2 ** 2 * m_b ** \
        2 * r ** 2 + 4 * B_2 * m_w * r ** 2 + 4 * a_2 ** 2 * m_b * m_w * r ** \
        2 + 4 * c_2 ** 2 * m_b * m_w * r ** 2 + (a_2 ** 2 + (-1) * c_2 ** 2) * \
        m_b ** 2 * r ** 2 * cos(2 * theta) + 2 * a_2 * c_2 * m_b ** 2 * r ** \
        2 * sin(2 * theta)) ** (-1) * ((-80) * J_w * K ** 2 * theta_dot * r + \
        (-40) * K ** 2 * m_b * theta_dot * r ** 3 + (-80) * K ** 2 * m_w * \
        theta_dot * r ** 3 + (-80) * f_v * J_w * theta_dot * r * R + (-40) * \
        f_v * m_b * theta_dot * r ** 3 * R + (-80) * f_v * m_w * theta_dot * \
        r ** 3 * R + 80 * J_w * K ** 2 * x_dot + 40 * K ** 2 * m_b * r ** 2 * \
        x_dot + 80 * K ** 2 * m_w * r ** 2 * x_dot + 80 * f_v * J_w * R * \
        x_dot + 40 * f_v * m_b * r ** 2 * R * x_dot + 80 * f_v * m_w * r ** 2 \
        * R * x_dot + (-400) * c_2 * K ** 2 * m_b * theta_dot * r ** 2 * \
        cos(theta) + 4 * a_2 * g * J_w * m_b * r * R * cos(theta) + (-400) * \
        c_2 * f_v * m_b * theta_dot * r ** 2 * R * cos(theta) + 2 * a_2 * g * \
        m_b ** 2 * r ** 3 * R * cos(theta) + 4 * a_2 * g * m_b * m_w * r ** 3 \
        * R * cos(theta) + 400 * c_2 * K ** 2 * m_b * r * x_dot * cos(theta) + \
        400 * c_2 * f_v * m_b * r * R * x_dot * cos(theta) + (-2) * a_2 * c_2 \
        * m_b ** 2 * theta_dot ** 2 * r ** 3 * R * cos(2 * theta) + 400 * a_2 \
        * K ** 2 * m_b * theta_dot * r ** 2 * sin(theta) + 4 * c_2 * g * J_w * \
        m_b * r * R * sin(theta) + 400 * a_2 * f_v * m_b * theta_dot * r ** 2 \
        * R * sin(theta) + 2 * c_2 * g * m_b ** 2 * r ** 3 * R * sin(theta) + \
        4 * c_2 * g * m_b * m_w * r ** 3 * R * sin(theta) + (-400) * a_2 * K \
        ** 2 * m_b * r * x_dot * sin(theta) + (-400) * a_2 * f_v * m_b * r * \
        R * x_dot * sin(theta) + a_2 ** 2 * m_b ** 2 * theta_dot ** 2 * r ** 3 \
        * R * sin(2 * theta) + (-1) * c_2 ** 2 * m_b ** 2 * theta_dot ** 2 * r \
        ** 3 * R * sin(2 * theta) + (-2) * f_d * r * R * (2 * J_w + m_b * r ** \
        2 + 2 * m_w * r ** 2 + 10 * c_2 * m_b * r * cos(theta) + (-10) * a_2 * \
        m_b * r * sin(theta)) * tanh(500 * r ** (-1) * (2 * theta_dot * r + \
        (-2) * x_dot)) + 2 * f_d * r * R * (2 * J_w + m_b * r ** 2 + 2 * m_w * \
        r ** 2 + 10 * c_2 * m_b * r * cos(theta) + (-10) * a_2 * m_b * r * \
        sin(theta)) * tanh(500 * r ** (-1) * ((-2) * theta_dot * r + 2 * \
        x_dot)))

        self.g_3 = lambda theta: (-2) * K * r * R ** (-1) * V_nom * ((-10) * \
        (B_2 + (a_2 ** 2 + c_2 ** 2) * m_b) + (-1) * c_2 * m_b * r * \
        cos(theta) + a_2 * m_b * r * sin(theta)) * (2 * B_2 * J_w + 2 * a_2 ** \
        2 * J_w * m_b + 2 * c_2 ** 2 * J_w * m_b + B_2 * m_b * r ** 2 + a_2 ** \
        2 * m_b ** 2 * r ** 2 + c_2 ** 2 * m_b ** 2 * r ** 2 + 2 * B_2 * m_w * \
        r ** 2 + 2 * a_2 ** 2 * m_b * m_w * r ** 2 + 2 * c_2 ** 2 * m_b * m_w \
        * r ** 2 + (-1) * c_2 ** 2 * m_b ** 2 * r ** 2 * cos(theta) ** 2 + \
        (-1) * a_2 ** 2 * m_b ** 2 * r ** 2 * sin(theta) ** 2 + a_2 * c_2 * \
        m_b ** 2 * r ** 2 * sin(2 * theta)) ** (-1)

        self.g_4 = lambda theta: (-4) * K * R ** (-1) * V_nom * (2 * J_w + m_b \
        * r ** 2 + 2 * m_w * r ** 2 + 10 * c_2 * m_b * r * cos(theta) + (-10) \
        * a_2 * m_b * r * sin(theta)) * (4 * B_2 * J_w + 4 * a_2 ** 2 * J_w * \
        m_b + 4 * c_2 ** 2 * J_w * m_b + 2 * B_2 * m_b * r ** 2 + a_2 ** 2 * \
        m_b ** 2 * r ** 2 + c_2 ** 2 * m_b ** 2 * r ** 2 + 4 * B_2 * m_w * r \
        ** 2 + 4 * a_2 ** 2 * m_b * m_w * r ** 2 + 4 * c_2 ** 2 * m_b * m_w * \
        r ** 2 + (a_2 ** 2 + (-1) * c_2 ** 2) * m_b ** 2 * r ** 2 * cos(2 * \
        theta) + 2 * a_2 * c_2 * m_b ** 2 * r ** 2 * sin(2 * theta)) ** (-1)

    def drift(self, x, t):
        _, theta, x_dot, theta_dot = x
        return array([x_dot, theta_dot, self.f_3(x_dot, theta, theta_dot), self.f_4(x_dot, theta, theta_dot)])

    def act(self, x, t):
        _, theta, _, _ = x
        return array([[0], [0], [self.g_3(theta)], [self.g_4(theta)]])