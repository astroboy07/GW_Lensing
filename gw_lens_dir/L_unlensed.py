#from gwsim.models import LISAunlensed
import numpy as np
import scipy
from scipy import integrate
import mpmath as mp

class Lunlensed():

    def __init__(self, params = None):

        self.params = params

        assert type(self.params == dict), "Parameters should be a dictionary"

        self.theta_s = params['theta_s']
        self.phi_s = params['phi_s']
        self.theta_l = params['theta_l']
        self.phi_l = params['phi_l']
        self.mcz = params['mcz']
        self.dist = params['dist']
        self.eta = params['eta']
        # Have to comment these two variables for overlap_imp_max
        #self.tc = params['tc']
        self.phi_c = params['phi_c'] 

        self.F_MIN = 20
        self.tc = params['tc']
        

    def mass_conv(self):
        """Converts chirp mass to total mass. M = mcz/eta^(3/5)
        """

        M_val = self.mcz/np.power(self.eta, 3/5)
        return M_val

    def l_dot_n(self):
        """TODO
        """

        cos_term = np.cos(self.theta_s) * np.cos(self.theta_l)
        sin_term = np.sin(self.theta_s) * np.sin(self.theta_l) * np.cos(self.phi_s - self.phi_l)

        inner_prod = cos_term + sin_term
        return inner_prod

    def amp(self):
        """TODO
        """

        amplitude = np.sqrt(5 / 96) * np.power(np.pi, -2 / 3) * np.power(self.mcz, 5 / 6) / (self.dist)
        return amplitude

    def psi(self, f):
        """eqn 3.13 in Cutler-Flanaghan 1994
        """

        front_terms = 2 * np.pi * f * self.tc - self.phi_c - np.pi / 4
        main_coeffs = 0.75 * np.power(8 * np.pi * self.mcz * f, -5 / 3)
        main_terms = (1 + 20 / 9 * (743 / 336 + 11 / 4 * self.eta) * np.power(np.pi * self.mass_conv() * f, 2 / 3)
                        - (16 * np.pi) * np.power(np.pi * self.mass_conv() * f, 1))

        psi_val = front_terms + main_coeffs * (main_terms)
        return psi_val

    def psi_s(self):

        numerator = np.cos(self.theta_l)-np.cos(self.theta_s)*(self.l_dot_n())
        denominator = np.sin(self.theta_s)*np.sin(self.theta_l)*np.sin(self.phi_l-self.phi_s)

        psi_s_val = np.arctan2(numerator, denominator)
        return psi_s_val


    def fIp(self):
        """TODO
        """

        term_1 = (1 / 2 * (1 + np.power(np.cos(self.theta_s), 2)) * np.cos(2*self.phi_s)* np.cos(2*self.psi_s()))
        term_2 = (np.cos(self.theta_s) * np.sin(2*self.phi_s)* np.sin(2*self.psi_s()))

        fIp_val = term_1 - term_2
        return fIp_val

    def fIc(self):
        """TODO
        """

        term_1 = (1 / 2 * (1 + np.power(np.cos(self.theta_s), 2)) * np.cos(2*self.phi_s)
                    * np.sin(2*self.psi_s()))
        term_2 = (np.cos(self.theta_s) * np.sin(2*self.phi_s)
                    * np.cos(2*self.psi_s()))

        fIc_val = term_1 + term_2
        return fIc_val

    def lambdaI(self):
        """TODO
        """

        term_1 = np.power(2 * self.l_dot_n() * self.fIc(), 2)
        term_2 = np.power((1 + np.power(self.l_dot_n(), 2)) * self.fIp(), 2)

        lambdaI_val = np.sqrt(term_1 + term_2)
        return lambdaI_val

    def phi_pI(self):
        """TODO
        """

        numerator = (2 * self.l_dot_n() * self.fIc())
        denominator = ((1 + np.power(self.l_dot_n(), 2)) * self.fIp())

        phi_pI_val = np.arctan2(numerator, denominator)
        return phi_pI_val

    def hI(self, f):
        """TODO
        """

        term_1 = self.lambdaI()
        term_2 = (np.exp(-1j * self.phi_pI()))
        term_3 = self.amp() * np.power(f, -7 / 6)
        term_4 = np.exp(1j * self.psi(f))

        signal_I = term_1 * term_2 * term_3 * term_4
        
        return signal_I

    """NOISE CURVE
    """

    def Sn(self, f):
        """From table 1 of arXiv:0903.0338. Changed from iLIGO to aLIGO.
        """
        fs = 20
        if f < fs:
            Sn_val = np.inf
        else:
            S0 = 1E-49
            f0 = 215
            Sn_temp = np.power(f/f0, -4.14) - 5 * np.power(f/f0, -2) + 111 * ((1 - np.power(f/f0, 2) + 0.5 * np.power(f/f0, 4)) / (1 + 0.5 * np.power(f/f0, 2)))
            Sn_val = Sn_temp * S0

        return Sn_val

    """SNR
    """

    def Snr(self):
        """eqn 31 of Takahashi/Nakamura.
        Note: Use this definition of snr for unlensed case. For lensed case, refer to the snr function in analysis directory.
        """
        # limit needs to be calculated properly
        integrand = lambda f: np.power(np.abs(self.hI(f)), 2)/self.Sn(f)
        lower_limit = 20 # Change low limit to 20 Hz for aLIGO. It's 40 Hz for iLIGO.
        upper_limit = 1/(np.power(6, 3/2) * np.pi * self.mass_conv())


        snr_temp, snr_temp_err = scipy.integrate.quad(integrand, lower_limit, upper_limit)
        snr_squared = 4 * snr_temp
        snr_val = np.sqrt(snr_squared)

        return snr_val

    """For Fisher matrix
    """
    def h_fisher(self, f, params):
        """TODO
        """

        self.params = params

        assert type(self.params == dict), "Parameters should be a dictionary"

        self.theta_s = params['theta_s']
        self.phi_s = params['phi_s']
        self.theta_l = params['theta_l']
        self.phi_l = params['phi_l']
        self.mcz = params['mcz']
        self.dist = params['dist']
        self.eta = params['eta']
        self.tc = params['tc']
        self.phi_c = params['phi_c']

        # This is the problem
        #self.tc = self.get_tc()

        return self.hI(f)
    
    def get_tc(self):
        """Solving eqn 3.10 of Cutler and Flanagan for t=0 at f=f_min=20 Hz
        """
        upper_limit = 1/(np.power(6, 3/2) * np.pi * self.mass_conv())
        F = upper_limit
        #F = 20
        x = np.power(np.pi * self.mass_conv() * F, 2 / 3)
        coeffs = 5 * np.power(8 * np.pi * F, -8 / 3) * np.power(self.mcz, -5 / 3)
        terms = 1 + 4 / 3 * (743 / 336 + (11 * self.eta) / 4) * x - (32 * np.pi) / 5 * np.power(x, 3 / 2)
        return self.tc - coeffs * terms, upper_limit