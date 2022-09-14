import multiprocessing
from re import M
import time
import pickle
import csv
import numpy as np
import scipy.special as sc
import pandas as pd
import mpmath as mp
import scipy as sp
#from scipy.integrate import quad
from scipy.optimize import dual_annealing
from scipy.signal import find_peaks, find_peaks_cwt
from mpmath import *
mp.pretty = True 

#from gwsim.analysis.overlap_optimize import strain

class overlap_sis():

    solar_mass = 4.92624076 * 10**-6 #[solar_mass] = sec
    giga_parsec = 1.02927125 * 10**17 #[giga_parsec] = sec
    year = 31557600 #[year] = sec


    def __init__(self, params_source = None, params_temp = None):
        
        self.params_source = params_source
        self.params_temp = params_temp

        # Defining source parameters with (t0, phi0)
        self.theta_s_source = params_source['theta_s_source']
        self.phi_s_source = params_source['phi_s_source']
        self.theta_l_source = params_source['theta_l_source']
        self.phi_l_source = params_source['phi_l_source']
        self.mcz_source = params_source['mcz_source']
        self.dist_source = params_source['dist_source']
        self.eta_source = params_source['eta_source']
        self.t0 = params_source['t0']
        self.phi_0 = params_source['phi_0']
        # Adding the lensing params for source
        self.M_lz_source = params_source['M_lz_source']
        self.y_source = params_source['y_source']

        # Defining template parameters 
        self.theta_s_temp = params_temp['theta_s_temp']
        self.phi_s_temp = params_temp['phi_s_temp']
        self.theta_l_temp = params_temp['theta_l_temp']
        self.phi_l_temp = params_temp['phi_l_temp']
        self.mcz_temp = params_temp['mcz_temp']
        self.dist_temp = params_temp['dist_temp']
        self.eta_temp = params_temp['eta_temp']
        #self.tc = params_temp['tc']
        #self.phi_c = params_temp['phi_c']
        
        

    def limit(self, params_source, params_template):
    
        low_limit = 20
        f_cut_source = 1 / (np.power(6, 3/2) * np.pi * ((self.mcz_source) / (np.power(self.eta_source, 3/5))))
        f_cut_temp = 1 / (np.power(6, 3/2) * np.pi * ((self.mcz_temp) / (np.power(self.eta_temp, 3/5))))

        if f_cut_temp < f_cut_source:
            upper_limit = f_cut_temp

        else:
            upper_limit = f_cut_source
        
        return low_limit, upper_limit, f_cut_source, f_cut_temp

    def mag(self, lens = 'sis'):
    
        if lens == 'pm':
            mu_plus = np.abs(0.5 + (self.y_source ** 2 + 2) / (2 * self.y_source * (self.y_source ** 2 + 4) ** 0.5))
            mu_minus = np.abs(0.5 - (self.y_source ** 2 + 2) / (2 * self.y_source * (self.y_source ** 2 + 4) ** 0.5))
        
        elif lens == 'sis':
            mu_plus = np.abs(1 + 1 / self.y_source)
            mu_minus = np.abs(-1 + 1 / self.y_source)
        
        return mu_minus / mu_plus, mu_plus, mu_minus

    def time_del(self, lens = 'sis'):
    
        if lens == 'pm':
            first_term = (self.y_source * (self.y_source ** 2 + 4) ** 0.5) / 2
            second_term = np.log(((self.y_source ** 2 + 4) ** 0.5 + self.y_source) / ((self.y_source ** 2 + 4) ** 0.5 - self.y_source))
            tds = 4 * self.M_lz_source * (first_term + second_term)
        
        elif lens == 'sis':
            tds = 8 * self.M_lz_source * self.y_source
        
        return tds

    def strain(self, f, theta_s, phi_s, theta_l, phi_l, mcz, dist, eta, tc, phi_c):
        """
        This file is just functionized form of L_unlensed(which was "object-oriented"). This was mainly created
        for the optimization of overlap function.
        """
        def mass_conv(mcz, eta):
            """Converts chirp mass to total mass. M = mcz/eta^(3/5)
            """

            M_val = mcz/np.power(eta, 3/5)
            return M_val

        def l_dot_n(theta_s, theta_l, phi_s, phi_l):
            """TODO
            """

            cos_term = np.cos(theta_s) * np.cos(theta_l)
            sin_term = np.sin(theta_s) * np.sin(theta_l) * np.cos(phi_s - phi_l)

            inner_prod = cos_term + sin_term
            return inner_prod

        def amp(mcz, dist):
            """TODO
            """

            amplitude = np.sqrt(5 / 96) * np.power(np.pi, -2 / 3) * np.power(mcz, 5 / 6) / (dist)
            return amplitude

        def psi(f, tc, phi_c, mcz, eta):
            """eqn 3.13 in Cutler-Flanaghan 1994
            """

            front_terms = 2 * np.pi * f * tc - phi_c - np.pi / 4
            main_coeffs = 0.75 * np.power(8 * np.pi * mcz * f, -5 / 3)
            main_terms = (1 + 20 / 9 * (743 / 336 + 11 / 4 * eta) * np.power(np.pi * mass_conv(mcz, eta) * f, 2 / 3)
                            - (16 * np.pi) * np.power(np.pi * mass_conv(mcz, eta) * f, 1))

            psi_val = front_terms + main_coeffs * (main_terms)
            return psi_val

        def psi_s(theta_s, theta_l, phi_s, phi_l):

            numerator = np.cos(theta_l)-np.cos(theta_s)*(l_dot_n(theta_s, theta_l, phi_s, phi_l))
            denominator = np.sin(theta_s)*np.sin(theta_l)*np.sin(phi_l-phi_s)

            psi_s_val = np.arctan2(numerator, denominator)
            return psi_s_val


        def fIp(theta_s, phi_s):
            """TODO
            """

            term_1 = (1 / 2 * (1 + np.power(np.cos(theta_s), 2)) * np.cos(2*phi_s)* np.cos(2*psi_s(theta_s, theta_l, phi_s, phi_l)))
            term_2 = (np.cos(theta_s) * np.sin(2*phi_s)* np.sin(2*psi_s(theta_s, theta_l, phi_s, phi_l)))

            fIp_val = term_1 - term_2
            return fIp_val

        def fIc(theta_s, phi_s):
            """TODO
            """

            term_1 = (1 / 2 * (1 + np.power(np.cos(theta_s), 2)) * np.cos(2*phi_s)
                        * np.sin(2*psi_s(theta_s, theta_l, phi_s, phi_l)))
            term_2 = (np.cos(theta_s) * np.sin(2*phi_s)
                        * np.cos(2*psi_s(theta_s, theta_l, phi_s, phi_l)))

            fIc_val = term_1 + term_2
            return fIc_val

        def lambdaI():
            """TODO
            """

            term_1 = np.power(2 * l_dot_n(theta_s, theta_l, phi_s, phi_l) * fIc(theta_s, phi_s), 2)
            term_2 = np.power((1 + np.power(l_dot_n(theta_s, theta_l, phi_s, phi_l), 2)) * fIp(theta_s, phi_s), 2)

            lambdaI_val = np.sqrt(term_1 + term_2)
            return lambdaI_val

        def phi_pI():
            """TODO
            """

            numerator = (2 * l_dot_n(theta_s, theta_l, phi_s, phi_l) * fIc(theta_s, phi_s))
            denominator = ((1 + np.power(l_dot_n(theta_s, theta_l, phi_s, phi_l), 2)) * fIp(theta_s, phi_s))

            phi_pI_val = np.arctan2(numerator, denominator)
            return phi_pI_val

        term_1 = lambdaI()
        term_2 = (np.exp(-1j * phi_pI()))
        term_3 = amp(mcz, dist) * np.power(f, -7 / 6)
        term_4 = np.exp(1j * psi(f, tc, phi_c, mcz, eta))

        signal_I = term_1 * term_2 * term_3 * term_4
        return signal_I

    '''Adding in the lens
    '''

    '''Amplification factor in geometrical optics limit
    '''
    def F_source_sis(self, f):
        '''computes the amplification factor for source SIS lens.
        Parameters
        ----------
        f : float
            frequency
        y : float
            source position
        Return
        ----------
        F_val : float, complex
            Amplification factor for point mass
        '''
        w = 8 * pi * self.M_lz_source * f
        pre_factor = exp(1j * (w / 2) * (self.y_source**2 + 2 * (self.y_source + 0.5)))
        func = lambda n: (gamma(1 + n / 2) / fac(n)) * (2 * w * exp(1j * 3 * (pi / 2))) ** (n / 2) * hyp1f1(1 + n / 2, 1, -1j * (w / 2) * self.y_source ** 2)
        series_sum = nsum(func, [0, inf])

        F_val_source_sis = np.complex128(pre_factor * series_sum, dtype = np.complex128)

        return F_val_source_sis

    def F_source_pm(self, f):
        
        w = 8 * np.pi * self.M_lz_source * f
        x_m = 0.5 * (self.y_source + np.sqrt(self.y_source**2 + 4))
        phi_m = np.power((x_m - self.y_source) , 2) / 2 - np.log(x_m)

        first_term = np.exp(np.pi * w / 4 + 1j * (w / 2) * (np.log(w / 2) - 2 * phi_m)) 
        second_term = sc.gamma(1 - 1j * (w / 2)) 
        third_term = mp.hyp1f1(1j * w / 2, 1, 1j * (w / 2) * (self.y_source**2), maxterms = 10**6)

        F_val_pm_temp = first_term * second_term * third_term
        F_val_pm = np.complex128(F_val_pm_temp, dtype = np.complex128)

        return F_val_pm
    
    def F_geo_source_sis(self, f):
        '''computes the amplification factor for source SIS lens in the geometrical optics limit.

        Parameters
        ----------
        f : float
            frequency

        y : float
            source position

        Return
        ----------
        F_val : float, complex
            Amplification factor for point mass
        '''
        if self.y_source <= 1:
            self.flux_ratio = self.mag()[0]
            self.td = self.time_del()
            F_geo_val_source_pm = 1 - 1j * np.sqrt(self.flux_ratio) * np.exp(2 * np.pi * 1j * f * self.td)
        else:
            F_geo_val_source_pm = self.mag()[1]

        return F_geo_val_source_pm

    def F_geo_source_pm(self, f):
        '''computes the amplification factor for source point mass lens in the geometrical optics limit.

        Parameters
        ----------
        f : float
            frequency

        y : float
            source position

        Return
        ----------
        F_val : float, complex
            Amplification factor for point mass
        '''
        self.flux_ratio = 0.6
        self.td = 0.0265
        F_geo_val_source_pm = 1 - 1j * np.sqrt(self.flux_ratio) * np.exp(2 * np.pi * 1j * f * self.td)

        return F_geo_val_source_pm    

    def Sn(self, f):
        """ ALIGO noise curve from arXiv:0903.0338
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
    '''
     def get_f_transition(self):
        """ Calculate the frequency (f_transition) after which geometrical optics is valid
        """
        freq_range = my_lin(0.001, 250, 500)

        # Calculate the amplification factor for a range of frequency
        Fsis_series = np.zeros_like(freq_range)
        for i in range(len(freq_range)):
            Fsis_series[i] = np.abs(self.F_source_sis(freq_range[i]))

        # Get the peaks (specifically the first one)    
        peaks, _ = find_peaks(Fsis_series) 
        
        return freq_range[peaks][0]
    '''

    def signal_source_wo(self, f, t_c, phi_c): # Remove t_c and phi_c for SNR calculation
        
        hI_source = self.strain(
            f, 
            self.theta_s_source,
            self.phi_s_source, 
            self.theta_l_source,
            self.phi_l_source,
            self.mcz_source,
            self.dist_source,
            self.eta_source,
            self.t0,
            self.phi_0
        )
        
        amp_factor_source_pm = self.F_source_pm(f)

        return hI_source * amp_factor_source_pm
    
    def signal_source_go(self, f, t_c, phi_c): 
        
        hI_source = self.strain(
            f, 
            self.theta_s_source,
            self.phi_s_source, 
            self.theta_l_source,
            self.phi_l_source,
            self.mcz_source,
            self.dist_source,
            self.eta_source,
            self.t0,
            self.phi_0
        )
        
        amp_factor_source_pm = self.F_geo_source_pm(f)

        return hI_source * amp_factor_source_pm

    def signal_temp(self, f, t_c, phi_c): 

        hI_temp = self.strain(
            f, 
            self.theta_s_temp,
            self.phi_s_temp, 
            self.theta_l_temp,
            self.phi_l_temp,
            self.mcz_temp,
            self.dist_temp,
            self.eta_temp,
            t_c,
            phi_c
        )

        return hI_temp 

    def integrand_num_wo(self, f, t_c, phi_c):
        integrand_num_wo = self.signal_source_wo(f, t_c, phi_c) * np.conjugate(self.signal_temp(f, t_c, phi_c)) / self.Sn(f)
        return integrand_num_wo
    #################################
    def integrand_num_wo_real(self, f, t_c, phi_c):
        integrand_num_wo = self.signal_source_wo(f, t_c, phi_c) * np.conjugate(self.signal_temp(f, t_c, phi_c)) / self.Sn(f)
        return np.real(integrand_num_wo)
    
    def integrand_num_wo_imag(self, f, t_c, phi_c):
        integrand_num_wo = self.signal_source_wo(f, t_c, phi_c) * np.conjugate(self.signal_temp(f, t_c, phi_c)) / self.Sn(f)
        return np.imag(integrand_num_wo)
    #################################
    def integrand_num_go(self, f, t_c, phi_c):
        integrand_num_go = self.signal_source_go(f, t_c, phi_c) * np.conjugate(self.signal_temp(f, t_c, phi_c)) / self.Sn(f)
        return integrand_num_go
    
    def integrand_deno_1_wo(self, f, t_c, phi_c):
        integrand_deno_1_wo = self.signal_source_wo(f, t_c, phi_c) * np.conjugate(self.signal_source_wo(f, t_c, phi_c)) / self.Sn(f)
        return integrand_deno_1_wo
    
    def integrand_deno_1_go(self, f, t_c, phi_c):
        integrand_deno_1_go = self.signal_source_go(f, t_c, phi_c) * np.conjugate(self.signal_source_go(f, t_c, phi_c)) / self.Sn(f)
        return integrand_deno_1_go
    
    def integrand_deno_2(self, f, t_c, phi_c):
        integrand_deno_2 = self.signal_temp(f, t_c, phi_c) * np.conjugate(self.signal_temp(f, t_c, phi_c)) / self.Sn(f)
        return integrand_deno_2

    def overlap(self, x):

        t_c = x[0]
        phi_c = x[1]
    
        num_temp, num_err = sp.integrate.quad(
            self.integrand_num_wo, 
            self.limit(self.params_source, self.params_temp)[0], 
            self.limit(self.params_source, self.params_temp)[1], 
            args = (t_c, phi_c)
        )

        deno_temp_1, deno_temp_err_1 = sp.integrate.quad(
            self.integrand_deno_1_wo,
            self.limit(self.params_source, self.params_temp)[0], 
            self.limit(self.params_source, self.params_temp)[2], 
            args = (t_c, phi_c)
        )

        deno_temp_2, deno_temp_err_2 = sp.integrate.quad(
            self.integrand_deno_2,
            self.limit(self.params_source, self.params_temp)[0], 
            self.limit(self.params_source, self.params_temp)[3], 
            args = (t_c, phi_c)
        )
        num = 4 * np.real(num_temp)
        deno = np.sqrt((4 * np.real(deno_temp_1)) * (4 * np.real(deno_temp_2)))
        overlap_temp = num / deno
        #print(f"time delay is {self.time_del()} so wave optics")
        
        return -1 * overlap_temp
    
    def overlap_num(self, x):
        t_c = x[0]
        phi_c = x[1]

        num_temp_real, num_err_real = sp.integrate.quad(
            self.integrand_num_wo_real, 
            self.limit(self.params_source, self.params_temp)[0], 
            self.limit(self.params_source, self.params_temp)[1], 
            args = (t_c, phi_c)
        )

        num_temp_imag, num_err_imag = sp.integrate.quad(
            self.integrand_num_wo_imag, 
            self.limit(self.params_source, self.params_temp)[0], 
            self.limit(self.params_source, self.params_temp)[1], 
            args = (t_c, phi_c)
        )

        return num_temp_real + 1j * num_temp_imag

"""
Trying the multiprocessing 
"""

solar_mass = 4.92624076 * 10**-6 #[solar_mass] = sec
giga_parsec = 1.02927125 * 10**17 #[giga_parsec] = sec
year = 31557600 #[year] = sec


initial_params_source = {
    'theta_s_source' : 0.0, 
    'phi_s_source' : 0.0, 
    'theta_l_source' : 0.0, 
    'phi_l_source' : 0.0, 
    'mcz_source' : 20 * solar_mass, 
    'dist_source': 1 * giga_parsec, 
    'eta_source' : 0.25, 
    't0' : 0.0, 
    'phi_0' : 0.0,
    'M_lz_source': 2253 * solar_mass,
    'y_source': 0.2
}

initial_params_template = {
    'theta_s_temp' : 0.0, 
    'phi_s_temp' : 0.0, 
    'theta_l_temp' : 0.0, 
    'phi_l_temp' : 0.0, 
    'mcz_temp' : 20 * solar_mass, 
    'dist_temp': 1 * giga_parsec, 
    'eta_temp' : 0.25, 
    #'tc' : 0.0, 
    #'phi_c' : 0.0,
}

# Custom array generator for uneven spaced lens mass. This is required in order to get more data points at the critical point. 
def my_lin(lb, ub, steps, spacing = 3):
    span = (ub-lb)
    dx = 1.0 / (steps-1)
    return np.array([lb + (i * dx) ** spacing * span for i in range(steps)])
    
if __name__ == "__main__":

    datPath = "/Users/saifali/Desktop/gwlensing/data/"
    start = time.time()

    y_source_range = np.array([0.2, 0.3, 0.4])
    M_lz_source_range = np.array([2260 * solar_mass, 1510 * solar_mass, 1119 * solar_mass])
    #M_lz_source_range = my_lin(1e2 * solar_mass, 5e4 * solar_mass, 3)
    #I_range = np.linspace(0.1, 1, 20)
    df_res = pd.DataFrame(columns=('M_lz', 'overlap', 'tc', 'phi_c'))

    for i in range(len(M_lz_source_range)):

        print(f"working lens mass is {M_lz_source_range[i] / solar_mass}")

        params_source = initial_params_source
        params_template = initial_params_template
        params_source['M_lz_source'] = M_lz_source_range[i]
        params_source['y_source'] = y_source_range[i] 
        bnds = [[-0.2, 0.2], [-np.pi, np.pi]]
        overlap_optimized = overlap_sis(params_source = params_source, params_temp = initial_params_template)
        overlap_max = dual_annealing(overlap_optimized.overlap, bounds = bnds, maxiter = 80)
        #df_res.loc[i] = [y_source_range[i], M_lz_source_range[i], np.abs(overlap_max.fun), overlap_max.x[0], overlap_max.x[1]]
        end = time.time()
        print(M_lz_source_range[i], np.abs(overlap_max.fun), overlap_max.x[0], overlap_max.x[1])
        print(f'elapsed time: {(end - start)/60}')
    #print(df_res)
    #df_res.to_csv(datPath + "overlap_pm_bigdip_troughs.csv", index = False)
    
