from tkinter import Y
import numpy as np
import matplotlib.pyplot as pp
import pandas as pd
import heapq
import pycbc.waveform
from pycbc.types import FrequencySeries
from pycbc.filter import match
from pycbc.psd import aLIGOZeroDetHighPower
import matplotlib as mpl
import time
import multiprocessing
from multiprocessing import Pool, freeze_support
from joblib import Parallel, delayed
from mpmath import *
import csv
#from joblib.externals.loky import set_loky_pickler
#set_loky_pickler("dill")

pd.set_option('display.float_format', lambda x: '%.3e' % x)
plotdirName = "/Users/saifali/Desktop/gwlensing/plots/"
solar_mass = 4.92624076 * 10**-6 #[solar_mass] = sec
giga_parsec = 1.02927125 * 10**17 #[giga_parsec] = sec

def unlensed_waveform(**args):
    
    theta_s = args['theta_s']
    phi_s = args['phi_s']
    theta_l = args['theta_l']
    phi_l = args['phi_l']
    mcz = args['mcz']
    dist = args['dist']
    eta = args['eta']
    tc = args['tc']
    phi_c = args['phi_c']
    flow = args['f_lower']
    
    df = args['delta_f']
    
    def limit(mcz, eta):
        low_limit = 20
        f_cut = 1 / (np.power(6, 3/2) * np.pi * ((mcz) / (np.power(eta, 3/5))))
        return low_limit, f_cut
    
    f = np.arange(flow, limit(mcz, eta)[1], df)
    #print(limit(mcz, eta)[1])
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
    
    signal_I = FrequencySeries(signal_I, delta_f = df)
    return signal_I, f

pycbc.waveform.add_custom_waveform('unlensed', unlensed_waveform, 'frequency', force=True)
hf, freq_arr = pycbc.waveform.get_fd_waveform(approximant="unlensed",
                                        theta_s = 0, phi_s = 0, theta_l = 0, phi_l = 0, 
                                        mcz = 20 * solar_mass, dist = 1 * giga_parsec, eta = 0.25,
                                        tc = 0, phi_c = 0,
                                        delta_f=1/4, f_lower = 20)

def Sn(f):
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
    
psd_analytical = np.zeros_like(freq_arr)
for i in range(len(freq_arr)):
    psd_analytical[i] = Sn(freq_arr[i])
psd_analytical = FrequencySeries(psd_analytical, delta_f = 1/4)

def amp_fact_sis_wo(y, ML, f = freq_arr):
        '''computes the amplification factor for source SIS lens.
        Parameters
        ----------
        f : array
            frequency
        y : float
            source position
        Return
        ----------
        F_val : array, complex
            Amplification factor for SIS
        '''
        F_val_sis = np.zeros_like(f, dtype = np.complex128)
        for i in range(len(f)):
            w = 8 * pi * ML * f[i]
            pre_factor = exp(1j * (w / 2) * (y**2 + 2 * (y + 0.5)))
            func = lambda n: (gamma(1 + n / 2) / fac(n)) * (2 * w * exp(1j * 3 * (pi / 2))) ** (n / 2) * hyp1f1(1 + n / 2, 1, -1j * (w / 2) * y ** 2)
            series_sum = nsum(func, [0, inf])

            F_val_sis[i] = np.complex128(pre_factor * series_sum, dtype = np.complex128)

        return F_val_sis

def pycbc_match(i, y, ML, return_dict = None, hf = hf):
    hf_lensed_source = hf * amp_fact_sis_wo(y, ML)
    pycbc_match = match(hf_lensed_source, hf, psd=psd_analytical, low_frequency_cutoff=20)[0]
    return_dict[i] = [y, ML, pycbc_match]
    #return pycbc_match


if __name__ == "__main__":
    
    num_pts = 20
    y_range_temp = np.logspace(np.log10(0.1), np.log(1), num_pts)
    ML_range_temp = np.logspace(np.log10(1e2 * solar_mass), np.log10(1e4 * solar_mass), num_pts)
    y_mesh, ML_mesh = np.meshgrid(y_range_temp, ML_range_temp)
    y_range = y_mesh.flatten()
    ML_range = ML_mesh.flatten()

    start = time.strftime("%H%M%S")
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    processes = []

    datPath = "/Users/saifali/Desktop/gwlensing/data/"
    for i in range(len(y_range)):
        print(i)
        process = multiprocessing.Process(target = pycbc_match, args=(i, y_range[i], ML_range[i], return_dict))
        processes.append(process)
        process.start()
    
    for proc in processes:
        proc.join()
        
    w = csv.writer(open(datPath + "pycbc_match_sis.csv", "w"))
    for key, value in return_dict.items():
        #print(return_dict.items())
        w.writerow([key, value])

    print(f'start time: {start}')
    
    
    

