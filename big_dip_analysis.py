import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

solar_mass = 4.92624076 * 10**-6 #[solar_mass] = sec

plotdirName = "/Users/saifali/Desktop/gwlensing/plots/"

file_list = ["pycbc_1d_match_pm_y_0pt2.csv",
            "pycbc_1d_match_pm_y_0pt3.csv",
            "pycbc_1d_match_pm_y_0pt4.csv"]
y_list = [0.2, 0.3, 0.4]

for i in range(len(file_list)):
    data_1d_y_0pt2 = pd.read_csv('/Users/saifali/Desktop/gwlensing/data/'+str(file_list[i]), header = None, converters = {1: eval})
    data_1d_y_0pt2['ML'] = np.array([data_1d_y_0pt2[1][i][1] for i in range(len(data_1d_y_0pt2))])
    data_1d_y_0pt2['match'] = np.array([data_1d_y_0pt2[1][i][2] for i in range(len(data_1d_y_0pt2))])
    data_1d_y_0pt2 = data_1d_y_0pt2.sort_values(by = 0)
    plt.plot(data_1d_y_0pt2['ML'] / solar_mass, 1 - data_1d_y_0pt2['match'], label = f'y = {y_list[i]}')
    plt.legend()
# fig, ax1 = plt.subplots(1, 1, figsize = (10, 8), sharey = True, gridspec_kw={'wspace': .05})


# ax1.tick_params(axis='both', which='major', labelsize=25)
# ax1.set_xlabel(r'$M_L$ $[M_\odot]$', fontsize = 25)
# ax1.set_ylabel(r'$\epsilon$', fontsize = 35)
# ax1.grid()
plt.show()