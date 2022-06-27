import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from matplotlib.ticker import StrMethodFormatter, NullFormatter
from matplotlib.ticker import MaxNLocator
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 25
mpl.rcParams['legend.handlelength'] = 3.0
mpl.rcParams['axes.linewidth'] = 1.0
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['xtick.major.width'] = 1.0
mpl.rcParams['xtick.minor.width'] = 1.0
mpl.rcParams['xtick.major.size'] = 5.0
mpl.rcParams['xtick.minor.size'] = 3.0
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['ytick.major.width'] = 1.0
mpl.rcParams['ytick.minor.width'] = 1.0
mpl.rcParams['ytick.major.size'] = 5.0
mpl.rcParams['ytick.minor.size'] = 3.0
mpl.rcParams['ytick.right'] = True

plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = '\\usepackage{sfmath}'

solar_mass = 4.92624076 * 10**-6 #[solar_mass] = sec

plotdirName = "/Users/saifali/Desktop/gwlensing/plots/"

file_list = [
            "pycbc_1d_match_pm_y_0pt2.csv",
            "pycbc_1d_match_pm_y_0pt3.csv",
            "pycbc_1d_match_pm_y_0pt4.csv"
            ]
y_list = [0.2, 0.3, 0.4]

plt.figure(figsize = (10, 8))
for i in range(len(file_list)):
    data_1d_y_0pt2 = pd.read_csv('/Users/saifali/Desktop/gwlensing/data/'+str(file_list[i]), header = None, converters = {1: eval})
    data_1d_y_0pt2['ML'] = np.array([data_1d_y_0pt2[1][i][1] for i in range(len(data_1d_y_0pt2))])
    data_1d_y_0pt2['match'] = np.array([data_1d_y_0pt2[1][i][2] for i in range(len(data_1d_y_0pt2))])
    data_1d_y_0pt2 = data_1d_y_0pt2.sort_values(by = 0)
    plt.plot(data_1d_y_0pt2['ML'] / solar_mass, 1 - data_1d_y_0pt2['match'], label = f'y = {y_list[i]}')
    plt.legend()
# fig, ax1 = plt.subplots(1, 1, figsize = (10, 8), sharey = True, gridspec_kw={'wspace': .05})

# c = 'red'
# # compare troughs
# plt.scatter(.0261, 0.17, s=70, facecolors='none', edgecolors=c)
# plt.scatter(.0368, 0.2, s=70, facecolors='none', edgecolors=c)
# plt.scatter(.0475, 0.19, s=70, facecolors='none', edgecolors=c)

# # compare troughs and crests
# plt.scatter(.0319, 0.23, s=70, facecolors='none', edgecolors=c)
# plt.scatter(.0268, 0.2, s=70, facecolors='none', edgecolors=c, marker = '^')
# plt.scatter(.0416, 0.23, s=70, facecolors='none', edgecolors=c, marker = '^')

plt.xlabel(r'$M_L$', fontsize = 25)
plt.ylabel(r'$\epsilon$', fontsize = 25)
plt.tick_params(axis='both', which='major', labelsize=25)
plt.legend(fontsize = 25)
plt.grid()

plt.show()
# plt.savefig(plotdirName + 'bigdip_analysis_points.pdf', dpi = 500, bbox_inches = 'tight')