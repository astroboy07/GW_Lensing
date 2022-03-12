import subprocess
import numpy as np
from numpy.lib.utils import source
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.3e' % x)

datadirName = "/Users/saifali/Desktop/gwlensing/SIE_glafic/data/"
plotdirName = "/Users/saifali/Desktop/gwlensing/SIE_glafic/sie_plots/"
# RUN GLAFIC  
def run_glafic(values):

    v = values
    dat_file = 'out_point.dat'
    template_file = 'example.input'
    config_file = 'case.input'

    # Creates temporary .input file for glafic system
    with open(config_file, 'w') as case:
        # Copies template file except for flagged lines
        with open(template_file, 'r') as template:
            for line in template:
                if "**SIE**" in line:
                    lens = f"lens sie {v['lens_z']} {v['lens_sigma']} {v['lens_x']} {v['lens_y']} {v['lens_ellip']} {v['lens_theta']} {v['lens_r_core']} 0.0\n"
                    case.writelines(lens)
                elif "**POINT**" in line:
                    point = f"point {v['source_z']} {v['source_x']} {v['source_y']}\n"
                    case.writelines(point)
                else:
                    case.writelines(line)
            template.close()
        case.close()
    

    run = subprocess.check_output(f"glafic {config_file} > /dev/null 2>&1", shell=True)

    output = np.loadtxt(dat_file)   # Loads dat file into numpy array
    #print(output)
    return output

# GET THE MAGNIFICATION AND TIME DELAY FOR TWO IMAGES SYSTEM (YET)
def magnifications(values):

    output = run_glafic(values)
    if output[0, 0] == 4.0:
        # mu_1, mu_2, mu_3, mu_4, td_1, td_2, td_3, td_4
        return output[1, 2], output[2, 2], output[3, 2], output[4, 2], output[1, 3], output[2, 3], output[3, 3], output[4, 3]
    else:
        # mu_1, mu_2, td_1, td_2
        return output[1, 2], output[2, 2], output[1, 3], output[2, 3]
    
# PLOTTING FUNCTION
def plots(values, i):
    from matplotlib import markers
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
    from matplotlib.ticker import StrMethodFormatter, NullFormatter

    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['font.size'] = 20
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

    rfile = 'out_crit_'+ str(i)+'.dat'
    data = np.loadtxt(rfile, comments = '#')

    ix1 = data[: ,0]
    iy1 = data[: ,1]
    sx1 = data[: ,2]
    sy1 = data[: ,3]
    ix2 = data[: ,4]
    iy2 = data[: ,5]
    sx2 = data[: ,6]
    sy2 = data[: ,7]

    '''
    rfile = 'out_mesh.dat'
    data = np.loadtxt(rfile, comments = '#')

    ix3 = data[: ,0]
    iy3 = data[: ,1]
    sx3 = data[: ,2]
    sy3 = data[: ,3]
    ix4 = data[: ,4]
    iy4 = data[: ,5]
    sx4 = data[: ,6]
    sy4 = data[: ,7]
    '''
    

    rfile = 'out_point_' + str(i) + '.dat'
    data = np.loadtxt(rfile, comments = '#')

    sxx = data[0, 2]
    syy = data[0, 3]
    ixx = data[1: ,0]
    iyy = data[1: ,1]

    plt.figure(figsize = (8, 16))
    
    plt.subplot(2, 1, 1)

    #edited
    xmin = -5e-4
    xmax = 5e-4
    ymin = -5e-4
    ymax = 5e-4

    #plt.plot([ix3, ix4], [iy3, iy4], '-', color = 'grey', zorder = 0, lw = 0.5)
    plt.plot([ix1, ix2], [iy1, iy2], '-', color = 'blue', zorder = 1)

    plt.scatter(ixx, iyy, s = 30, marker = '^', color = 'red', zorder = 2)

    ax = plt.gca()
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])

    plt.title('image plane') 
    plt.xlabel('$\\theta_1$ [arcsec]')
    plt.ylabel('$\\theta_2$ [arcsec]')
    plt.subplots_adjust(top = 0.95)
    plt.subplots_adjust(bottom = 0.15)
    plt.subplots_adjust(left = 0.15)

    plt.subplot(2, 1, 2)

    #edited
    xmin = -5e-4
    xmax = 5e-4
    ymin = -5e-4
    ymax = 5e-4

    #plt.plot([sx3, sx4], [sy3, sy4], '-', color = 'grey', zorder = 0, lw = 0.5)
    plt.plot([sx1, sx2], [sy1, sy2], '-', color = 'blue', zorder = 1)

    plt.scatter(sxx, syy, s = 30, marker = '^', color = 'red', zorder = 2)

    ax = plt.gca()
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])

    plt.title('source plane') 
    plt.xlabel('$\\beta_1$ [arcsec]')
    plt.ylabel('$\\beta_2$ [arcsec]')
    plt.subplots_adjust(top = 0.95)
    plt.subplots_adjust(bottom = 0.15)
    plt.subplots_adjust(left = 0.15)

    ofile = 'plot_point_' + str(i)

    #plt.savefig(plotdirName + ofile + '.png', dpi = 150)
    plt.savefig(plotdirName + ofile + '.pdf', bbox_inches = 'tight')
    

# GET THE RADIAL DISTANCES FOR THE CAUSTICS AT DIFFERENT POLAR ANGLES
def radial_distance_caustics(values, theta):
    
    df_radial_distance = pd.DataFrame(columns=('radius', 'images_num'))

    source_x_range = np.linspace(0.0, 2.2e-4, 200)
    source_y_range = np.linspace(0.0, 2.2e-4, 200)
    images_num = np.zeros_like(source_x_range)
    radius = np.zeros_like(source_x_range)

    if theta == 0:
        for i in range(len(source_x_range)):
            values = initial_values
            values['source_x'] = source_x_range[i]
            radius[i] = source_x_range[i]
            images_num[i] = run_glafic(values)[0, 0]
            df_radial_distance.loc[i] = [radius[i], images_num[i]]

    elif theta == 45:
        for i in range(len(source_x_range)):
            values = initial_values
            values['source_x'] = source_x_range[i]
            values['source_y'] = source_y_range[i]
            radius[i] = np.sqrt(values['source_x'] ** 2 + values['source_y'] ** 2)
            images_num[i] = run_glafic(values)[0, 0]
            df_radial_distance.loc[i] = [radius[i], images_num[i]]

    elif theta == 90:
        for i in range(len(source_y_range)):
            values = initial_values
            values['source_y'] = source_y_range[i]
            radius[i] = source_y_range[i]
            images_num[i] = run_glafic(values)[0, 0]
            df_radial_distance.loc[i] = [radius[i], images_num[i]]
    
    else:
        for i in range(len(source_x_range)):
            values = initial_values
            values['source_x'] = source_x_range[i] 
            values['source_y'] = source_x_range[i] * np.tan(float(theta) * (np.pi / 180))
            radius[i] = np.sqrt(values['source_x'] ** 2 + values['source_y'] ** 2)
            images_num[i] = run_glafic(values)[0, 0]
            df_radial_distance.loc[i] = [radius[i], images_num[i]]
        

    print(np.c_[radius, images_num])

    return df_radial_distance.to_csv(datadirName + "radial_distance_caustics_" + str(theta) + ".csv", index = False)

# GET THE EINSTEIN RADIUS AND MASS INSIDE THE EINSTEIN RADIUS
def get_einstein_radius(values):

    run = run_glafic(values = values)
    data_einstein = 'out_ein2.dat'
    output_einstein = np.loadtxt(data_einstein)
    
    # return einstein radius (arcseconds) and mass inside einstein radius (solar_mass / h)
    return np.array([output_einstein[-2], output_einstein[-1]])


################################################################################################################################################################


initial_values = {'lens_z':0.5, 
                'lens_sigma': 6, 
                'lens_x': 0.0, 
                'lens_y': 0.0, 
                'lens_ellip': 0.1, 
                'lens_theta': 0.0,
                'lens_r_core': 0.0,
                'source_z': 1.0,
                'source_x': 0.,
                'source_y': 0.
                }

#print(run_glafic(initial_values))
#print(magnifications(values = initial_values))
#radial_distance_caustics(values = initial_values, theta = 90)
#print(f'Einstein radius and mass inside it:{get_einstein_radius(values = initial_values)}')
#plots(initial_values, 3)


# FOR TWO IMAGES SYSTEM

def my_lin(lb, ub, steps, spacing = 3):
    span = (ub-lb)
    dx = 1.0 / (steps-1)
    return np.array([lb + (i * dx) ** spacing * span for i in range(steps)])

'''
#source_x_range = np.linspace(0.16, 1.0, 15)
source_x_range = np.linspace(0.29e-4, 1.98e-4, 10)
df = pd.DataFrame(columns=('source_x', 'mu_1', 'mu_2', 'td_1', 'td_2'))
#df = pd.DataFrame(columns=('source_x', 'x_1', 'y_1', 'x_2', 'y_2', 'mu_1', 'mu_2', 'td_1', 'td_2'))

for i in range(len(source_x_range)):
    values = initial_values
    values['source_x'] = source_x_range[i]
    df.loc[i] = [source_x_range[i], magnifications(values)[0], magnifications(values)[1], magnifications(values)[2], magnifications(values)[3]]
    #plot = plots(values)
df.to_csv(datadirName + "flux_twoimages_theta_0.csv", index = False)
print(df)
'''

'''
source_y_range = np.linspace(0.33e-4, 1.85e-4, 10)
#source_y_range = my_lin(0.18, 0.9, 15)
df = pd.DataFrame(columns=('source_y', 'mu_1', 'mu_2', 'td_1', 'td_2'))
#df = pd.DataFrame(columns=('source_y', 'x_1', 'y_1', 'x_2', 'y_2', 'mu_1', 'mu_2', 'td_1', 'td_2'))
for i in range(len(source_y_range)):
    values = initial_values
    values['source_y'] = source_y_range[i]
    df.loc[i] = [source_y_range[i] , magnifications(values)[0], magnifications(values)[1], magnifications(values)[2], magnifications(values)[3]]
    #plot = plots(values)
df.to_csv(datadirName + "flux_twoimages_theta_90.csv", index = False)
print(df)
'''

'''
source_x_range = np.linspace(0.24e-4, 1.28e-4, 10)
source_y_range = np.linspace(0.24e-4, 1.28e-4, 10)
#source_x_range = my_lin(0.06, 0.7, 15)
#source_y_range = my_lin(0.06, 0.7, 15)
df = pd.DataFrame(columns=('source_x', 'mu_1', 'mu_2', 'td_1', 'td_2'))

for i in range(len(source_x_range)):
    values = initial_values
    values['source_x'] = source_x_range[i]
    values['source_y'] = source_y_range[i]
    df.loc[i] = [source_x_range[i] * np.sqrt(2), magnifications(values)[0], magnifications(values)[1], magnifications(values)[2], magnifications(values)[3]]
    #plot = plots(values)
df.to_csv(datadirName + "flux_twoimages_theta_45.csv", index = False)
print(df)
'''





# FOR FOUR IMAGES SYSTEM

'''
def my_lin(lb, ub, steps, spacing = 0.3):
    span = (ub-lb)
    dx = 1.0 / (steps-1)
    return np.array([lb + (i * dx) ** spacing * span for i in range(steps)])

'''

'''
source_x_range = np.linspace(0.1e-4, 0.59e-4, 10)
#source_x_range = my_lin(0.01, 0.15, 15)
df = pd.DataFrame(columns=('source_x', 'mu_1', 'mu_2', 'mu_3', 'mu_4', 'td_1', 'td_2', 'td_3', 'td_4'))

for i in range(len(source_x_range)):
    values = initial_values
    values['source_x'] = source_x_range[i]
    df.loc[i] = [source_x_range[i] , magnifications(values)[0], magnifications(values)[1], magnifications(values)[2], magnifications(values)[3], 
                 magnifications(values)[4], magnifications(values)[5], magnifications(values)[6], magnifications(values)[7]]
    

    # FOR FLIPPING IMAGES CASES (DONE MANUALLY)
    # if magnifications(values)[4] > magnifications(values)[5] and magnifications(values)[6] < magnifications(values)[4]:
    #     df.loc[i] = [source_x_range[i] , magnifications(values)[1], magnifications(values)[2], magnifications(values)[3], magnifications(values)[0], 
    #            magnifications(values)[5], magnifications(values)[6], magnifications(values)[7], magnifications(values)[4]]

    # if magnifications(values)[4] > magnifications(values)[5] and magnifications(values)[6] > magnifications(values)[4]:
    #     df.loc[i] = [source_x_range[i] , magnifications(values)[1], magnifications(values)[0], magnifications(values)[2], magnifications(values)[3], 
    #            magnifications(values)[5], magnifications(values)[4], magnifications(values)[6], magnifications(values)[7]]

    # elif magnifications(values)[7] == 0:
    #     df.loc[i] = [source_x_range[i] , magnifications(values)[3], magnifications(values)[1], magnifications(values)[0], magnifications(values)[2], 
    #            magnifications(values)[7], magnifications(values)[5], magnifications(values)[4], magnifications(values)[6]]
    # else:
    #     df.loc[i] = [source_x_range[i] , magnifications(values)[0], magnifications(values)[3], magnifications(values)[1], magnifications(values)[2], 
    #            magnifications(values)[4], magnifications(values)[7], magnifications(values)[5], magnifications(values)[6]]  
    
      
    #plot = plots(values)
df.to_csv(datadirName + "flux_fourimages_theta_0_sigma=6.csv", index = False)
print(df)
'''

'''
source_x_range = np.linspace(0.02e-4, 0.23e-4, 10)
source_y_range = np.linspace(0.02e-4, 0.23e-4, 10)
df = pd.DataFrame(columns=('source_x', 'mu_1', 'mu_2', 'mu_3', 'mu_4', 'td_1', 'td_2', 'td_3', 'td_4'))

for i in range(len(source_x_range)):
    values = initial_values
    values['source_x'] = source_x_range[i]
    values['source_y'] = source_y_range[i]
    df.loc[i] = [source_x_range[i] * np.sqrt(2) , magnifications(values)[0], magnifications(values)[1], magnifications(values)[2], magnifications(values)[3],
                magnifications(values)[4], magnifications(values)[5], magnifications(values)[6], magnifications(values)[7]]

    
    # if magnifications(values)[6] > magnifications(values)[7]:
    #     df.loc[i] = [source_x_range[i] * np.sqrt(2) , magnifications(values)[1], magnifications(values)[3], magnifications(values)[2], magnifications(values)[0],
    #                 magnifications(values)[5], magnifications(values)[7], magnifications(values)[6], magnifications(values)[4]]
    # elif magnifications(values)[6] > magnifications(values)[5] and magnifications(values)[6] > magnifications(values)[7]:
    #     df.loc[i] = [source_x_range[i] * np.sqrt(2), magnifications(values)[0], magnifications(values)[3], magnifications(values)[1], magnifications(values)[2],
    #                 magnifications(values)[4], magnifications(values)[7], magnifications(values)[5], magnifications(values)[6]]
    # else:
    #     df.loc[i] = [source_x_range[i] * np.sqrt(2), magnifications(values)[1], magnifications(values)[2], magnifications(values)[3], magnifications(values)[0],
    #                 magnifications(values)[5], magnifications(values)[6], magnifications(values)[7], magnifications(values)[4]]
    
        
    #plot = plots(values)
df.to_csv(datadirName + "flux_fourimages_theta_45_sigma=6.csv", index = False)
print(df)
'''

'''
source_y_range = np.linspace(0.1e-4, 0.7e-4, 10)
#source_y_range = my_lin(0.01, 0.17, 15)
df = pd.DataFrame(columns=('source_y', 'mu_1', 'mu_2', 'mu_3', 'mu_4', 'td_1', 'td_2', 'td_3', 'td_4'))
for i in range(len(source_y_range)):
    values = initial_values
    values['source_y'] = source_y_range[i]
    df.loc[i] = [source_y_range[i] , magnifications(values)[0], magnifications(values)[1], magnifications(values)[2], magnifications(values)[3],
                magnifications(values)[4], magnifications(values)[5], magnifications(values)[6], magnifications(values)[7]] 
    
    # FOR THE FLIPPING IMAGES CASE
    # if magnifications(values)[4] > magnifications(values)[5] and magnifications(values)[5] < magnifications(values)[6] and magnifications(values)[6] > magnifications(values)[7]:
    #     df.loc[i] = [source_y_range[i] , magnifications(values)[3], magnifications(values)[1], magnifications(values)[2], magnifications(values)[0],
    #                 magnifications(values)[7], magnifications(values)[5], magnifications(values)[6], magnifications(values)[4]]
    # elif magnifications(values)[4] > magnifications(values)[5] and magnifications(values)[5] > magnifications(values)[6]and magnifications(values)[6] > magnifications(values)[7]:
    #     df.loc[i] = [source_y_range[i] , magnifications(values)[3], magnifications(values)[2], magnifications(values)[1], magnifications(values)[0],
    #                 magnifications(values)[7], magnifications(values)[6], magnifications(values)[5], magnifications(values)[4]] 
    # elif magnifications(values)[4] > magnifications(values)[5] and magnifications(values)[5] > magnifications(values)[6] and magnifications(values)[6] < magnifications(values)[7] and magnifications(values)[5] > magnifications(values)[7]:
    #     df.loc[i] = [source_y_range[i] , magnifications(values)[2], magnifications(values)[3], magnifications(values)[1], magnifications(values)[0],
    #                 magnifications(values)[6], magnifications(values)[7], magnifications(values)[5], magnifications(values)[4]] 
    # elif magnifications(values)[4] > magnifications(values)[5] and magnifications(values)[5] < magnifications(values)[6] and magnifications(values)[6] < magnifications(values)[7]:
    #     df.loc[i] = [source_y_range[i] , magnifications(values)[1], magnifications(values)[2], magnifications(values)[3], magnifications(values)[0],
    #                 magnifications(values)[5], magnifications(values)[6], magnifications(values)[7], magnifications(values)[4]]
    # elif magnifications(values)[4] > magnifications(values)[5] and magnifications(values)[5] > magnifications(values)[6] and magnifications(values)[6] < magnifications(values)[7] and magnifications(values)[5] < magnifications(values)[7]:
    #     df.loc[i] = [source_y_range[i] , magnifications(values)[2], magnifications(values)[1], magnifications(values)[3], magnifications(values)[0],
    #                 magnifications(values)[6], magnifications(values)[5], magnifications(values)[7], magnifications(values)[4]]
    #plot = plots(values)
df.to_csv(datadirName + "flux_fourimages_theta_90_sigma=6.csv", index = False)
print(df)
'''


# Keeping the r constant and varying the polar angle theta for sigma = 6
'''
y_scaled = 0.08
ein_rad = get_einstein_radius(values = initial_values)[0]
r = y_scaled * ein_rad
theta_range = np.linspace(np.pi * 0.01, np.pi / 2.022, 10)
df = pd.DataFrame(columns=('theta', 'mu_1', 'mu_2', 'mu_3', 'mu_4', 'td_1', 'td_2', 'td_3', 'td_4'))
for i in range(len(theta_range)):
    values = initial_values
    values['source_x'] = r * np.cos(theta_range[i])
    values['source_y'] = r * np.sin(theta_range[i])
    df.loc[i] = [theta_range[i] , magnifications(values)[0], magnifications(values)[1], magnifications(values)[2], magnifications(values)[3],
                magnifications(values)[4], magnifications(values)[5], magnifications(values)[6], magnifications(values)[7]] 
df.to_csv(datadirName + "flux_fourimages_r_0.25e-4_sigma=6.csv", index = False)
print(df)

'''

# varying the parameter e for sigma = 6. fixed y and theta 
'''
y_scaled = 0.02
ein_rad = get_einstein_radius(values = initial_values)[0]
r = y_scaled * ein_rad
theta = 0
e_range = np.linspace(0.08, 0.8, 10)
df = pd.DataFrame(columns=('theta', 'mu_1', 'mu_2', 'mu_3', 'mu_4', 'td_1', 'td_2', 'td_3', 'td_4'))
for i in range(len(e_range)):
    values = initial_values
    values['source_x'] = r * np.cos(theta)
    values['source_y'] = r * np.sin(theta)
    values['lens_ellip'] = e_range[i]
    df.loc[i] = [e_range[i] , magnifications(values)[0], magnifications(values)[1], magnifications(values)[2], magnifications(values)[3],
                magnifications(values)[4], magnifications(values)[5], magnifications(values)[6], magnifications(values)[7]] 
df.to_csv(datadirName + "flux_fourimages_y_0.02_theta_0_sigma=6.csv", index = False)
print(df)
'''





