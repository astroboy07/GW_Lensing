import subprocess
import numpy as np
from numpy.lib.utils import source
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.3e' % x)
import time

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
    from matplotlib.ticker import MaxNLocator

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

    ein_rad = get_einstein_radius(values)[0]

    rfile = 'out_crit.dat'
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
    
    theta_E_sis = 1.976582e-04
    

    rfile = 'out_point.dat'
    data = np.loadtxt(rfile, comments = '#')

    sxx = data[0, 2]
    syy = data[0, 3]
    ixx = data[1: ,0]
    iyy = data[1: ,1]

    plt.figure(figsize = (8, 16))
    
    plt.subplot(2, 1, 2)

    #edited
    xmin = -1.5
    xmax = 1.5
    ymin = -1.5
    ymax = 1.5

    #plt.plot([ix3, ix4], [iy3, iy4], '-', color = 'grey', zorder = 0, lw = 0.5)
    plt.plot([ix1 / theta_E_sis, ix2 / theta_E_sis], [iy1 / theta_E_sis, iy2 / theta_E_sis], '-', color = 'blue', zorder = 1)

    plt.scatter(ixx / theta_E_sis, iyy / theta_E_sis, s = [71.5, 204.5, 546.0, 399.85], marker = '^', color = 'red', zorder = 2, facecolors='none')
    plt.grid()
    # plt.text(ixx[0] + 0.00002, iyy[0] + 0.00002, '(0.5)', color = 'red')
    # plt.text(ixx[1] + 0.00002, iyy[1] + 0.00002, '(0)', color = 'red')
    # plt.text(ixx[2] - 0.00008, iyy[2] + 0.00002, '(0)', color = 'red')
    # plt.text(ixx[3] + 0.00002, iyy[3] + 0.000007, '(0.5)', color = 'red')
    ax = plt.gca()
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])

    plt.title('image plane') 
    plt.xlabel('$\\xi_1$ [arcsec]')
    plt.ylabel('$\\xi_2$ [arcsec]')
    plt.subplots_adjust(top = 0.95)
    plt.subplots_adjust(bottom = 0.15)
    plt.subplots_adjust(left = 0.15)
    plt.gca().tick_params(axis='x', pad=15)

    plt.subplot(2, 1, 1)

    #edited
    xmin = -1.5
    xmax = 1.5
    ymin = -1.5
    ymax = 1.5

    #plt.plot([sx3, sx4], [sy3, sy4], '-', color = 'grey', zorder = 0, lw = 0.5)
    plt.plot([sx1 / theta_E_sis, sx2 / theta_E_sis], [sy1 / theta_E_sis, sy2 / theta_E_sis], '-', color = 'blue', zorder = 1)

    plt.scatter(sxx / theta_E_sis, syy / theta_E_sis, s = 30, marker = 'o', color = 'red', zorder = 2)
    plt.grid()

    ax = plt.gca()
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])

    plt.title('source plane') 
    plt.xlabel('$\\eta_1$ [arcsec]')
    plt.ylabel('$\\eta_2$ [arcsec]')
    plt.subplots_adjust(top = 0.95)
    plt.subplots_adjust(bottom = 0.15)
    plt.subplots_adjust(left = 0.15)
    plt.gca().tick_params(axis='x', pad=15)

    ofile = 'plot_point_' + str(i)

    #plt.savefig(plotdirName + ofile + '.png', dpi = 150)
    #plt.show()
    plt.savefig(plotdirName + ofile + '.pdf', bbox_inches = 'tight', dpi = 300)
    #plt.savefig(plotdirName + ofile + '.eps', bbox_inches = 'tight', dpi = 300)
    #print(ixx, iyy)

# GET THE RADIAL DISTANCES FOR THE CAUSTICS AT DIFFERENT POLAR ANGLES
def radial_distance_caustics(values, theta):
    
    df_radial_distance = pd.DataFrame(columns=('radius', 'images_num'))

    source_x_range = np.linspace(0.0, 8.0e-4, 300)
    source_y_range = np.linspace(0.0, 8.0e-4, 300)
    images_num = np.zeros_like(source_x_range)
    radius = np.zeros_like(source_x_range)
    '''
    if theta == int(theta):
        for i in range(len(source_x_range)):
            print(theta)
            values = initial_values
            values['source_x'] = source_x_range[i]
            radius[i] = source_x_range[i]
            images_num[i] = run_glafic(values)[0, 0]
            print(radius[i], images_num[i])
            df_radial_distance.loc[i] = [radius[i], images_num[i]]

    elif theta == int(theta):
        for i in range(len(source_x_range)):
            print(theta)
            values = initial_values
            values['source_x'] = source_x_range[i]
            values['source_y'] = source_y_range[i]
            radius[i] = np.sqrt(values['source_x'] ** 2 + values['source_y'] ** 2)
            images_num[i] = run_glafic(values)[0, 0]
            df_radial_distance.loc[i] = [radius[i], images_num[i]]

    elif theta == int(theta):
        for i in range(len(source_y_range)):
            print(theta)
            values = initial_values
            values['source_y'] = source_y_range[i]
            radius[i] = source_y_range[i]
            images_num[i] = run_glafic(values)[0, 0]
            df_radial_distance.loc[i] = [radius[i], images_num[i]]
    
    else:
        for i in range(len(source_x_range)):
            print(theta)
            values = initial_values
            values['source_x'] = source_x_range[i] 
            values['source_y'] = source_x_range[i] * np.tan(float(theta) * (np.pi / 180))
            radius[i] = np.sqrt(values['source_x'] ** 2 + values['source_y'] ** 2)
            images_num[i] = run_glafic(values)[0, 0]
            df_radial_distance.loc[i] = [radius[i], images_num[i]]
    '''
    if theta == 90.0:
        for i in range(len(source_y_range)):
            #print(theta)
            values = initial_values
            values['source_x'] = 0
            values['source_y'] = source_y_range[i]
            radius[i] = source_y_range[i]
            images_num[i] = run_glafic(values)[0, 0]
            #print(radius[i], images_num[i])
            df_radial_distance.loc[i] = [radius[i], images_num[i]]
            
    elif theta > 86 and theta < 90:
        source_x_range = np.linspace(0.0, 5.0e-5, 300)
        source_y_range = np.linspace(0.0, 5.0e-5, 300)
        images_num = np.zeros_like(source_x_range)
        radius = np.zeros_like(source_x_range)
        for i in range(len(source_y_range)):
            #print(theta)
            values = initial_values
            values['source_x'] = source_x_range[i]
            values['source_y'] = source_x_range[i] * np.tan(float(theta) * (np.pi / 180))
            #print(f"source_x = {values['source_x']} and source_y = {values['source_y']}")
            radius[i] = np.sqrt(values['source_x'] ** 2 + values['source_y'] ** 2)
            #print(run_glafic(values))
            images_num[i] = run_glafic(values)[0, 0]
            # print(values['source_x'], values['source_y'], images_num[i])
            df_radial_distance.loc[i] = [radius[i], images_num[i]]
            
    else:
        for i in range(len(source_x_range)):
            values = initial_values
            values['source_x'] = source_x_range[i] 
            values['source_y'] = source_x_range[i] * np.tan(float(theta) * (np.pi / 180))
            radius[i] = np.sqrt(values['source_x'] ** 2 + values['source_y'] ** 2)
            images_num[i] = run_glafic(values)[0, 0]
            #print(values['source_x'], values['source_y'], images_num[i])
            df_radial_distance.loc[i] = [radius[i], images_num[i]]
    
    caustic_in = []
    caustic_out = []
    df_get_radius_caustic_in = df_radial_distance.loc[df_radial_distance['images_num'] > 2.0]
    caustic_in.append(df_get_radius_caustic_in.iloc[-1]['radius'])

    df_get_radius_caustic_out = df_radial_distance.loc[df_radial_distance['images_num'] > 1.0]
    caustic_out.append(df_get_radius_caustic_out.iloc[-1]['radius'])

    # print(theta, caustic_in, caustic_out)
    # df_get_radial_distance = pd.DataFrame(columns=('caustic_in', 'caustic_out'))
    # df_get_radial_distance.loc[theta] = [caustic_in[0], caustic_out[0]]

    return theta, caustic_in[0], caustic_out[0]

    #print(np.c_[radius, images_num])

    #return df_radial_distance.to_csv(datadirName + "radial_distance_caustics_e=0.6_" + str(theta) + ".csv", index = False)

# GET THE EINSTEIN RADIUS AND MASS INSIDE THE EINSTEIN RADIUS
def get_einstein_radius(values):

    run = run_glafic(values = values)
    data_einstein = 'out_ein2.dat'
    output_einstein = np.loadtxt(data_einstein)
    
    # return einstein radius (arcseconds) and mass inside einstein radius (solar_mass / h)
    return np.array([output_einstein[-2], output_einstein[-1]])

################################################################################################################################################################

initial_values = {'lens_z':0.5, 
                'lens_sigma': 4, 
                'lens_x': 0.0, 
                'lens_y': 0.0, 
                'lens_ellip': 0.2, 
                'lens_theta': 0.0,
                'lens_r_core': 0.0,
                'source_z': 1.0,
                'source_x': 0., 
                'source_y': 0.
                }

#print(run_glafic(initial_values))
# print(magnifications(values = initial_values))
#radial_distance_caustics(values = initial_values, theta = 90)
# print(f'Einstein radius and mass inside it:{get_einstein_radius(values = initial_values)}')
# plots(initial_values, 3)

'''
start = time.time()
theta_range = np.linspace(0, 90, 60)
df_get_radial_distance = pd.DataFrame(columns=('theta', 'caustic_in', 'caustic_out'))
for i in range(len(theta_range)):
    theta, inner_caustic, outer_caustic = radial_distance_caustics(initial_values, theta_range[i])
    print(i, theta, inner_caustic, outer_caustic)
    df_get_radial_distance.loc[i] = [theta, inner_caustic, outer_caustic]
df_get_radial_distance.to_csv(datadirName + "radial_distance_caustics_e=0.4_final.csv", index = False)
end = time.time()

print(f'time elapsed {(end - start) / 60}')
'''

# FOR TWO IMAGES SYSTEM

'''
def my_lin(lb, ub, steps, spacing = 3):
    span = (ub-lb)
    dx = 1.0 / (steps-1)
    return np.array([lb + (i * dx) ** spacing * span for i in range(steps)])
'''

def two_images_analysis(angle, num_points) -> pd.DataFrame:

    df = pd.DataFrame(columns=('source_x', 'mu_1', 'mu_2', 'td_1', 'td_2'))
    if int(angle) != 90:
        theta = np.deg2rad(angle)
        source_x_range = np.logspace(np.log10(0.5e-5), np.log10(2.1e-4), num_points)
        source_y_range = source_x_range * np.tan(theta)
    else:
        source_y_range = np.logspace(np.log10(0.5e-5), np.log10(2.1e-4), num_points)
        source_x_range = np.zeros(num_points)

    for i in range(source_x_range.shape[0]):
        values = initial_values
        values['source_x'] = source_x_range[i]
        values['source_y'] = source_y_range[i]
        imgs_count = run_glafic(values)[0, 0]
        print(i, values['source_x'], values['source_y'], imgs_count)
        if int(imgs_count) == 2:
            df.loc[i] = [
                        np.sqrt(values['source_x']**2 + values['source_y']**2), magnifications(values)[0], magnifications(values)[1], 
                        magnifications(values)[2], magnifications(values)[3]
                        ]
        else:
            pass
    df.to_csv(datadirName + f"flux_twoimages_theta_{str(angle)}_sigma={str(initial_values['lens_sigma'])}.csv", index = False)
    print(df)

# call the two images analysis function   
# two_images_analysis(45.0, 200)


# FOR FOUR IMAGES SYSTEM

def check_negatives(arr):
    '''utility function defined to check if the generated four images 
    have 2 positive and 2 negative parity images (not required now!)
    '''
    negatives = []
    for number in arr:
        if number < 0:
            negatives += [number]
    return len(negatives)

def four_images_analysis(angle, num_points) -> pd.DataFrame:

    df = pd.DataFrame(columns=('source_x', 'mu_1', 'mu_2', 'mu_3', 'mu_4', 'td_1', 'td_2', 'td_3', 'td_4'))
    if int(angle) != 90:
        theta = np.deg2rad(angle)
        source_x_range = np.logspace(np.log10(0.02e-5), np.log10(0.8e-4), num_points)
        source_y_range = source_x_range * np.tan(theta)
    else:
        source_y_range = np.logspace(np.log10(0.02e-5), np.log10(0.8e-4), num_points)
        source_x_range = np.zeros(num_points)

    for i in range(source_x_range.shape[0]):
        values = initial_values
        values['source_x'] = source_x_range[i]
        values['source_y'] = source_y_range[i]
        imgs_count = run_glafic(values)[0, 0]
        print(i, values['source_x'], values['source_y'], imgs_count)
        if int(imgs_count) == 4:
            df.loc[i] = [
                        np.sqrt(values['source_x']**2 + values['source_y']**2), 
                        magnifications(values)[0], magnifications(values)[1], magnifications(values)[2], magnifications(values)[3],
                        magnifications(values)[4], magnifications(values)[5], magnifications(values)[6], magnifications(values)[7]
                        ]
        else:
            pass
    df.to_csv(datadirName + f"flux_fourimages_theta_{str(angle)}_sigma={str(initial_values['lens_sigma'])}.csv", index = False)
    print(df)

# call the four images analysis function   
# four_images_analysis(90.0, 100)


# Keeping the y constant and varying the polar angle theta for sigma = 6
'''
y_scaled = 0.04
ein_rad = get_einstein_radius(values = initial_values)[0]
r = y_scaled * ein_rad
print(r)
theta_range = np.linspace(np.pi * 0.001, np.pi * 0.499, 100)
df = pd.DataFrame(columns=('theta', 'mu_1', 'mu_2', 'mu_3', 'mu_4', 'td_1', 'td_2', 'td_3', 'td_4'))
for i in range(len(theta_range)):
    values = initial_values
    values['source_x'] = r * np.cos(theta_range[i])
    values['source_y'] = r * np.sin(theta_range[i])
    df.loc[i] = [theta_range[i] , magnifications(values)[0], magnifications(values)[1], magnifications(values)[2], magnifications(values)[3],
                magnifications(values)[4], magnifications(values)[5], magnifications(values)[6], magnifications(values)[7]] 
df.to_csv(datadirName + "flux_fourimages_y_0.04_sigma=6.csv", index = False)
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





