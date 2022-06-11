from operator import index
import random
from hashlib import new
import subprocess
from unicodedata import name
import numpy as np
from numpy.lib.utils import source
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.float_format', lambda x: '%.3e' % x)


CORE_DIRNAME = "/Users/saifali/Desktop/gwlensing/"
GLAFIC_DIRNAME = "SIE_glafic/data/"
DAYS2SEC = 86400
PI = np.pi
#datadirName = "/Users/saifali/Desktop/gwlensing/SIE_glafic/data/"
#plotdirName = "/Users/saifali/Desktop/gwlensing/SIE_glafic/sie_plots/"



class glafic_trial():

    def __init__(self, glafic_params = None):
        
        self.lens_z = glafic_params['lens_z']
        self.lens_sigma = glafic_params['lens_sigma'] 
        self.lens_x = glafic_params['lens_x']
        self.lens_y = glafic_params['lens_y']
        self.lens_ellip = glafic_params['lens_ellip']
        self.lens_theta = glafic_params['lens_theta']
        self.lens_r_core = glafic_params['lens_r_core']
        self.source_z = glafic_params['source_z']
        self.source_x = glafic_params['source_x']
        self.source_y = glafic_params['source_y']
    
    def run_glafic(self):

        dat_file = 'out_point.dat'
        template_file = 'example.input'
        config_file = 'case.input'

        # Creates temporary .input file for glafic system
        with open(config_file, 'w') as case:
            # Copies template file except for flagged lines
            with open(template_file, 'r') as template:
                for line in template:
                    if "**SIE**" in line:
                        lens = f"lens sie {self.lens_z} {self.lens_sigma} {self.lens_x} {self.lens_y} {self.lens_ellip} {self.lens_theta} {self.lens_r_core} 0.0\n"
                        case.writelines(lens)
                    elif "**POINT**" in line:
                        point = f"point {self.source_z} {self.source_x} {self.source_y}\n"
                        case.writelines(point)
                    else:
                        case.writelines(line)
                template.close()
            case.close()
        

        run = subprocess.check_output(f"glafic {config_file} > /dev/null 2>&1", shell=True)

        output = np.loadtxt(dat_file)   # Loads dat file into numpy array
        #print(output)
        return output

    def magnifications(self):

        output = self.run_glafic()
        if output[0, 0] == 4.0:
            # mu_1, mu_2, mu_3, mu_4, td_1, td_2, td_3, td_4
            return output[1, 2], output[2, 2], output[3, 2], output[4, 2], output[1, 3], output[2, 3], output[3, 3], output[4, 3]
        else:
            # mu_1, mu_2, td_1, td_2
            return output[1, 2], output[2, 2], output[1, 3], output[2, 3]
    
    def get_einstein_radius(self):

        output = self.run_glafic()
        data_einstein = 'out_ein2.dat'
        output_einstein = np.loadtxt(data_einstein)
        
        # return einstein radius (arcseconds) and mass inside einstein radius (solar_mass / h)
        return np.array([output_einstein[-2], output_einstein[-1]])

    def sort_fourimgs(self):

        fourimgs = np.array(self.magnifications())
        print(f'Unsorted two/four images:\n{fourimgs}')
        if len(fourimgs) == 4:
            print('No four images!')
            return np.array([])
        else:
            old_td_ind = [0, 1, 2, 3]
            new_td_ind = [0, 0, 0, 0]
            sorted_arr = np.zeros(10)

            old_td_arr = np.array(fourimgs[-4:])
            new_td_arr = np.sort(old_td_arr)
            new_td_ind[0] = np.where(new_td_arr[0] == old_td_arr)[0][0]
            new_td_ind[1] = np.where(new_td_arr[1] == old_td_arr)[0][0]
            new_td_ind[2] = np.where(new_td_arr[2] == old_td_arr)[0][0]
            new_td_ind[3] = np.where(new_td_arr[3] == old_td_arr)[0][0]

            if len(new_td_ind) != len(set(new_td_ind)):
                dup_ind = [idx for idx, item in enumerate(new_td_ind) if item in new_td_ind[:idx]]
                non_common_ind = list(set(old_td_ind) - set(new_td_ind))[0]
                #print(non_common_ind)
                #print(dup_ind)
                #new_td_ind[i_row][dup_ind[0]] += 1
                new_td_ind[dup_ind[0]] = non_common_ind

            old_mu_arr = np.array(fourimgs[:4])
            new_mu_arr = np.array([old_mu_arr[new_td_ind[0]], old_mu_arr[new_td_ind[1]], old_mu_arr[new_td_ind[2]], old_mu_arr[new_td_ind[3]]])
            
            sorted_arr[0] = new_mu_arr[0]
            sorted_arr[1] = new_mu_arr[1]
            sorted_arr[2] = new_mu_arr[2]
            sorted_arr[3] = new_mu_arr[3]
            #sorted_arr[4] = new_td_arr[0] * DAYS2SEC
            sorted_arr[4] = new_td_arr[1] * DAYS2SEC
            sorted_arr[5] = new_td_arr[2] * DAYS2SEC
            sorted_arr[6] = new_td_arr[3] * DAYS2SEC
            sorted_arr[7] = (new_td_arr[2] - new_td_arr[1]) * DAYS2SEC
            sorted_arr[8] = (new_td_arr[3] - new_td_arr[1]) * DAYS2SEC
            sorted_arr[9] = (new_td_arr[3] - new_td_arr[2]) * DAYS2SEC

            print('Sorting is done! \n')
            return sorted_arr
    
    def geo_opt_fourimgs(self):
        
        sorted_fourimages = self.sort_fourimgs()
        print(sorted_fourimages)
        if len(sorted_fourimages) == 10:
            td_arr = np.concatenate((np.array([0]), sorted_fourimages[4:7]))
            sorted_go_fourimgs = np.zeros(8)
            imag_ind = [1, 2, 3, 4]
            td_dict = {0: '21', 1: '31', 2: '41', 3: '32', 4: '42', 5: '43'}

            wave_opt_td = []
            geo_opt_td = []
            td_row = np.array(sorted_fourimages[4:])
            ind = np.where(td_row < 0.05)[0]
            if len(ind) == 0: 
                #pass
                print('No image pairs in wave optics regime \n')
            else:
                for j in ind:
                    print(f'Image pairs in wave optics regime are: {td_dict[j]} \n')
                    #wave_opt_td[i] += [td_dict[j]]
                    wave_opt_td += [int(k) for k in "".join([td_dict[j]])]
                wave_opt_td = list(set(wave_opt_td))
            geo_opt_td = list(set(imag_ind) - set(wave_opt_td))
            
            for l in range(len(geo_opt_td)):
                sorted_go_fourimgs[geo_opt_td[l] - 1] = sorted_fourimages[geo_opt_td[l] - 1]
                sorted_go_fourimgs[geo_opt_td[l] + 3] = td_arr[geo_opt_td[l] - 1]
            print(f'Sorted four images in geo optics regime:\n{sorted_go_fourimgs}')
            return sorted_go_fourimgs
        else:
            print('No four images to sort and classify in geo optics!')
            return np.array([])
            

def go_fourimages_counts(num_pts = 6000):

    # parameters for glafic
    iniitial_glafic_params = {'lens_z':0.5, 
                'lens_sigma': 6, 
                'lens_x': 0.0, 
                'lens_y': 0.0, 
                'lens_ellip': 0.2, 
                'lens_theta': 0.0,
                'lens_r_core': 0.0,
                'source_z': 1.0,
                'source_x': 0.,
                'source_y': 0.
                }

    # Monte Carlo simulation
    source_x_min = 0.0
    source_y_min = 0.0

    ## for sigma = 6 km/s
    source_x_max = 0.9e-4
    source_y_max = 0.9e-4

    source_x_range = source_x_max - source_x_min
    source_y_range = source_y_max - source_y_min

    source_x = np.linspace(source_x_min, source_x_max, num_pts)
    source_y = np.linspace(source_y_min, source_y_max, num_pts)
    # source_xx, source_yy = np.meshgrid(source_x, source_y)
    # source_xx = source_xx.flatten()
    # source_yy = source_yy.flatten()
    data = pd.DataFrame(columns = ('source_x', 'source_y', 'mu_1', 'mu_2', 'mu_3', 'mu_4', 'td_1', 'td_2', 'td_3', 'td_4'))
    for i in range(source_x.shape[0]):
        glafic_params = iniitial_glafic_params
        source_x_rand = source_x_min + source_x_range * random.random()
        source_y_rand = source_y_min + source_y_range * random.random()

        glafic_params['source_x'] = source_x_rand
        glafic_params['source_y'] = source_y_rand
        glafic_init = glafic_trial(glafic_params)
        sorted_go_4imgs = glafic_init.geo_opt_fourimgs()
        #print(f'sorted go 4imgs: {sorted_go_4imgs}')
        if sorted_go_4imgs.shape[0] == 0:
            #data.loc[i] = [source_x_rand, source_y_rand, 0, 0, 0, 0, 0, 0, 0, 0]
            pass
        else:
            data.loc[i] = [glafic_params['source_x'], glafic_params['source_y'], sorted_go_4imgs[0], sorted_go_4imgs[1], sorted_go_4imgs[2], sorted_go_4imgs[3], sorted_go_4imgs[4], 
                        sorted_go_4imgs[5], sorted_go_4imgs[6], sorted_go_4imgs[7]]
    data.to_csv(CORE_DIRNAME + GLAFIC_DIRNAME + 'monte_carlo_go_4images.csv', index = False)
    print(data)

'''
# Call the class glafic_trial
glafic_params = {'lens_z':0.5, 
                'lens_sigma': 8, 
                'lens_x': 0.0, 
                'lens_y': 0.0, 
                'lens_ellip': 0.2, 
                'lens_theta': 0.0,
                'lens_r_core': 0.0,
                'source_z': 1.0,
                'source_x': 0.,
                'source_y': 0.
                }


glafic_init = glafic_trial(glafic_params)
print(glafic_init.run_glafic())
print(f'Einstein radius and mass inside it:{get_einstein_radius(values = initial_values)}')
# print(glafic_init.magnifications(), '\n')
# print(glafic_init.sort_fourimgs())
# print(glafic_init.geo_opt_fourimgs())
'''

# Call the function
go_fourimages_counts()