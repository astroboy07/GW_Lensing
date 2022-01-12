import subprocess
import numpy as np
import pandas as pd


def magnifications(values):

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
    print(output)

    '''
    if output[0,0] == 2.0:
        return output[1, 2], output[2, 2]
    else:
        print("Not two image system")
    '''
    
    
    
    
initial_values = {'lens_z':0.5, 
                'lens_sigma': 300.0, 
                'lens_x': 0.0, 
                'lens_y': 0.0, 
                'lens_ellip': 0.2, 
                'lens_theta': 0.0,
                'lens_r_core': 0.0,
                'source_z': 1.0,
                'source_x': 0.9,
                'source_y': 0.
                }

print(magnifications(values = initial_values))

