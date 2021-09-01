# This script illustrates how to use lensingGW to solve a binary point mass lens model, assuming radians

import numpy as np
from lensinggw.postprocess.postprocess import plot_images

# coordinates, first define them in scaled units [x (radians) /thetaE_tot]
y0,y1 = 0.1, 0.5 * np.sqrt(3) 
l0,l1 = 0.,0.  

# redshifts                                                                                                                  
zS = 2.0 
zL = 0.5  

# masses 
mL1  = 100                                                                   

# convert to radians
from lensinggw.utils.utils import param_processing

thetaE = param_processing(zL, zS, mL1)                                                                                                                              
  

beta0,beta1 = y0*thetaE,y1*thetaE                                                 
eta10,eta11 = l0*thetaE,l1*thetaE                                                                                                
#eta20,eta21 = -l0*thetaE,l1*thetaE  

# lens model
lens_model_list     = ['POINT_MASS'] 
kwargs_point_mass_1 = {'center_x': eta10, 'center_y': eta11, 'theta_E': thetaE} 
#kwargs_point_mass_2 = {'center_x': eta20,'center_y': eta21, 'theta_E': thetaE2} 
kwargs_lens_list    = [kwargs_point_mass_1]   

# indicate the first lens as macromodel and solve with the two-step procedure
from lensinggw.solver.images import microimages

solver_kwargs = {'SearchWindowMacro': 4*thetaE}   

Img_ra, Img_dec, pixel_width  = microimages(source_pos_x    = beta0,
                                            source_pos_y    = beta1,
                                            lens_model_list = lens_model_list,
                                            kwargs_lens     = kwargs_lens_list,
                                            **solver_kwargs)                                                            
                                                                       
# time delays, magnifications, Morse indices and amplification factor
from lensinggw.utils.utils import TimeDelay, magnifications, getMinMaxSaddle
from lensinggw.amplification_factor.amplification_factor import geometricalOpticsMagnification

tds = TimeDelay(Img_ra, Img_dec,
                beta0, beta1,
                zL, zS,
                lens_model_list, kwargs_lens_list)                
mus = magnifications(Img_ra, Img_dec, lens_model_list, kwargs_lens_list)
ns  = getMinMaxSaddle(Img_ra, Img_dec, lens_model_list, kwargs_lens_list) 
                
print('Time delays (seconds): ', tds)
print('magnifications: ',  mus)
print('Morse indices: ',ns)

dummy_frequencies = np.linspace(0,100,100)
F = geometricalOpticsMagnification(dummy_frequencies,
                                   Img_ra,Img_dec,
                                   beta0,beta1,
                                   zL,zS,
                                   lens_model_list,
                                   kwargs_lens_list)

print('Geometrical optics amplification factor:', F)

plot_images(output_folder = '/Users/saifali/Desktop/gwlensing/plots/', 
            file_name = 'test_plot_point_mass',
            source_pos_x = beta0,
            source_pos_y = beta1,
            lens_model_list = lens_model_list,
            kwargs_lens_list = kwargs_lens_list,
            ImgRA = Img_ra,
            ImgDEC = Img_dec,
            Mu = mus,
            Td = tds,
            xlabel = 'ra(rad)',
            ylabel = 'dec(rad)'
            ) 