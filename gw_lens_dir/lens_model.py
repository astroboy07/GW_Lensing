# This script illustrates how to use lensingGW to solve a SIE lens model, assuming radians

import numpy as np
from lensinggw.utils.utils import param_processing # required for converting coordinates to radians
from lensinggw.solver.images import microimages, OneDeflector
from lensinggw.utils.utils import TimeDelay, magnifications, getMinMaxSaddle
from lensinggw.amplification_factor.amplification_factor import geometricalOpticsMagnification
from lensinggw.postprocess.postprocess import plot_images


plot_path = '/Users/saifali/Desktop/gwlensing/plots/'
class gwlens_class():

    def __init__(self, params = None):

        self.params = params

        assert type(self.params == dict)

        # coordinates, first define them in scaled units [x (radians) / thetaE]
        # source coordinate
        self.y0 = params['y0'] 
        self.y1 = params['y1']
        # lens coordinate
        self.l0 = params['l0']
        self.l1 = params['l1']
        # source redshit
        self.zs = params['zs']
        # lens redshift
        self.zl = params['zl']
        # lens mass
        self.ml = params['ml']

    # get the Einstein radius
    def thetaE(self):

        return param_processing(self.zl, self.zs, self.ml) 

    # convert the coordinates to radians
    def beta(self):

        return np.array([self.y0 * self.thetaE(), self.y1 * self.thetaE()])
    
    def eta(self):

        return np.array([self.l0 * self.thetaE(), self.l1 * self.thetaE()])

    def lens_model(self, e1 = 0.11, e2 = 0):
        """ Refer to https://lenstronomy.readthedocs.io/en/latest/lenstronomy.LensModel.html for the available
        lens model in lenstronomy package. 
        """
        lens_model_list = ['SIE']
        kwargs_lens_model = {'center_x': self.eta()[0],'center_y': self.eta()[1], 'theta_E': self.thetaE(), 'e1': e1, 'e2': e2}
        #kwargs_point_mass = {'center_x': self.eta()[0],'center_y': self.eta()[1], 'theta_E': self.thetaE()}
        kwargs_lens_list = [kwargs_lens_model] 
        solver_kwargs = {'SearchWindowMacro': 4 * self.thetaE(), # size of the first macromodel grid
                        'Verbose': False,
                        'OnlyMacro': True, 
                        #'PixelsMacro': 10**4,
                        #'Optimization': True
                        } 

        if len(lens_model_list) == 1:
            Img_ra, Img_dec, pixel_width = OneDeflector(source_pos_x = self.beta()[0], 
                                                        source_pos_y = self.beta()[1], 
                                                        lens_model_list = lens_model_list, 
                                                        kwargs_lens = kwargs_lens_list, 
                                                        **solver_kwargs)

            return Img_ra, Img_dec, pixel_width, lens_model_list, kwargs_lens_list
        else:
            Img_ra, Img_dec, MacroImg_ra, MacroImg_dec, pixel_width = microimages(source_pos_x = self.beta()[0], 
                                                                                    source_pos_y = self.beta()[1], 
                                                                                    lens_model_list = lens_model_list, 
                                                                                    kwargs_lens = kwargs_lens_list, 
                                                                                    **solver_kwargs)

            return Img_ra, Img_dec, MacroImg_ra, MacroImg_dec, pixel_width, lens_model_list, kwargs_lens_list
    
    def tds(self):
        """ computes the time delays (in seconds) of images
        """

        return TimeDelay(self.lens_model()[0], self.lens_model()[1], self.beta()[0], self.beta()[1], self.zl, self.zs, self.lens_model()[-2], self.lens_model()[-1])

    def mu(self):
        """ computes the magnification of images
        """
        return magnifications(self.lens_model()[0], self.lens_model()[1], self.lens_model()[-2], self.lens_model()[-1])

    def mi(self):
        """ computes the morse indices of images
        """
        return getMinMaxSaddle(self.lens_model()[0], self.lens_model()[1], self.lens_model()[-2], self.lens_model()[-1])

    def plot_img(self, plot_name, x_label = 'ra(rad)', y_label = 'dec(rad)'):
        """ plots images with magnification, time delays, caustics and critical curves
        """
        plot_images(output_folder = plot_path, 
                    file_name = plot_name,
                    source_pos_x = self.beta()[0],
                    source_pos_y = self.beta()[1],
                    lens_model_list = self.lens_model()[-2],
                    kwargs_lens_list = self.lens_model()[-1],
                    ImgRA = self.lens_model()[0],
                    ImgDEC = self.lens_model()[1],
                    Mu = self.mu(),
                    Td = self.tds(),
                    xlabel = x_label,
                    ylabel = y_label,
                    #mag_map = True,
                    #compute_window_x = 10**-12,
                    #compute_window_y = 10**-12
                    )                