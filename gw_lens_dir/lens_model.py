# This script illustrates how to use lensingGW to solve a SIE lens model, assuming radians

import numpy as np
from lensinggw.utils.utils import param_processing # required for converting coordinates to radians
from lensinggw.solver.images import microimages, OneDeflector
from lensinggw.utils.utils import TimeDelay, magnifications, getMinMaxSaddle
from lensinggw.amplification_factor.amplification_factor import geometricalOpticsMagnification

class gwlens_class():

    def __init__(self, params = None):

        self.params = params

        assert type(self.params == dict)

        # coordinates, first define them in scaled units [x (radians) /thetaE_tot]
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

    def lens_model(self, e1 = 0.1, e2 = 0.1):
        """ Refer to https://lenstronomy.readthedocs.io/en/latest/lenstronomy.LensModel.html for the available
        lens model in lenstronomy package. 
        """
        lens_model_list = ['POINT_MASS']
        #kwargs_point_mass = {'center_x': self.eta()[0],'center_y': self.eta()[1], 'theta_E': self.thetaE(), 'e1': e1, 'e2': e2}
        kwargs_point_mass = {'center_x': self.eta()[0],'center_y': self.eta()[1], 'theta_E': self.thetaE()}
        kwargs_lens_list = [kwargs_point_mass] 
        solver_kwargs = {'SearchWindowMacro': 4*self.thetaE(), 'Verbose': False} # size of the first macromodel grid

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
