import os
import numpy as np
import warnings
import montepython.io_mp as io_mp
from montepython.likelihood_class import Likelihood
import scipy.constants as conts
from scipy import interpolate as itp
from scipy.interpolate import RectBivariateSpline

class baoonly_lyaxqso_sdss_dr16(Likelihood):

    # initialization routine

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        print('Including eBOSS Lya-QSO.')
        self.lyaxqso_data = np.loadtxt(os.path.join(self.data_directory, self.data_file))
        self.lyaxqso_DM = np.unique(self.lyaxqso_data[:, 0]) 
        self.lyaxqso_DH = np.unique(self.lyaxqso_data[:, 1]) 
        self.lyaxqso_lkl = np.reshape(self.lyaxqso_data[:, 2], [self.lyaxqso_DM.shape[0], self.lyaxqso_DH.shape[0]]) 
        self.lyaxqso_Interp = RectBivariateSpline(self.lyaxqso_DM, self.lyaxqso_DH, self.lyaxqso_lkl, kx=3, ky=3)
        # end of initialization
    # compute likelihood

    def loglkl(self, cosmo, data):
        loglkl = 0.0
        
        DM_at_z = cosmo.angular_distance(self.lyaxqso_dr16_z_eff) * (1. + self.lyaxqso_dr16_z_eff)
        H_at_z = cosmo.Hubble(self.lyaxqso_dr16_z_eff)
        DH_at_z = 1.0/H_at_z
        rd = cosmo.rs_drag() * self.rs_rescale

        theo_DM_at_z = DM_at_z / rd
        theo_DH_at_z_in_Mpc_inv = DH_at_z / rd
        # SANov21: Shouold there by a negative sign here? Should we divide by two as well?
        loglkl = np.log(float(self.lyaxqso_Interp(theo_DM_at_z, theo_DH_at_z_in_Mpc_inv)[0]))
        
        return loglkl/2.0
