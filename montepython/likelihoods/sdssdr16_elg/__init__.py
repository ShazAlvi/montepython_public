import os
import numpy as np
import warnings
import montepython.io_mp as io_mp
from montepython.likelihood_class import Likelihood
import scipy.constants as conts
from scipy import interpolate as itp
from scipy.interpolate import RectBivariateSpline

class sdssdr16_elg(Likelihood):

    # initialization routine

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        # are there conflicting experiments?
        # SAlvi: How should this be changed?
        conflicting_experiments = [
            'bao', 'bao_boss', 'bao_known_rs'
            'bao_boss_aniso', 'bao_boss_aniso_gauss_approx']
        for experiment in conflicting_experiments:
            if experiment in data.experiments:
                raise io_mp.LikelihoodError(
                    'conflicting BAO measurments')
        # Read the datafile.
        print('Including eBOSS ELG.')
        self.elg_data = np.loadtxt(os.path.join(self.data_directory, self.data_file))
        self.elg_DV = self.elg_data[:, 0] 
        self.elg_lkl = self.elg_data[:, 1]
        # Create erd degree spline to interpolate later. 
        self.elg_DV_Interp = itp.splrep(self.elg_DV, self.elg_lkl)
        # end of initialization
    # compute likelihood

    def loglkl(self, cosmo, data):
        
        loglkl = 0.0
        rd = cosmo.rs_drag() * self.rs_rescale
        DM_at_z = cosmo.angular_distance(self.elg_dr16_z_eff) * (1.0 + self.elg_dr16_z_eff)
        H_at_z = cosmo.Hubble(self.elg_dr16_z_eff)
        DH_at_z = 1.0/H_at_z
        # Compute the theoretical value of the observable
        theo_DV_at_z = np.cbrt(self.elg_dr16_z_eff*DH_at_z*DM_at_z**2)/(rd)
        # Interpolate the value within the spline and take the log.
        loglkl = np.log(itp.splev(theo_DV_at_z, self.elg_DV_Interp))
            
        return loglkl/2.0