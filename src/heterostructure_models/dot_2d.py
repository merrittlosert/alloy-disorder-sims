"""
File for creating 2D models of a Si/SiGe heterostructure with alloy disorder and steps.
"""

from dataclasses import dataclass
import numpy as np

import constants

@dataclass 
class Dot2D:
    """
    Class for a 2D heterostructure, with options for steps and alloy disorder.
    """

    si_concentrations: np.ndarray # 1D array of Si concentrations for each layer

    nx: int = 100 # number of lattice sites along x direction
    dx: float = constants.SI_LATTICE_CONSTANT # lattice spacing in x direction, in m
    dx_nm: float = dx * 1e9

    disorder_model: str = 'alloy' # Options: 'none', 'alloy'
    step_model: str = 'none' # Options: 'none', 'single-step', 'staircase', or 'custom'


    y_orbital_spacing: float | None = None # the orbital confinement energy along the y direction, in eV

    # For single-step model, the index of the step location along x, in units of dx.
    # For staircase models, the index is the location of one of the steps, and the rest are determined by step_spacing.
    step_position_index: int | None = None 
    step_spacing: int | None = None # for staircase model, the spacing between steps, in units of dx

    # For staircase and custom step models, the list of step locations along x, in units of dx
    step_location_list: list[int] | None = None

    def __post_init__(self):
        self.n_layers = len(self.si_concentrations)

        self.effective_lattice = None


        if self.disorder_model == 'alloy':
            if self.y_orbital_spacing is None:
                raise ValueError('Must specify y_orbital_spacing for alloy disorder model.')
            
            #self._n_eff_y = np.ceil(constants.HBAR**2 / (2*constants.ELECTRON_MASS_SI * self.y_orbital_spacing * constants.ELECTRON_CHARGE) / self.dx_nm**2)
            omega = self.y_orbital_spacing * constants.ELEMENTARY_CHARGE / constants.HBAR
            ay = np.sqrt(constants.HBAR / constants.SI_TRANSVERSE_MASS / omega)
            self._n_eff_y = 2 * np.sqrt(2 * np.pi) * ay * self.dx / constants.SI_LATTICE_CONSTANT**2

        if self.step_model == 'single-step' or self.step_model == 'staircase':
            if self.step_position_index is None:
                raise ValueError('Must specify step_position_index for single-step or staircase models.')

            elif self.step_position_index < 0 or self.step_position_index >= self.nx:
                raise ValueError('step_position_index must be between 0 and nx-1.')
            
        elif self.step_model == 'staircase':
            if self.step_spacing is None:
                raise ValueError('Must specify step_spacing for staircase model.')
            
        elif self.step_model == 'custom':
            if self.step_location_list is None:
                raise ValueError('Must specify step_location_list for custom step model.')
            

        if self.step_model == 'none':
            # Use the relevant function to generate the 2D lattice, depending on the type of disorder used
            self._generate_2d_lattice_no_step()

        elif self.step_model == 'single-step':
            self._generate_2d_lattice_single_step()

        elif self.step_model == 'staircase':

            step_list = []

            for j in range(self.nx):
                if (self.step_position_index-j) % self.step_spacing == 0:
                    step_list.append(j)

            self.step_location_list = step_list
            self._generate_2d_lattice_from_step_list()

        elif self.step_model == 'custom':
            self._generate_2d_lattice_from_step_list()


    def _generate_2d_lattice_no_step(self):
        """
        Returns a 2D square lattice with a no steps, of dimensions nx by len(si_concentrations).
        This lattice contains no alloy disorder
        """

        lattice = np.zeros((self.nx, self.n_layers))

        for i in range(self.nx):
            for k in range(self.n_layers):
                xk = self.si_concentrations[k]
                if self.disorder_model == 'alloy':
                    lattice[i,k] = np.random.binomial(self._n_eff_y, xk)/self._n_eff_y
                elif self.disorder_model == 'none':
                    lattice[i,k] = xk

        self.effective_lattice = lattice


    def _generate_2d_lattice_single_step(self):
        """
        Returns a 2D square lattice with a single step, of dimensions nx by len(si_concentrations)-1
        """

        nz = self.n_layers - 1
        lattice = np.zeros((self.nx, nz))

        for i in range(self.nx):
            for k in range(nz):
                if i < self.step_position_index:
                    xk = self.si_concentrations[k]
                else:
                    xk = self.si_concentrations[k+1]

                if self.disorder_model == 'alloy':
                    #lattice[i,k] = np.random.binomial(self._n_eff_y, xk)/self._n_eff_y
                    var_si = self._n_eff_y * (xk * (1 - xk))
                    lattice[i,k] = xk + np.random.normal(0, np.sqrt(var_si)) / self._n_eff_y

                elif self.disorder_model == 'none':
                    lattice[i,k] = xk

        self.effective_lattice = lattice


    def _generate_2d_lattice_from_step_list(self):
        """
        Returns a 2D square lattice with a staircase of steps, where the location of the steps is stored in self.step_location_list.
        """

        total_num_steps = len(self.step_location_list)

        nz = len(self.si_concentrations) - total_num_steps
        lattice = np.zeros((self.nx, nz))

        for i in range(self.nx):
            for k in range(nz):

                # Find which plateau a given x location should be on
                plateau_found = False
                for l in range(len(self.step_location_list)):
                    if not plateau_found:
                        step_loc = self.step_location_list[l]

                        # if i is less than a step location, then we know it belongs on the plateau
                        # that ends at that location
                        if i < step_loc:
                            plateau_found = True
                            xk = self.si_concentrations[k+l]

                # if no plateau has yet been found, then the pleateau is the farthest right.
                if not plateau_found:
                    xk = self.si_concentrations[k+total_num_steps]

                if self.disorder_model == 'alloy':
                    #lattice[i,k] = np.random.binomial(self._n_eff_y, xk)/self._n_eff_y
                    var_si = self._n_eff_y * (xk * (1 - xk))
                    lattice[i,k] = xk + np.random.normal(0, np.sqrt(var_si)) / self._n_eff_y

                elif self.disorder_model == 'none':
                    lattice[i,k] = xk

        self.effective_lattice = lattice




















