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
    dx_unit_cells: int = 1 # spacing in units of a0

    step_model: str = 'none' # Options: 'none', 'single-step', 'staircase', or 'custom'

    # For single-step model, the index of the step location along x, in units of dx.
    # For staircase models, the index is the location of one of the steps, and the rest are determined by step_spacing.
    # For custom step models, the list of step locations along x, in units of dx
    step_position: int | None = None 
    step_spacing: int | None = None # for staircase model, the spacing between steps, in units of dx

    # For custom steps, we can choose to have upward or downward steps.
    # Can also be used to set the single-step or staircase step size.
    step_magnitudes: list[int] | int | None = None
    
    def __post_init__(self):

        self.dx = self.dx_unit_cells * constants.SI_LATTICE_CONSTANT # lattice spacing in x direction, in m
        self.dx_nm = 1e9 * self.dx

        self.effective_lattice = None

        if self.step_model == 'single-step' or self.step_model == 'staircase':
            if self.step_position is None or not isinstance(self.step_position, int):
                raise ValueError('Must specify an integer step_position for single-step or staircase models.')
            
            elif self.step_position < 0 or self.step_position >= self.nx:
                raise ValueError('step_position must be between 0 and nx-1.')
            
            if self.step_model == 'single-step':
                if self.step_magnitudes is None:
                    self._step_magnitudes = 1
                elif isinstance(self.step_magnitudes, int):
                    self._step_magnitudes = self.step_magnitudes
                else:
                    raise ValueError("step_magnitudes must be an integer or None for single-step")

            if self.step_model == 'staircase':
                if self.step_magnitudes is None:
                    self._step_magnitudes = 1
                elif isinstance(self.step_magnitudes, int):
                    self._step_magnitudes = self.step_magnitudes
                else:
                    raise ValueError("step_magnitudes must be an integer or None for single-step")
                
                if self.step_spacing is None:
                    raise ValueError('Must specify step_spacing for staircase model.')
            
                step_list = []

                for j in range(self.nx):
                    if (self.step_position-j) % self.step_spacing == 0:
                        step_list.append(j)

                self._step_location_list = step_list
                self._step_magnitudes = self._step_magnitudes * np.ones_like(step_list)

                        
        elif self.step_model == 'custom':
            if self.step_position is None or not (isinstance(self.step_position, int) or isinstance(self.step_position, list)):
                raise ValueError('Must specify step_position (either int or list[int]) for custom step model.')
            
            if isinstance(self.step_position, int):
                self._step_location_list = [self.step_position]
            else:
                self._step_location_list = self.step_position
            
            # Handle custom step magnitudes
            if self.step_magnitudes is None:
                self._step_magnitudes = np.ones_like(self.step_position)
            
            elif isinstance(self.step_magnitudes, int):
                self._step_magnitudes = self.step_magnitudes * np.ones_like(self.step_position)
            
            elif len(self.step_magnitudes) != len(self.step_position):
                raise ValueError('step_magnitudes must either be an integer or a list of equal length to step_position.')
            
            else:
                self._step_magnitudes = self.step_magnitudes


            
        if self.step_model == 'none':
            # Use the relevant function to generate the 2D lattice, depending on the type of steps
            self._generate_2d_lattice_no_step()

        elif self.step_model == 'single-step':
            self._generate_2d_lattice_single_step()

        elif self.step_model == 'staircase' or self.step_model == 'custom':
            self._generate_2d_lattice_from_step_list()


    def _generate_2d_lattice_no_step(self):
        """
        Generates a 2D square lattice with a no steps, of dimensions nx by len(si_concentrations).
        This lattice contains no alloy disorder
        """

        lattice = np.zeros((self.nx, len(self.si_concentrations)))

        for i in range(self.nx):
            for k in range(len(self.si_concentrations)):
                xk = self.si_concentrations[k]
                lattice[i,k] = xk

        self.effective_lattice = lattice


    def _generate_2d_lattice_single_step(self):
        """
        Generates a 2D square lattice with a single step, of dimensions nx by len(si_concentrations)-1
        """

        nz = len(self.si_concentrations) - abs(self._step_magnitudes)
        lattice = np.zeros((self.nx, nz))

        if self._step_magnitudes > 0:
            pos_step = True 
        else:
            pos_step = False


        for i in range(self.nx):
            for k in range(nz):
                if i < self.step_position:
                    if pos_step:
                        xk = self.si_concentrations[k]
                    else:
                        xk = self.si_concentrations[k + abs(self._step_magnitudes)]

                else:
                    if pos_step:
                        xk = self.si_concentrations[k + self._step_magnitudes]
                    else:
                        xk = self.si_concentrations[k]

                lattice[i,k] = xk

        self.effective_lattice = lattice


    def _generate_2d_lattice_from_step_list(self):
        """
        Returns a 2D square lattice with a staircase of steps, where the location of the steps is stored in self.step_position.
        """

        total_step_offset = int(np.sum(self._step_magnitudes))

        accumulated_offset = np.array([np.sum(self._step_magnitudes[:l+1]) for l in range(len(self._step_location_list))]).astype(int)
        max_offset = accumulated_offset[np.argmax(np.abs(accumulated_offset))]

        nz = len(self.si_concentrations) - (max_offset)
        lattice = np.zeros((self.nx, nz))

        for i in range(self.nx):
            for k in range(nz):

                # Find which plateau a given x location should be on
                plateau_found = False
                for l in range(len(self._step_location_list)):
                    if not plateau_found:
                        step_offset = int(np.sum(self._step_magnitudes[:l]))
                        step_loc = self._step_location_list[l]

                        # if i is less than a step location, then we know it belongs on the plateau
                        # that ends at that location
                        if i < step_loc:
                            plateau_found = True
                            #xk = self.si_concentrations[k+l]
                            xk = self.si_concentrations[k + step_offset - max_offset]

                # if no plateau has yet been found, then the pleateau is the farthest right.
                if not plateau_found:
                    xk = self.si_concentrations[k + total_step_offset - max_offset]

                lattice[i,k] = xk

        self.effective_lattice = lattice


    def generate_random_alloy_lattice(self, dot_radius_nm_y):
        omega = constants.HBAR / constants.SI_TRANSVERSE_MASS / (1e-9 * dot_radius_nm_y)**2

        #omega = y_orbital_spacing * constants.ELEMENTARY_CHARGE / constants.HBAR
        #ay = np.sqrt(constants.HBAR / constants.SI_TRANSVERSE_MASS / omega)
        self._n_eff_y = 2 * np.sqrt(2 * np.pi) * (1e-9 * dot_radius_nm_y) * self.dx / constants.SI_LATTICE_CONSTANT**2
        random_lattice = np.zeros_like(self.effective_lattice)

        (nx, nz) = np.shape(random_lattice)

        for i in range(nx):
            for j in range(nz):
                xk = self.effective_lattice[i,j]
                var_si = self._n_eff_y * (xk * (1 - xk))
                random_lattice[i,j] = xk + np.random.normal(0, np.sqrt(var_si)) / self._n_eff_y

        return random_lattice





















