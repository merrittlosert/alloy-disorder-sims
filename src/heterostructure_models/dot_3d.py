"""
File for creating 3D models of a Si/SiGe heterostructure with alloy disorder and steps.
"""

from dataclasses import dataclass
import numpy as np

import constants
import heterostructure_models.dot_2d as d2

@dataclass 
class Dot3D:
    """
    Class for a 3D heterostructure, with options for steps and alloy disorder.
    """

    si_concentrations: np.ndarray | None = None # 1D array of Si concentrations for each layer

    nx: int = 25 # number of lattice sites along x direction
    dx_unit_cells: int = 4 # spacing, in units of a0

    ny: int = 25 # number of lattice sites along x direction
    dy_unit_cells: int = 4 # spacing, in units of a0

    # We use the 2D dot functionality to define steps
    step_model: d2.Dot2D | None = None

    
    def __post_init__(self):

        if self.si_concentrations is None and self.step_model is None:
            raise ValueError("Must input either a si_concentrations numpy array or a Dot2D step_model")
        
        if self.step_model is not None:
            # If we have provided a Dot2D object, adopt x and z dimensions from this object
            (nx, nz) = np.shape(self.step_model.effective_lattice)
            self.nx = nx 
            self.nz = nz
            self.dx_unit_cells = self.step_model.dx_unit_cells


        self.effective_lattice = None

        self.dx = self.dx_unit_cells * constants.SI_LATTICE_CONSTANT # lattice spacing in x direction, in m
        self.dx_nm = 1e9 * self.dx

        self.dy = self.dy_unit_cells * constants.SI_LATTICE_CONSTANT # lattice spacing in x direction, in m
        self.dy_nm = 1e9 * self.dy


        # There are 2 atoms per unit cell per layer in Si
        self.n_eff = int(np.round(2 * self.dx * self.dy / (constants.SI_LATTICE_CONSTANT**2)))

        if self.step_model is None:
            # Use the relevant function to generate the 2D lattice, depending on the type of steps
            self._generate_3d_lattice_no_step()

        else:
            self._generate_3d_lattice_from_2D_step_model()

    def _generate_3d_lattice_no_step(self):
        """
        Generates a 3D square lattice with a no steps, of dimensions nx by len(si_concentrations).
        This lattice contains no alloy disorder
        """

        lattice = np.zeros((self.nx, self.ny, len(self.si_concentrations)))

        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(len(self.si_concentrations)):
                    xk = self.si_concentrations[k]
                    lattice[i,j,k] = xk

        self.effective_lattice = lattice


    def _generate_3d_lattice_from_2D_step_model(self):
        """
        Generates a 3D square lattice with steps adopted from the Dot2D object.
        """

        nx, nz = np.shape(self.step_model.effective_lattice)
        lattice = np.zeros((nx, self.ny, nz))

        for i in range(nx):
            for j in range(self.ny):
                for k in range(nz):
                    xk = self.step_model.effective_lattice[i, k]
                    lattice[i,j,k] = xk

        self.effective_lattice = lattice


    def generate_random_alloy_lattice(self):
        """
        Using the effective_lattice model, generate a random instantiation of alloy disorder
        """
       
        random_lattice = np.zeros_like(self.effective_lattice)

        (nx, ny, nz) = np.shape(random_lattice)

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    xk = self.effective_lattice[i,j,k]

                    # Because our lattice will have a finite number of atoms per cell, generate with a binomial
                    random_lattice[i,j,k] = np.random.binomial(self.n_eff, xk) / self.n_eff

        return random_lattice
    




















