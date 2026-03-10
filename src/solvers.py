"""
Various solvers for Si/SiGe, including the two-band tight-binding model and effective mass models
"""

from dataclasses import dataclass
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as las
from math import *

import constants
import helpers.solver_helpers as helpers

@dataclass
class BaseSolver:
    """
    Base class for the Schrodinger equation solvers.
    """

    effective_lattice: np.ndarray # Either a 1D, 2D, or 3D array of si concentrations 

    bulk_si_concentration: float = 0.7
    well_si_concentration: float = 1.0

    vertical_field: float = 0 # in V/nm

    def __post_init__(self):

        self.dz_nm = 1e9 * constants.SI_LATTICE_CONSTANT / 4 # spacing 
        self.dz = 1e-9 * self.dz_nm

        
        # Define the conduction band offset for the given profile
        y = 1-self.bulk_si_concentration
        y_well = 1-self.well_si_concentration
        self.delta_E_Si = -0.502*(1-self.bulk_si_concentration)
        self.delta_E_Ge = 0.743 - 0.625*(1-self.bulk_si_concentration)

        # conduction band offset in eV
        self.cb_offset = (y-y_well)*( ((1-y_well)/y)*self.delta_E_Si - (y_well/(1-y))*self.delta_E_Ge)

        # set the Hamiltonian to None so that it will be lazily computed the first time it is accessed
        self._H = None

        self._valley_splitting = None
        self.n_eigenstates_calculated: int | None = None


    def vertical_field_potential(self, layer_from_bottom):
        """
        Returns the potential due to a vertical electric field
        """
        return self.vertical_field * (self.dz_nm) * layer_from_bottom
    
    def fractional_cb_offset(self, si_concentration):
        """
        Returns the potential corresponding to a given si_concentration
        """
        return (si_concentration-self.bulk_si_concentration)/(self.well_si_concentration-self.bulk_si_concentration) * self.cb_offset
    

    def solve(self, n_lowest_eigenstates=2):
        """
        Solve the Hamiltonian for the first n_lowest_eigenstates eigenstates
        """

        if self.n_eigenstates_calculated is not None and self.n_eigenstates_calculated >= n_lowest_eigenstates:
            return self._lowest_evals[:n_lowest_eigenstates], self._lowest_evecs[:,:n_lowest_eigenstates]

        if self.sparse:
            # H_x will be a sparse matrix
            ham_sp = self.hamiltonian
            w, v = las.eigsh(ham_sp, k=n_lowest_eigenstates, which='SA')
        else:
            ham = self.hamiltonian
            w, v = la.eigh(ham)

        e_vals = np.sort(w)
        sort_index = np.argsort(w) 
        
        self._lowest_evals = e_vals[:n_lowest_eigenstates]
        self._lowest_evecs = self._normalize(v[:,sort_index[:n_lowest_eigenstates]])
        self._solved_num = n_lowest_eigenstates

        self.n_eigenstates_calculated = n_lowest_eigenstates

        return self._lowest_evals, self._lowest_evecs



    @property
    def hamiltonian(self):
        """
        Lazily constructs the hamiltonian matrix the first time it is accessed, then returns the stored value on subsequent accesses
        """
        if self._H is None:
            self._compute_H()
        return self._H
    

    @property
    def valley_splitting(self):
        """
        Lazily computes the valley splitting. Implementation depends on the method.
        """
        if self._valley_splitting is None:
            self._compute_valley_splitting()
        return self._valley_splitting 
    

    
    





@dataclass 
class TwoBand(BaseSolver):
    """
    Base class for the two-band tight-binding solvers.
    We use the model of Boykin et al., PRB 70, 165325 (2004)
    """

    sparse: bool = True

    def __post_init__(self):
        super().__post_init__()

        # Couplings that reproduce the position and curvature of the conduction band minima
        self.nnn_coupling_z =  2 * constants.HBAR**2 / (constants.SI_LATERAL_MASS * constants.SI_LATTICE_CONSTANT**2 * (np.sin(constants.K_0 * constants.SI_LATTICE_CONSTANT / 4))**2) / constants.ELEMENTARY_CHARGE
        self.nn_coupling_z =  4 * self.nnn_coupling_z * np.cos(constants.K_0 * constants.SI_LATTICE_CONSTANT / 4)



    def _compute_valley_splitting(self):
        """
        Compute the valley splitting as the energy difference between the first two eigenstates
        """
        if self.n_eigenstates_calculated is None or self.n_eigenstates_calculated < 2:
            self.solve(n_lowest_eigenstates=2)

        self._valley_splitting = self._lowest_evals[1] - self._lowest_evals[0]
        return self._valley_splitting

  





@dataclass
class TwoBand_1D(TwoBand):

    def __post_init__(self):
        super().__post_init__()

        if len(np.shape(self.effective_lattice)) != 1:
            raise ValueError("The effective lattice should be a 1D array for the 1D model")
        
        self.nz = len(self.effective_lattice)

        self._z00_nm = None 
        self._z11_nm = None 
        self._z01_nm = None
        

    def _onsite_term(self, i):
        """
        A function that takes as input the index i and returns the onsite potential from vertical field and Ge concentration terms
        """
        return self.vertical_field_potential(i) + self.fractional_cb_offset(self.effective_lattice[i])

    def _compute_H(self):
        """
        Construct the 1D TB Hamiltonian
        """

        ham = helpers.compute_1D_H(
            nz = self.nz, 
            onsite_func = self._onsite_term, 
            nn_coupling_z = self.nn_coupling_z, 
            nnn_coupling_z = self.nnn_coupling_z,
            use_sparse=self.sparse
        )
        self._H = ham


    def _normalize(self, evecs):
        return evecs / np.linalg.norm(evecs, axis=0) / np.sqrt(self.dz)
    
    @property
    def z00_nm(self):
        """
        Lazily computes the dipolar matrix element z00
        """
        if self._z00_nm is None:
            self._compute_dipolar_matrix_elements()
        return self._z00_nm
    
    @property
    def z11_nm(self):
        """
        Lazily computes the dipolar matrix element z11
        """
        if self._z11_nm is None:
            self._compute_dipolar_matrix_elements()
        return self._z11_nm
    
    @property
    def z01_nm(self):
        """
        Lazily computes the dipolar matrix element z01
        """
        if self._z01_nm is None:
            self._compute_dipolar_matrix_elements()
        return self._z01_nm
    

    def _compute_dipolar_matrix_elements(self):
        """
        Lazily computes the dipolar matrix elements between ground and excited valleys.
        """
        if self.n_eigenstates_calculated is None or self.n_eigenstates_calculated < 2:
            self.solve(n_lowest_eigenstates=2)

        zs = np.arange(self.nz) * self.dz

        v0 = self._lowest_evecs[:,0]
        v1 = self._lowest_evecs[:,1]

        # Compute z matrix elements, converting to nm
        self._z00_nm = 1e9 * np.sum(self.dz * v0 * v0 * zs)
        self._z01_nm = 1e9 * np.sum(self.dz * v0 * v1 * zs)
        self._z11_nm = 1e9 * np.sum(self.dz * v1 * v1 * zs)



@dataclass
class TwoBand_2D(TwoBand):

    dx_nm: float | None = None
    lateral_confinement_energy: float = 2e-3 # eV
    center_x_nm: float | None = None # center of the confinement in nm

    periodic: bool = False # if we should use periodic BCs

    def __post_init__(self):
        super().__post_init__()

        if len(np.shape(self.effective_lattice)) != 2:
            raise ValueError("The effective lattice should be a 2D array for the 2D model")

        if self.dx_nm is None:
            raise ValueError("Please specify a lattice spacing dx_nm for the 2D model")
        
        self.dx = 1e-9*self.dx_nm
        
        (nx, nz) = np.shape(self.effective_lattice)
        self.nx = nx
        self.nz = nz

        if self.center_x_nm is None:
            self.center_x_nm = self.dx_nm * nx/2
            self.center_x = 1e-9*self.center_x_nm
        else:
            self.center_x = (1e-9)*self.center_x_nm


        self.omega_x = constants.ELEMENTARY_CHARGE * self.lateral_confinement_energy / constants.HBAR

        self.nn_coupling_x = -constants.HBAR**2 / (2 * constants.SI_TRANSVERSE_MASS * self.dx**2) / constants.ELEMENTARY_CHARGE

        # These will be computed if called for
        self._x00_nm = None
        self._x01_nm = None 
        self._x11_nm = None

        self._z00_nm = None
        self._z01_nm = None 
        self._z11_nm = None


    def _confinement(self, i):
        """
        The lateral confinement potential at index i
        """
        x = i*self.dx
        c = (1/2) * constants.SI_TRANSVERSE_MASS*(self.omega_x**2 * (x-self.center_x)**2 ) / constants.ELEMENTARY_CHARGE # divide by e to convert to eV
        return c
    
    def _qw_offset(self, i, k):
        """
        The quantum well potential at index i, k
        """
        s = self.effective_lattice[i,k]
        return self.fractional_cb_offset(s)

    def _onsite_term(self, i, k):
        """
        The potential terms at index i, k
        """
        on = self.vertical_field_potential(k) + self._confinement(i) + self._qw_offset(i,k)
        return on


    def _compute_H(self):
        """
        Construct the 2D TB Hamiltonian
        """

        ham = helpers.compute_2D_H(
            nx = self.nx, 
            nz = self.nz, 
            onsite_func = self._onsite_term,
            nn_coupling_z = self.nn_coupling_z, 
            nnn_coupling_z = self.nnn_coupling_z,
            nn_coupling_x = self.nn_coupling_x,
            use_sparse = self.sparse,
            periodic = self.periodic,
        )

        self._H = ham


    def wf_2D_matrix_from_vector(self, v):
        """
        Convert 1D vectors into 2D arrays
        """

        mat = np.zeros((self.nx, self.nz), dtype=np.double)
        for ind in range(len(v)):
            (i,k) = helpers.coordinates_2D(ind, nz=self.nz)
            mat[i,k] = v[ind]

        return mat
    

    def _normalize(self, evecs):
        """
        Normalize the wavefunctions
        """
        return evecs / np.linalg.norm(evecs, axis=0) / np.sqrt(self.dz * self.dx)
    


    @property
    def x00_nm(self):
        """
        Lazily computes the dipolar matrix element x00
        """
        if self._x00_nm is None:
            self._compute_dipolar_matrix_elements()
        return self._x00_nm
    
    @property
    def x11_nm(self):
        """
        Lazily computes the dipolar matrix element x11
        """
        if self._x11_nm is None:
            self._compute_dipolar_matrix_elements()
        return self._x11_nm
    
    @property
    def x01_nm(self):
        """
        Lazily computes the dipolar matrix element x01
        """
        if self._x01_nm is None:
            self._compute_dipolar_matrix_elements()
        return self._x01_nm
    
    @property
    def z00_nm(self):
        """
        Lazily computes the dipolar matrix element z00
        """
        if self._z00_nm is None:
            self._compute_dipolar_matrix_elements()
        return self._z00_nm
    
    @property
    def z11_nm(self):
        """
        Lazily computes the dipolar matrix element z11
        """
        if self._z11_nm is None:
            self._compute_dipolar_matrix_elements()
        return self._z11_nm
    
    @property
    def z01_nm(self):
        """
        Lazily computes the dipolar matrix element z01
        """
        if self._z01_nm is None:
            self._compute_dipolar_matrix_elements()
        return self._z01_nm

    def _compute_dipolar_matrix_elements(self):
        """
        Lazily computes the dipolar matrix elements between ground and excited valleys.

        For x matrix elements, we set the origin at center_x.
        """
        if self.n_eigenstates_calculated is None or self.n_eigenstates_calculated < 2:
            self.solve(n_lowest_eigenstates=2)

        x_range = np.arange(self.nx) * self.dx
        z_range = np.arange(self.nz) * self.dz
        zs, xs = np.meshgrid(z_range, x_range)

        xs = xs - self.center_x

        v0 = self.wf_2D_matrix_from_vector(self._lowest_evecs[:,0])
        v1 = self.wf_2D_matrix_from_vector(self._lowest_evecs[:,1])

        # Compute x and z matrix elements, converting to nm
        self._x00_nm = 1e9 * np.sum(self.dx * self.dz * v0 * v0 * xs)
        self._x01_nm = 1e9 * np.sum(self.dx * self.dz * v0 * v1 * xs)
        self._x11_nm = 1e9 * np.sum(self.dx * self.dz * v1 * v1 * xs)

        self._z00_nm = 1e9 * np.sum(self.dx * self.dz * v0 * v0 * zs)
        self._z01_nm = 1e9 * np.sum(self.dx * self.dz * v0 * v1 * zs)
        self._z11_nm = 1e9 * np.sum(self.dx * self.dz * v1 * v1 * zs)










@dataclass
class TwoBand_3D(TwoBand):

    dx_nm: float | None = None
    dy_nm: float | None = None

    center_x_nm: float | None = None # center of the confinement (x) in nm
    center_y_nm: float | None = None # center of the confinement (y) in nm

    lateral_confinement_energy_x: float = 2e-3 # eV
    lateral_confinement_energy_y: float = 2e-3 # eV

    periodic: bool = False # if we should use periodic BCs

    def __post_init__(self):
        super().__post_init__()

        if len(np.shape(self.effective_lattice)) != 3:
            raise ValueError("The effective lattice should be a 3D array for the 3D model")

        if self.dx_nm is None:
            raise ValueError("Please specify a lattice spacing dx_nm for the 2D model")
        if self.dy_nm is None:
            raise ValueError("Please specify a lattice spacing dy_nm for the 2D model")
        
        self.dx = 1e-9*self.dx_nm
        self.dy = 1e-9*self.dy_nm

        (nx, ny, nz) = np.shape(self.effective_lattice)
        self.nx = nx
        self.ny = ny
        self.nz = nz

        if self.center_x_nm is None:
            self.center_x_nm = self.dx_nm * nx/2
            self.center_x = 1e-9*self.center_x_nm
        else:
            self.center_x = (1e-9)*self.center_x_nm

        if self.center_y_nm is None:
            self.center_y_nm = self.dy_nm * ny/2
            self.center_y = 1e-9*self.center_y_nm
        else:
            self.center_y = (1e-9)*self.center_y_nm


        self.omega_x = constants.ELEMENTARY_CHARGE * self.lateral_confinement_energy_x / constants.HBAR
        self.omega_y = constants.ELEMENTARY_CHARGE * self.lateral_confinement_energy_y / constants.HBAR

        self.nn_coupling_x = -constants.HBAR**2 / (2 * constants.SI_TRANSVERSE_MASS * self.dx**2) / constants.ELEMENTARY_CHARGE
        self.nn_coupling_y = -constants.HBAR**2 / (2 * constants.SI_TRANSVERSE_MASS * self.dy**2) / constants.ELEMENTARY_CHARGE 

        # These will be computed if called for
        self._x00_nm = None
        self._x01_nm = None 
        self._x11_nm = None

        self._y00_nm = None
        self._y01_nm = None 
        self._y11_nm = None

        self._z00_nm = None
        self._z01_nm = None 
        self._z11_nm = None
     

    def _confinement(self, i, j):
        """
        The lateral confinement at index i, j
        """
        if self.periodic == False:
            x = i*self.dx
            y = j*self.dy

            c = (1/2)*constants.SI_TRANSVERSE_MASS*(self.omega_x**2 * (x-self.center_x)**2 + self.omega_y**2 * (y-self.center_y)**2)/constants.ELEMENTARY_CHARGE # divide by e to convert to eV
            return c
        else:
            return 0
        
    def _qw_offset(self, i, j, k):
        """
        The quantum well potential at index i, j, k
        """
        s = self.effective_lattice[i,j,k]
        return self.fractional_cb_offset(s)

    def _onsite_term(self, i, j, k):
        """
        The potential terms at index i, j, k
        """
        on = self.vertical_field_potential(k) + self._confinement(i,j) + self._qw_offset(i,j,k)
        return on


    def _compute_H(self):
        """
        Construct the 3D TB Hamiltonian
        """

        ham = helpers.compute_3D_H(
                nx = self.nx, 
                ny = self.ny,
                nz = self.nz, 
                onsite_func = self._onsite_term,
                nn_coupling_z = self.nn_coupling_z, 
                nnn_coupling_z = self.nnn_coupling_z,
                nn_coupling_x = self.nn_coupling_x,
                nn_coupling_y = self.nn_coupling_y,
                use_sparse = self.sparse,
                periodic = self.periodic,
        )
        self._H = ham
    


    def wf_3D_matrix_from_vector(self, v):
        """
        Convert 1D vectors into 3D arrays
        """

        mat = np.zeros((self.nx, self.ny, self.nz), dtype=np.double)
        for ind in range(len(v)):
            (i,j,k) = helpers.coordinates_3D(ind, ny=self.ny, nz=self.nz)
            mat[i,j,k] = v[ind]

        return mat
    

    def _normalize(self, evecs):
        """
        Normalize wavefunctions
        """
        return evecs / np.linalg.norm(evecs, axis=0) / np.sqrt(self.dz * self.dx * self.dy)
    

    @property
    def x00_nm(self):
        """
        Lazily computes the dipolar matrix element x00
        """
        if self._x00_nm is None:
            self._compute_dipolar_matrix_elements()
        return self._x00_nm
    
    @property
    def x11_nm(self):
        """
        Lazily computes the dipolar matrix element x11
        """
        if self._x11_nm is None:
            self._compute_dipolar_matrix_elements()
        return self._x11_nm
    
    @property
    def x01_nm(self):
        """
        Lazily computes the dipolar matrix element x01
        """
        if self._x01_nm is None:
            self._compute_dipolar_matrix_elements()
        return self._x01_nm
    
    @property
    def y00_nm(self):
        """
        Lazily computes the dipolar matrix element y00
        """
        if self._y00_nm is None:
            self._compute_dipolar_matrix_elements()
        return self._y00_nm
    
    @property
    def y11_nm(self):
        """
        Lazily computes the dipolar matrix element y11
        """
        if self._y11_nm is None:
            self._compute_dipolar_matrix_elements()
        return self._y11_nm
    
    @property
    def y01_nm(self):
        """
        Lazily computes the dipolar matrix element y01
        """
        if self._y01_nm is None:
            self._compute_dipolar_matrix_elements()
        return self._y01_nm
    
    @property
    def z00_nm(self):
        """
        Lazily computes the dipolar matrix element z00
        """
        if self._z00_nm is None:
            self._compute_dipolar_matrix_elements()
        return self._z00_nm
    
    @property
    def z11_nm(self):
        """
        Lazily computes the dipolar matrix element z11
        """
        if self._z11_nm is None:
            self._compute_dipolar_matrix_elements()
        return self._z11_nm
    
    @property
    def z01_nm(self):
        """
        Lazily computes the dipolar matrix element z01
        """
        if self._z01_nm is None:
            self._compute_dipolar_matrix_elements()
        return self._z01_nm

    def _compute_dipolar_matrix_elements(self):
        """
        Lazily computes the dipolar matrix elements between ground and excited valleys.

        For x matrix elements, we set the origin at center_x.
        """
        if self.n_eigenstates_calculated is None or self.n_eigenstates_calculated < 2:
            self.solve(n_lowest_eigenstates=2)

        x_range = np.arange(self.nx) * self.dx
        y_range = np.arange(self.ny) * self.dy
        z_range = np.arange(self.nz) * self.dz
    
        ys, xs, zs = np.meshgrid(y_range, x_range, z_range)

        xs = xs - self.center_x
        ys = ys - self.center_y

        v0 = self.wf_3D_matrix_from_vector(self._lowest_evecs[:,0])
        v1 = self.wf_3D_matrix_from_vector(self._lowest_evecs[:,1])


        # Compute x and z matrix elements, converting to nm
        self._x00_nm = 1e9 * np.sum(self.dx * self.dy * self.dz * v0 * v0 * xs)
        self._x01_nm = 1e9 * np.sum(self.dx * self.dy * self.dz * v0 * v1 * xs)
        self._x11_nm = 1e9 * np.sum(self.dx * self.dy * self.dz * v1 * v1 * xs)

        self._y00_nm = 1e9 * np.sum(self.dx * self.dy * self.dz * v0 * v0 * ys)
        self._y01_nm = 1e9 * np.sum(self.dx * self.dy * self.dz * v0 * v1 * ys)
        self._y11_nm = 1e9 * np.sum(self.dx * self.dy * self.dz * v1 * v1 * ys)

        self._z00_nm = 1e9 * np.sum(self.dx * self.dy * self.dz * v0 * v0 * zs)
        self._z01_nm = 1e9 * np.sum(self.dx * self.dy * self.dz * v0 * v1 * zs)
        self._z11_nm = 1e9 * np.sum(self.dx * self.dy * self.dz * v1 * v1 * zs)

        






@dataclass 
class EffectiveMass(BaseSolver):
    """
    Base class for the EM solvers.
    This model ignores valley physics, and just uses the effective masses of Si. 
    Valley splitting is computed perturbatively using the EMA.
    """

    sparse: bool = True

    def __post_init__(self):
        super().__post_init__()

        # For effective mass, we ignore valleys and use the standard nn coupling that reproduces the curvature of the conduction band minima
        self.nn_coupling_z = -constants.HBAR**2 / (2 * constants.SI_LATERAL_MASS * self.dz**2) / constants.ELEMENTARY_CHARGE

        self._inter_valley_coupling = None


    def _compute_valley_splitting(self):
        """
        Lazily compute the valley splitting using EM theory
        """
        if self._inter_valley_coupling is None:
            self._compute_inter_valley_coupling()
        
        self._valley_splitting = 2 * abs(self._inter_valley_coupling)
        return self._valley_splitting


    @property
    def inter_valley_coupling(self):
        """
        Lazily compute the inter-valley coupling using EM theory, given the effective_lattice of Si concentrations.
        """
        if self._inter_valley_coupling is None:
            self._compute_inter_valley_coupling()
        return self._inter_valley_coupling
    
        




@dataclass
class EffectiveMass_1D(EffectiveMass):

    def __post_init__(self):
        super().__post_init__()

        if len(np.shape(self.effective_lattice)) != 1:
            raise ValueError("The effective lattice should be a 1D array for the 1D model")
        
        self.nz = len(self.effective_lattice)

    def _onsite_term(self, i):
        """
        A function that takes as input the index i and returns the onsite potential from vertical field and Ge concentration terms
        """
        return self.vertical_field_potential(i) + self.fractional_cb_offset(self.effective_lattice[i])

    def _compute_H(self):
        """
        Construct the 1D EM Hamiltonian
        """
        ham = helpers.compute_1D_H(
            nz = self.nz, 
            onsite_func = self._onsite_term, 
            nn_coupling_z = self.nn_coupling_z, 
            nnn_coupling_z = 0,
            use_sparse=self.sparse
        )

        self._H = ham


    def _compute_inter_valley_coupling(self):
        """
        Compute the inter-valley coupling from self.effective_lattice using EM theory
        """
        _, v0 = self.solve(n_lowest_eigenstates=1)
        psi_0 = v0[:,0]

        z_arr = 1e-9 * np.arange(self.nz) * self.dz_nm
        pot_arr = self.fractional_cb_offset(self.effective_lattice)
        coupling = np.sum(self.dz * np.abs(psi_0**2) * np.exp(-1j * 2 * constants.K_0 * z_arr) * pot_arr)

        self._inter_valley_coupling = coupling
        return coupling
    
    
    def sigma_delta(self, dot_size_nm_x: float, dot_size_nm_y: float):
        """
        Compute sigma_delta given the dot sizes in x and y (equal to sqrt(HBAR / SI_TRANSVERSE_MASS / omega_x(y))).

        Note: for the most accurate results, the input lattice should be smooth.
        For these calculations, we assume self.effective_lattice respresents an expected Si
        concentration at each lattice site. If the input lattice contains disorder, results for
        sigma_delta will be slightly elevated.
        """
        _, v0 = self.solve(n_lowest_eigenstates=1)
        psi_0 = np.abs(v0[:,0])

        dot_size_nm = np.sqrt(dot_size_nm_x * dot_size_nm_y)

        concs = self.effective_lattice
        delta_conc = self.well_si_concentration - self.bulk_si_concentration
        c = (1/np.pi) * ((constants.SI_LATTICE_CONSTANT**2 * self.cb_offset) / (8 * 1e-9 * dot_size_nm * delta_conc))**2 
        sd = np.sqrt( c * np.sum( psi_0**4 * concs * (1-concs) ) )

        return sd


    def _normalize(self, evecs):
        """
        When normalizing the envelope functions, ensure the maximum amplitude is positive
        """
        return evecs / np.linalg.norm(evecs, axis=0) / np.sqrt(self.dz) * np.sign(np.max(evecs, axis=0))

       



@dataclass
class EffectiveMass_2D(EffectiveMass):

    dx_nm: float | None = None
    lateral_confinement_energy: float = 2e-3 # eV
    center_x_nm: float | None = None # center of the confinement in nm

    periodic: bool = False # if we should use periodic BCs

    def __post_init__(self):
        super().__post_init__()

        if len(np.shape(self.effective_lattice)) != 2:
            raise ValueError("The effective lattice should be a 2D array for the 2D model")

        if self.dx_nm is None:
            raise ValueError("Please specify a lattice spacing dx_nm for the 2D model")
        
        self.dx = 1e-9*self.dx_nm
        
        (nx, nz) = np.shape(self.effective_lattice)
        self.nx = nx
        self.nz = nz

        if self.center_x_nm is None:
            self.center_x_nm = self.dx_nm * nx/2
            self.center_x = 1e-9*self.center_x_nm
        else:
            self.center_x = (1e-9)*self.center_x_nm


        self.omega_x = constants.ELEMENTARY_CHARGE * self.lateral_confinement_energy / constants.HBAR

        self.nn_coupling_x = -constants.HBAR**2 / (2 * constants.SI_TRANSVERSE_MASS * self.dx**2) / constants.ELEMENTARY_CHARGE


    
    def _confinement(self, i):
        """
        Horizontal confinement term at index i
        """
        x = i*self.dx
        c = (1/2) * constants.SI_TRANSVERSE_MASS*(self.omega_x**2 * (x-self.center_x)**2 ) / constants.ELEMENTARY_CHARGE # divide by e to convert to eV
        return c
    
    def _qw_offset(self, i, k):
        """
        QW offset potential at index i, k
        """
        s = self.effective_lattice[i,k]
        return self.fractional_cb_offset(s)

    def _onsite_term(self, i, k):
        """
        Sum of potential terms at index i, k
        """
        on = self.vertical_field_potential(k) + self._confinement(i) + self._qw_offset(i,k)
        return on


    def _compute_H(self):
        """
        Construct 2D EM Hamiltonian
        """

        ham = helpers.compute_2D_H(
            nx = self.nx, 
            nz = self.nz, 
            onsite_func = self._onsite_term,
            nn_coupling_z = self.nn_coupling_z, 
            nnn_coupling_z = 0,
            nn_coupling_x = self.nn_coupling_x,
            use_sparse = self.sparse,
            periodic = self.periodic,
        )

        self._H = ham

    def wf_2D_matrix_from_vector(self, v):
        """
        Convert from a 1D vector to a 2D array
        """

        mat = np.zeros((self.nx, self.nz), dtype=np.double)
        for ind in range(len(v)):
            (i,k) = helpers.coordinates_2D(ind, nz=self.nz)
            mat[i,k] = v[ind]

        return mat
    

    def _compute_inter_valley_coupling(self):
        """
        Computing the inter-valley coupling using self.effective_lattice and effective mass theory
        """
        _, v0 = self.solve(n_lowest_eigenstates=1)
        psi_0 = self.wf_2D_matrix_from_vector(v0[:,0])

        z_arr = 1e-9 * np.arange(self.nz) * self.dz_nm
        x_arr = 1e-9 * np.arange(self.nx) * self.dx_nm 

        pot_arr = self.fractional_cb_offset(self.effective_lattice)
        coupling = np.sum(self.dz * self.dx * np.abs(psi_0**2) * np.exp(-1j * 2 * constants.K_0 * z_arr) * pot_arr)

        self._inter_valley_coupling = coupling
        return coupling
    
    
    def sigma_delta(self, dot_size_nm_y: float):
        """
        Compute sigma_delta given the dot size in y (equal to sqrt(HBAR / SI_TRANSVERSE_MASS / omega)).
        """

        _, v0 = self.solve(n_lowest_eigenstates=1)
        psi_0 = self.wf_2D_matrix_from_vector(np.abs(v0[:,0]))

        # Effective "number of atoms" per unit cell after averaging along y
        n_eff_y = 2 * np.sqrt(2 * np.pi) * (1e-9 * dot_size_nm_y) * self.dx / constants.SI_LATTICE_CONSTANT**2
        concs = self.effective_lattice

        # Variance of Si conc. at each site
        var_Si = concs * (1 - concs) / n_eff_y 

        # Variance of potential at each site
        var_U = var_Si * (self.cb_offset / (self.well_si_concentration - self.bulk_si_concentration))**2 

        # Variance of Delta
        vd = (self.dx * self.dz)**2 * np.sum( psi_0**4 * var_U )

        return np.sqrt(vd)
    

    def _normalize(self, evecs):
        """
        When normalizing the envelope functions, ensure the maximum amplitude is positive
        """
        return evecs / np.linalg.norm(evecs, axis=0) / np.sqrt(self.dz * self.dx) * np.sign(np.max(evecs, axis=0))
    

       





@dataclass
class EffectiveMass_3D(EffectiveMass):

    dx_nm: float | None = None
    dy_nm: float | None = None

    center_x_nm: float | None = None # center of the confinement (x) in nm
    center_y_nm: float | None = None # center of the confinement (y) in nm

    lateral_confinement_energy_x: float = 2e-3 # eV
    lateral_confinement_energy_y: float = 2e-3 # eV

    periodic: bool = False # if we should use periodic BCs

    def __post_init__(self):
        super().__post_init__()

        if len(np.shape(self.effective_lattice)) != 3:
            raise ValueError("The effective lattice should be a 3D array for the 3D model")

        if self.dx_nm is None:
            raise ValueError("Please specify a lattice spacing dx_nm for the 3D model")
        
        if self.dy_nm is None:
            raise ValueError("Please specify a lattice spacing dy_nm for the 3D model")

        self.dx = 1e-9*self.dx_nm
        self.dy = 1e-9*self.dy_nm

        (nx, ny, nz) = np.shape(self.effective_lattice)
        self.nx = nx
        self.ny = ny
        self.nz = nz

        if self.center_x_nm is None:
            self.center_x_nm = self.dx_nm * nx/2
            self.center_x = 1e-9*self.center_x_nm
        else:
            self.center_x = (1e-9)*self.center_x_nm

        if self.center_y_nm is None:
            self.center_y_nm = self.dy_nm * ny/2
            self.center_y = 1e-9*self.center_y_nm
        else:
            self.center_y = (1e-9)*self.center_y_nm


        self.omega_x = constants.ELEMENTARY_CHARGE * self.lateral_confinement_energy_x / constants.HBAR
        self.omega_y = constants.ELEMENTARY_CHARGE * self.lateral_confinement_energy_y / constants.HBAR

        self.nn_coupling_x = -constants.HBAR**2 / (2 * constants.SI_TRANSVERSE_MASS * self.dx**2) / constants.ELEMENTARY_CHARGE
        self.nn_coupling_y = -constants.HBAR**2 / (2 * constants.SI_TRANSVERSE_MASS * self.dy**2) / constants.ELEMENTARY_CHARGE


    def _confinement(self, i, j):
        """
        Horizontal confinement term at index i, j
        """
        x = i*self.dx
        y = j*self.dy

        c = (1/2)*constants.SI_TRANSVERSE_MASS*(self.omega_x**2 * (x-self.center_x)**2 + self.omega_y**2 * (y-self.center_y)**2)/constants.ELEMENTARY_CHARGE # divide by e to convert to eV
        return c
    
    def _qw_offset(self, i, j, k):
        """
        QW Offset potential term at index i, j, k
        """
        s = self.effective_lattice[i,j,k]
        return self.fractional_cb_offset(s)

    def _onsite_term(self, i, j, k):
        """
        Sum of potential terms at index i, j, k
        """
        on = self.vertical_field_potential(k) + self._confinement(i,j) + self._qw_offset(i,j,k)
        return on


    def _compute_H(self):
        """
        Construct 3D EM Hamiltonian
        """

        ham = helpers.compute_3D_H(
                nx = self.nx, 
                ny = self.ny,
                nz = self.nz, 
                onsite_func = self._onsite_term,
                nn_coupling_z = self.nn_coupling_z, 
                nnn_coupling_z = 0,
                nn_coupling_x = self.nn_coupling_x,
                nn_coupling_y = self.nn_coupling_y,
                use_sparse = self.sparse,
                periodic = self.periodic,
        )

        self._H = ham

    def wf_3D_matrix_from_vector(self, v):
        """
        Convert from 1D vector to 3D array
        """

        mat = np.zeros((self.nx, self.ny, self.nz), dtype=np.double)
        for ind in range(len(v)):
            (i,j,k) = helpers.coordinates_3D(ind, ny=self.ny, nz=self.nz)
            mat[i,j,k] = v[ind]

        return mat
    

    def _compute_inter_valley_coupling(self):
        """
        Compute the inter-valley coupling from self.effective_lattice using EM theory
        """
        _, v0 = self.solve(n_lowest_eigenstates=1)
        psi_0 = self.wf_3D_matrix_from_vector(v0[:,0])

        z_arr = 1e-9 * np.arange(self.nz) * self.dz_nm

        pot_arr = self.fractional_cb_offset(self.effective_lattice)
        coupling = np.sum(self.dz * self.dx * self.dy * np.abs(psi_0**2) * np.exp(-1j * 2 * constants.K_0 * z_arr) * pot_arr)

        self._inter_valley_coupling = coupling
        return coupling
    
    
    def sigma_delta(self):
        """
        Computing the standard deviation of the inter-valley coupling.

        Since we have the 3D model, we don't need to specify the dot size. 
        """
        _, v0 = self.solve(n_lowest_eigenstates=1)
        psi_0 = self.wf_3D_matrix_from_vector(v0[:,0])

        # In Si, there are two atoms per layer per unit cell
        nc = 2 * (self.dx * self.dy) / (constants.SI_LATTICE_CONSTANT**2)

        # Variance of the potential at each lattice site
        var_U = (1/nc) * (self.cb_offset / (self.well_si_concentration - self.bulk_si_concentration))**2 * self.effective_lattice * (1 - self.effective_lattice)

        # Variance of Delta
        vd = (self.dx * self.dy * self.dz)**2 * np.sum( psi_0**4 * var_U )

        return np.sqrt(vd)
    

    def _normalize(self, evecs):
        """
        When normalizing the envelope functions, ensure the maximum amplitude is positive
        """
        return evecs / np.linalg.norm(evecs, axis=0) / np.sqrt(self.dz * self.dx * self.dy) * np.sign(np.max(evecs, axis=0))
    


