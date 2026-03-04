"""
Various solvers for Si/SiGe, including the two-band tight-binding model and effective mass models
"""

from dataclasses import dataclass
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as las
from math import *
from scipy import sparse
import cmath

import vertical_profile as vp
import constants



@dataclass
class BaseSolver:
    """
    Base class for the Schrodinger equation solvers.
    """

    profile: vp.VerticalProfile
    effective_lattice: np.ndarray # Either a 1D, 2D, or 3D array of si concentrations 

    vertical_field: float = 0 # in V/nm

    n_eigenstates_calculated: int | None = None

    def __post_init__(self):

        self.n_layers = self.profile.n_layers
        self.bulk_si_concentration = self.profile.bot_cap_si_concentration
        self.well_si_concentration = self.profile.well_si_concentration
        self.dz_nm = 1e9 * constants.SI_LATTICE_CONSTANT / 4 # spacing 

        
        # Define the conduction band offset for the given profile
        y = 1-self.profile.bot_cap_si_concentration
        y_well = 1-self.profile.well_si_concentration
        self.delta_E_Si = -0.502*(1-self.profile.bot_cap_si_concentration)
        self.delta_E_Ge = 0.743 - 0.625*(1-self.profile.bot_cap_si_concentration)

        self.cb_offset = (y-y_well)*( ((1-y_well)/y)*self.delta_E_Si - (y_well/(1-y))*self.delta_E_Ge)

        # set the Hamiltonian to None so that it will be lazily computed the first time it is accessed
        self._H = None

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

        if self.n_eigenstates_calculated is not None and self.n_eigenstates_calculated >= n_lowest_eigenstates:
            return self._lowest_evals[:n_lowest_eigenstates], self._lowest_evecs[:,:n_lowest_eigenstates]

        if self.sparse:
            # H_x will be a sparse matrix
            # Negate H to let us solve for max eigenvalues
            ham_sp = -self.hamiltonian
            w, v = las.eigsh(ham_sp, k=n_lowest_eigenstates, which='LR')
            w = -w
        else:
            ham = self.hamiltonian
            w, v = la.eigh(ham)

        e_vals = np.sort(w)
        sort_index = np.argsort(w) 

        self._lowest_evals = e_vals[:n_lowest_eigenstates]
        self._lowest_evecs = v[:,sort_index[:n_lowest_eigenstates]]
        self._solved_num = n_lowest_eigenstates

        self.n_eigenstates_calculated = n_lowest_eigenstates

        return self._lowest_evals, self._lowest_evecs
    
    def valley_splitting(self):
        if self.n_eigenstates_calculated is None or self.n_eigenstates_calculated < 2:
            self.solve(n_lowest_eigenstates=2)
        
        return self._lowest_evals[1] - self._lowest_evals[0]



    @property
    def hamiltonian(self):
        """
        Lazily constructs the hamiltonian matrix the first time it is accessed, then returns the stored value on subsequent accesses
        """
        if self._H is None:
            self._compute_H()
        return self._H
    
    @property
    def effective_1D_hamiltonian(self):
        """
        Lazily constructs an effective 1D potential by averaging the onsite terms across each layer.
        """
        if self._avg_onsites_by_layer is None:
            self._compute_1D_potential_equivalent()
        return self._avg_onsites_by_layer
    





@dataclass 
class TwoBand(BaseSolver):
    """
    Base class for the two-band tight-binding solvers.
    We use the model of Boykin et al., PRB 70, 165325 (2004)
    """

    sparse: bool = True

    def __post_init__(self):
        
        # Couplings that reproduce the position and curvature of the conduction band minima
        self.nnn_coupling_z =  2 * constants.HBAR**2 / (constants.SI_LATERAL_MASS * constants.SI_LATTICE_CONSTANT**2 * (np.sin(constants.K_0 * constants.SI_LATTICE_CONSTANT / 4))**2) / constants.ELEMENTARY_CHARGE
        self.nn_coupling_z =  4 * self.nnn_coupling_z * np.cos(constants.K_0 * constants.SI_LATTICE_CONSTANT / 4)

        super().__post_init__()

  



    



@dataclass
class TwoBand_1D(TwoBand):

    def __post_init__(self):
        if len(np.shape(self.effective_lattice)) != 1:
            raise ValueError("The effective lattice should be a 1D array for the 1D model")
        
        super().__post_init__()

    def _onsite_term(self, i):
        """
        A function that takes as input the index i and returns the onsite potential from vertical field and Ge concentration terms
        """
        return self.vertical_field_potential(i) + self.fractional_cb_offset(self.effective_lattice[i])

    def _compute_H(self):
        
        ham = np.zeros((self.n_layers,self.n_layers), dtype=np.cdouble)

        for i in range(self.n_layers):
            ham[i,i] = self._onsite_term(i)

            if i-1 >= 0:
                ham[i,i-1] = self.nn_coupling_z

            if i-2 >= 0:
                ham[i,i-2] = self.nnn_coupling_z

            if i+1 < self.n_layers:
                ham[i,i+1] = self.nn_coupling_z

            if i+2 < self.n_layers:
                ham[i,i+2] = self.nnn_coupling_z
        
        if self.sparse:
            self._H = sparse.dia_matrix(ham).tocsc()

        else:
            self._H = ham


    def _compute_1D_potential_equivalent(self):
        ham = self.hamiltonian
        (nx, _) = np.shape(ham)
        onsites = np.zeros(nx, dtype=np.cdouble)

        for i in range(nx):
            onsites[i] = ham[i,i]

        self._avg_onsites_by_layer = onsites



@dataclass
class TwoBand_2D(TwoBand):

    dx_nm: float | None = None
    lateral_confinement_energy: float = 2e-3 # eV
    center_x_nm: float | None = None # center of the confinement in nm

    def __post_init__(self):
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

        super().__post_init__()

    # assign a unique index to a lattice point
    def _index(self, i, k):
        return i*self.nz + k

    def _coordinates(self, index):
        i = int(np.floor(index/(self.nz)))
        k = int(np.remainder(index,self.nz))

        return (i,k)

    def _add_element_to_diag(self, offset, ind_curr, element, diag_dict):

        ind_1 = ind_curr
        ind_2 = offset+ind_1
        ind_d = max(ind_1,ind_2)
        if offset in diag_dict.keys():
            diag_dict[offset][ind_d-abs(offset)] = element
        else:
            diag_list = np.zeros(self.nx*self.nz - abs(offset))
            diag_list[ind_d-abs(offset)] = element
            diag_dict[offset] = diag_list


    def _add_element(self, ind_1, ind_2, element, diag_dict, hamiltonian):
        if self.sparse:
            offset = ind_2 - ind_1
            self._add_element_to_diag(offset, ind_1, element, diag_dict)
        else:
            hamiltonian[ind_1,ind_2] = element


    def _add_neighbor_couplings(self, i, k, diag_dict, hamiltonian):
        ind_curr = self._index(i,k)

        # z couplings
        if k-1 >= 0:
            ind_z = self._index(i,k-1)
            self._add_element(ind_curr,ind_z,self.nn_coupling_z, diag_dict, hamiltonian)
        if k-2 >= 0:
            ind_z = self._index(i,k-2)
            self._add_element(ind_curr,ind_z,self.nnn_coupling_z, diag_dict, hamiltonian)
        if k+1 < self.nz:
            ind_z = self._index(i,k+1)
            self._add_element(ind_curr,ind_z,self.nn_coupling_z, diag_dict, hamiltonian)
        if k+2 < self.nz:
            ind_z = self._index(i,k+2)
            self._add_element(ind_curr,ind_z,self.nnn_coupling_z, diag_dict, hamiltonian)

        # x couplings
        if i-1 >= 0:
            ind_x = self._index(i-1,k)
            self._add_element(ind_curr,ind_x,self.nn_coupling_x, diag_dict, hamiltonian)
        elif self.periodic:
            ind_x = self._index(self.nx-1,k)
            self._add_element(ind_curr,ind_x,self.nn_coupling_x, diag_dict, hamiltonian)

        if i+1 < self.nx:
            ind_x = self._index(i+1,k)
            self._add_element(ind_curr,ind_x,self.nn_coupling_x, diag_dict, hamiltonian)
        elif self.periodic:
            ind_x = self._index(0,k)
            self._add_element(ind_curr,ind_x,self.nn_coupling_x, diag_dict, hamiltonian)
           


    def _confinement(self, i):
        x = i*self.dx
        c = (1/2) * constants.SI_TRANSVERSE_MASS*(self.omega_x**2 * (x-self.center_x)**2 ) / constants.ELEMENTARY_CHARGE # divide by e to convert to eV
        return c
    
    def _qw_offset(self,i,k):
        s = self.effective_lattice[i,k]
        return self.fractional_cb_offset(s)

    def _onsite_term(self,i,k):
        on = self.vertical_field_potential(k) + self._confinement(i) + self._qw_offset(i,k)
        return on



    def _compute_H(self):
        diag_dict = dict()
        main_diag = np.zeros(self.nx*self.nz)
        
        if not self.sparse:
            hamiltonian = np.zeros((self.nx*self.nz, self.nx*self.nz), dtype=np.cdouble)
        else:
            hamiltonian = None

        for i in range(self.nx):
            for k in range(self.nz):
                ind_curr = self._index(i,k)

                # onsite terms
                on = self._onsite_term(i,k)

                if self.sparse:
                    self._add_element_to_diag(0,ind_curr,on, diag_dict)
                    main_diag[ind_curr] = on
                    hamiltonian = None
                else:
                    hamiltonian[ind_curr,ind_curr] = on

                self._add_neighbor_couplings(i,k,diag_dict,hamiltonian)


        if self.sparse:
            offsets = list()
            diags = list()
            for offset in diag_dict.keys():
                offsets.append(offset)
                diags.append(diag_dict[offset])
            self._H = sparse.diags(diags, offsets=offsets, shape=(self.nx*self.nz,self.nx*self.nz))
        else:
            self._H = hamiltonian


    # return array of onsites at given layer
    def _compute_1D_potential_equivalent(self, l):
        onsites = np.zeros(self.nx)
        for i in range(self.nx):
            on = self._onsite_term(i,l)
            onsites[i] = on
        return onsites


    def wf_2D_matrix_from_vector(self, v):

        mat = np.zeros((self.nx, self.nz), dtype=np.cdouble)
        for ind in range(len(v)):
            (i,k) = self._coordinates(ind)
            mat[i,k] = v[ind]

        return mat









@dataclass
class TwoBand_3D(TwoBand):

    dx_nm: float | None = None
    dy_nm: float | None = None

    center_x_nm: float | None = None # center of the confinement (x) in nm
    center_y_nm: float | None = None # center of the confinement (y) in nm

    lateral_confinement_energy_x: float = 2e-3 # eV
    lateral_confinement_energy_y: float = 2e-3 # eV


    def __post_init__(self):
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

        super().__post_init__()


    def _nearest_neighbor_coords(self,i,j,k):
        nn_inds = list()

        if i-1 >= 0:
            nn_inds.append((i-1,j,k))
        elif self.periodic:
            nn_inds.append((self.nx-1,j,k))
        if i+1 < self.nx:
            nn_inds.append((i+1,j,k))
        elif self.periodic:
            nn_inds.append((0,j,k))

        if j-1 >= 0:
            nn_inds.append((i,j-1,k))
        elif self.periodic:
            nn_inds.append((i,self.ny-1,k))
        if j+1 < self.ny:
            nn_inds.append((i,j+1,k))
        elif self.periodic:
            nn_inds.append((i,0,k))

        if k-1 >= 0:
            nn_inds.append((i,j,k-1))
        if k+1 < self.nz:
            nn_inds.append((i,j,k+1))

        return nn_inds

        # assign a unique index to a lattice point
    def _index(self, i, j, k):
        return i*self.ny*self.nz + j*self.nz + k

    def _coordinates(self, index):
        i = int(np.floor(index/(self.ny*self.nz)))
        j = int(np.floor( np.remainder(index, (self.ny*self.nz)) / (self.nz) ))
        k = int(np.remainder(index,self.nz))

        return (i,j,k)


    def _add_element_to_diag(self, offset, ind_curr, element, diag_dict):

        ind_1 = ind_curr
        ind_2 = offset+ind_1
        ind_d = max(ind_1,ind_2)
        if offset in diag_dict.keys():
            diag_dict[offset][ind_d-abs(offset)] = element
        else:
            diag_list = np.zeros(self.nx*self.ny*self.nz - abs(offset))
            diag_list[ind_d-abs(offset)] = element
            diag_dict[offset] = diag_list


    def _add_element(self, ind_1, ind_2, element, diag_dict, hamiltonian):
        if self.sparse:
            offset = ind_2 - ind_1
            self._add_element_to_diag(offset, ind_1, element, diag_dict)
        else:
            hamiltonian[ind_1,ind_2] = element


    def _add_neighbor_couplings(self, i, j, k, diag_dict, hamiltonian):
        ind_curr = self._index(i,j,k)

        # z couplings
        if k-1 >= 0:
            ind_z = self._index(i,j,k-1)
            self._add_element(ind_curr,ind_z,self.nn_coupling_z, diag_dict, hamiltonian)
        if k-2 >= 0:
            ind_z = self._index(i,j,k-2)
            self._add_element(ind_curr,ind_z,self.nnn_coupling_z, diag_dict, hamiltonian)
        if k+1 < self.nz:
            ind_z = self._index(i,j,k+1)
            self._add_element(ind_curr,ind_z,self.nn_coupling_z, diag_dict, hamiltonian)
        if k+2 < self.nz:
            ind_z = self._index(i,j,k+2)
            self._add_element(ind_curr,ind_z,self.nnn_coupling_z, diag_dict, hamiltonian)

        # x couplings
        if i-1 >= 0:
            ind_x = self._index(i-1,j,k)
            self._add_element(ind_curr,ind_x,self.nn_coupling_x, diag_dict, hamiltonian)
        elif self.periodic:
            ind_x = self._index(self.nx-1,j,k)
            self._add_element(ind_curr,ind_x,self.nn_coupling_x, diag_dict, hamiltonian)

        if i+1 < self.nx:
            ind_x = self._index(i+1,j,k)
            self._add_element(ind_curr,ind_x,self.nn_coupling_x, diag_dict, hamiltonian)
        elif self.periodic:
            ind_x = self._index(0,j,k)
            self._add_element(ind_curr,ind_x,self.nn_coupling_x, diag_dict, hamiltonian)
           
        # y couplings
        if j-1 >= 0:
            ind_y = self._index(i,j-1,k)
            self._add_element(ind_curr,ind_y,self.nn_coupling_y, diag_dict, hamiltonian)
        elif self.periodic:
            ind_y = self._index(i,self.ny-1,k)
            self._add_element(ind_curr,ind_y,self.nn_coupling_y, diag_dict, hamiltonian)

        if j+1 < self.ny:
            ind_y = self._index(i,j+1,k)
            self._add_element(ind_curr,ind_y,self.nn_coupling_y, diag_dict, hamiltonian)
        elif self.periodic:
            ind_y = self._index(i,0,k)
            self._add_element(ind_curr,ind_y,self.nn_coupling_y, diag_dict, hamiltonian)



    def _confinement(self, i, j):
        if self.periodic == False:
            x = i*self.dx
            y = j*self.dy

            c = (1/2)*constants.SI_TRANSVERSE_MASS*(self.omega_x**2 * (x-self.center_x)**2 + self.omega_y**2 * (y-self.center_y)**2)/constants.ELEMENTARY_CHARGE # divide by e to convert to eV
            return c
        else:
            return 0
        

    def _qw_offset(self,i,j,k):
        s = self.effective_lattice[i,j,k]
        return self.fractional_cb_offset(s)

    def _onsite_term(self,i,j,k):
        on = self.vertical_field_potential(k) + self._confinement(i,j) + self._qw_offset(i,j,k)
        return on


    def _compute_H(self):
        diag_dict = dict()
        main_diag = np.zeros(self.nx*self.ny*self.nz)
        
        if not self.sparse:
            hamiltonian = np.zeros((self.nx*self.ny*self.nz, self.nx*self.ny*self.nz), dtype=np.cdouble)
        else:
            hamiltonian = None

        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    ind_curr = self._index(i,j,k)

                    # onsite terms
                    on = self._onsite_term(i,j,k)

                    if self.sparse:
                        self._add_element_to_diag(0,ind_curr,on, diag_dict)
                        main_diag[ind_curr] = on
                        hamiltonian = None
                    else:
                        hamiltonian[ind_curr,ind_curr] = on


                    self._add_neighbor_couplings(i,j,k,diag_dict,hamiltonian)

   

        if self.sparse:
            offsets = list()
            diags = list()
            for offset in diag_dict.keys():
                offsets.append(offset)
                diags.append(diag_dict[offset])
            self._H = sparse.diags(diags, offsets=offsets, shape=(self.nx*self.ny*self.nz,self.nx*self.ny*self.nz))
        else:
            self._H = hamiltonian




    def _compute_1D_potential_equivalent(self):

        diag_dict = dict()
        main_diag = np.zeros(self.nx*self.ny*self.nz)

        onsites_by_layer = np.zeros(self.nz, dtype=np.cdouble)
        
        if not self.sparse:
            hamiltonian = np.zeros((self.nx*self.ny*self.nz, self.nx*self.ny*self.nz), dtype=np.cdouble)
        else:
            hamiltonian = None

        for k in range(self.nz):
            avg_onsite = 0

            for i in range(self.nx):
                for j in range(self.ny):

                    # Compute the average onsite term
                    on = self._onsite_term(i,j,k)
                    avg_onsite += on

                    self._add_neighbor_couplings(i,j,k,diag_dict,hamiltonian)

   

            # Now, add the average onsite term to each atom in the layer
            avg_onsite = avg_onsite/(self.nx*self.ny)

            onsites_by_layer[k] = avg_onsite

            for i in range(self.nx):
                for j in range(self.ny):
                    ind_curr = self._index(i,j,k)
                    if self.sparse:
                        self._add_element_to_diag(0,ind_curr, avg_onsite, diag_dict)
                        main_diag[ind_curr] = avg_onsite
                        hamiltonian = None
                    else:
                        hamiltonian[ind_curr,ind_curr] = avg_onsite

        if self.sparse:
            offsets = list()
            diags = list()
            for offset in diag_dict.keys():
                offsets.append(offset)
                diags.append(diag_dict[offset])
            self._H_x_avg_onsite = sparse.diags(diags, offsets=offsets, shape=(self.nx*self.ny*self.nz,self.nx*self.ny*self.nz))
        else:
            self._H_x_avg_onsite = hamiltonian

        self._avg_onsites_by_layer = onsites_by_layer

    


    def wf_3D_matrix_from_vector(self, v):

        mat = np.zeros((self.nx, self.ny, self.nz), dtype=np.cdouble)
        for ind in range(len(v)):
            (i,j,k) = self._coordinates(ind)
            mat[i,j,k] = v[ind]

        return mat



    def wf_3D_matrix_from_vector(self, v):

        mat = np.zeros((self.nx, self.ny, self.nz), dtype=np.cdouble)
        for ind in range(len(v)):
            (i,j,k) = self._coordinates(ind)
            mat[i,j,k] = v[ind]

        return mat

        


















