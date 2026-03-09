import numpy as np
from scipy import sparse


"""
Helpers for the 1D potential solvers
"""

def compute_1D_H(
        nz: int, 
        onsite_func: callable, 
        nn_coupling_z: float, 
        nnn_coupling_z: float, 
        use_sparse: bool
):
    """
    Constructs the 1D Hamiltonian.

    Inputs:
        nz (int) : Number of cells along z in the Hamiltonian matrix
        onsite_func (callable) : A function that takes index (i) and returns the onsite potential
        nn_coupling_z (float) : The nearest-neighbor coupling along z
        nnn_coupling_z (float) : The next-nearest-neighbor coupling along z
        use_sparse (bool) : Whether or not to construct a sparse matrix 

    Returns:
        Either the Hamiltonian matrix (1D np.ndarray) or a sparse list of diagonals, depending on the value of use_sparse
    """
    
    ham = np.zeros((nz, nz), dtype=np.double)

    for i in range(nz):
            ham[i,i] = onsite_func(i)

            if i-1 >= 0:
                ham[i,i-1] = nn_coupling_z

            if i-2 >= 0:
                ham[i,i-2] = nnn_coupling_z

            if i+1 < nz:
                ham[i,i+1] = nn_coupling_z

            if i+2 < nz:
                ham[i,i+2] = nnn_coupling_z
        
    
    if use_sparse:
        H = sparse.dia_matrix(ham).tocsc()

    else:
        H = ham

    return H





"""
Helpers for the 2D potential solvers
"""


def _add_element_to_diag(
        offset: int, 
        ind_curr: int, 
        element: float, 
        diag_dict: dict,
        tot_num_elements: int
):
    """
    Adds an element to the diagonal with offset

    Inputs:
        offset (int) : The offset of the diagonal
        ind_curr (int) : The current index
        element (float) : The value of the element to add
        diag_dict (dict) : A dict mapping offsets to list of diagonal elements
        tot_num_elements (int) : The total number of elements in the Hamiltonian
    """

    ind_1 = ind_curr
    ind_2 = offset+ind_1
    ind_d = max(ind_1,ind_2)
    if offset in diag_dict.keys():
        diag_dict[offset][ind_d-abs(offset)] = element
    else:
        diag_list = np.zeros(tot_num_elements - abs(offset))
        diag_list[ind_d-abs(offset)] = element
        diag_dict[offset] = diag_list


def _add_element(
        ind_1: int, 
        ind_2: int, 
        element: float, 
        diag_dict: dict, 
        hamiltonian: np.ndarray | None, 
        use_sparse: bool, 
        tot_num_elements: int
):
    """
    Adds an element coupling ind_1 and ind_2 to the Hamiultonian

    Inputs:
        ind_1 (int) : The left-index of the term in the Hamiltonian
        ind_2 (int) : The right-index of the term in the Hamiltonian
        element (float) : The value of the element to add
        diag_dict (dict) : A dict mapping offsets to list of diagonal elements
        hamiltonian (np.ndarray | None) : If not use_sparse, contains the N-D Hamiltonian
        use_sparse (bool) : Whether or not to use sparse methods 
        tot_num_elements (int) : The total number of elements in the Hamiltonian
    """

    if use_sparse:
        offset = ind_2 - ind_1
        _add_element_to_diag(offset, ind_1, element, diag_dict, tot_num_elements=tot_num_elements)
    else:
        hamiltonian[ind_1,ind_2] = element






def coordinates_2D(index: int, nz: int):
    """
    Returns the 2D coordinates of a cell with 1D index

    Inputs:
        index (int) : 1D index of the cell
        nz (int) : Number of cells along z in the Hamiltonian
    """
    i = int(np.floor(index/(nz)))
    k = int(np.remainder(index, nz))

    return (i,k)


def _index_2D(i: int, k: int, nz: int) -> int:
    """
    Returns the 1D index of a cell with 2D coordinates

    Inputs:
        i (int) : The x-index of the cell
        k (int) : The z-index of the cell
        nz (int) : Number of cells along z in the Hamiltonian
    """
    return i*nz + k


def _add_neighbor_couplings_2D(
        i: int, 
        k: int, 
        diag_dict: dict, 
        hamiltonian: np.ndarray | None,
        nn_coupling_z: float,
        nnn_coupling_z: float,
        nn_coupling_x: float,
        nx: int,
        nz: int,
        periodic: bool,
        use_sparse: bool,
    ):
    """
    Adds coupling to indices (i, k) in a 2D Hamiltonian

    Inputs:
        i (int) : x-index of the coupling term
        k (int) : z-index of the coupling term
        diag_dict (dict) : dict containing the diagonals of a sparse matrix at various offsets
        hamiltonian (np.ndarray or None) : The 2D Hamiltonian matrix, if use_sparse = False
        nn_coupling_z (float) : The nearest-neighbor coupling along z
        nnn_coupling_z (float) : The next-nearest-neighbor coupling along z
        nn_coupling_x (float) : The nearest-neighbor coupling along x
        nx (int) : Number of cells along x in the Hamiltonian matrix
        nz (int) : Number of cells along z in the Hamiltonian matrix
        periodic (bool) : Whether or not to use periodic BCs
        use_sparse (bool) : Whether or not to construct a sparse matrix 
    """

    ind_curr = _index_2D(i, k, nz)

    # z couplings
    if k-1 >= 0:
        ind_z = _index_2D(i, k-1, nz)
        _add_element(ind_curr, ind_z, nn_coupling_z, diag_dict, hamiltonian, use_sparse=use_sparse, tot_num_elements=nx*nz)
    if k-2 >= 0:
        ind_z = _index_2D(i, k-2, nz)
        _add_element(ind_curr, ind_z, nnn_coupling_z, diag_dict, hamiltonian, use_sparse=use_sparse, tot_num_elements=nx*nz)
    if k+1 < nz:
        ind_z = _index_2D(i, k+1, nz)
        _add_element(ind_curr, ind_z, nn_coupling_z, diag_dict, hamiltonian, use_sparse=use_sparse, tot_num_elements=nx*nz)
    if k+2 < nz:
        ind_z = _index_2D(i, k+2, nz)
        _add_element(ind_curr, ind_z, nnn_coupling_z, diag_dict, hamiltonian, use_sparse=use_sparse, tot_num_elements=nx*nz)

    # x couplings
    if i-1 >= 0:
        ind_x = _index_2D(i-1, k, nz)
        _add_element(ind_curr, ind_x, nn_coupling_x, diag_dict, hamiltonian, use_sparse=use_sparse, tot_num_elements=nx*nz)
    elif periodic:
        ind_x = _index_2D(nx-1, k, nz)
        _add_element(ind_curr, ind_x, nn_coupling_x, diag_dict, hamiltonian, use_sparse=use_sparse, tot_num_elements=nx*nz)

    if i+1 < nx:
        ind_x = _index_2D(i+1, k, nz)
        _add_element(ind_curr, ind_x, nn_coupling_x, diag_dict, hamiltonian, use_sparse=use_sparse, tot_num_elements=nx*nz)
    elif periodic:
        ind_x = _index_2D(0, k, nz)
        _add_element(ind_curr, ind_x, nn_coupling_x, diag_dict, hamiltonian, use_sparse=use_sparse, tot_num_elements=nx*nz)
        





def compute_2D_H(
        nx: int, 
        nz: int, 
        onsite_func: callable,
        nn_coupling_z: float, 
        nnn_coupling_z: float,
        nn_coupling_x: float,
        use_sparse: bool,
        periodic: bool,
):
    
    """
    Constructs the 2D Hamiltonian.

    Inputs:
        nx (int) : Number of cells along x in the Hamiltonian matrix
        nz (int) : Number of cells along z in the Hamiltonian matrix
        onsite_func (callable) : A function that takes indices (i, k) and returns the onsite potential
        nn_coupling_z (float) : The nearest-neighbor coupling along z
        nnn_coupling_z (float) : The next-nearest-neighbor coupling along z
        nn_coupling_x (float) : The nearest-neighbor coupling along x
        use_sparse (bool) : Whether or not to construct a sparse matrix 
        periodic (bool) : Whether or not to use periodic BCs

    Returns:
        Either the Hamiltonian matrix (2D np.ndarray) or a sparse list of diagonals, depending on the value of use_sparse
    
    """
    

    diag_dict = dict()
    main_diag = np.zeros(nx * nz)
    
    if not use_sparse:
        hamiltonian = np.zeros((nx*nz, nx*nz), dtype=np.double)
    else:
        hamiltonian = None

    for i in range(nx):
        for k in range(nz):
            ind_curr = _index_2D(i, k, nz)

            # onsite terms
            on = onsite_func(i,k)

            if use_sparse:
                _add_element_to_diag(
                    offset = 0, 
                    ind_curr = ind_curr, 
                    element = on, 
                    diag_dict = diag_dict, 
                    tot_num_elements = nx*nz
                )
 
                main_diag[ind_curr] = on
            else:
                hamiltonian[ind_curr,ind_curr] = on

            _add_neighbor_couplings_2D(
                i = i, 
                k = k, 
                diag_dict = diag_dict, 
                hamiltonian = hamiltonian,
                nn_coupling_z = nn_coupling_z,
                nnn_coupling_z = nnn_coupling_z,
                nn_coupling_x = nn_coupling_x,
                nx = nx, 
                nz = nz,
                periodic = periodic,
                use_sparse = use_sparse
            )


    if use_sparse:
        offsets = list()
        diags = list()
        for offset in diag_dict.keys():
            offsets.append(offset)
            diags.append(diag_dict[offset])
        ham = sparse.diags(diags, offsets=offsets, shape=(nx*nz, nx*nz))
    else:
        ham = hamiltonian

    return ham







"""
Helpers for the 3D potential solvers
"""


    # assign a unique index to a lattice point
def _index_3D(i: int, j: int, k: int, ny: int, nz: int) -> int:
    """
    Returns the 1D index of a cell with 3D coordinates

    Inputs:
        i (int) : The x-index of the cell 
        j (int) : The y-index of the cell
        k (int) : The z-index of the cell
        ny (int) : Number of cells along y in the Hamiltonian
        nz (int) : Number of cells along z in the Hamiltonian
    """
    return i*ny*nz + j*nz + k

def coordinates_3D(index: int, ny: int, nz: int):
    """
    Returns the 3D coordinates of a cell with 1D index

    Inputs:
        index (int) : 1D index of the cell
        ny (int) : Number of cells along y in the Hamiltonian
        nz (int) : Number of cells along z in the Hamiltonian
    """
    i = int(np.floor(index/(ny*nz)))
    j = int(np.floor( np.remainder(index, (ny*nz)) / (nz) ))
    k = int(np.remainder(index,nz))

    return (i,j,k)




def _add_neighbor_couplings_3D(
    i: int, 
    j: int,
    k: int, 
    diag_dict: dict, 
    hamiltonian: np.ndarray | None,
    nn_coupling_z: float,
    nnn_coupling_z: float,
    nn_coupling_x: float,
    nn_coupling_y: float,
    nx: int,
    ny: int,
    nz: int,
    periodic: bool,
    use_sparse: bool,
):
    """
    Adds coupling to indices (i, j, k) in a 2D Hamiltonian

    Inputs:
        i (int) : x-index of the coupling term
        j (int) : y-index of the coupling term
        k (int) : z-index of the coupling term
        diag_dict (dict) : dict containing the diagonals of a sparse matrix at various offsets
        hamiltonian (np.ndarray or None) : The 2D Hamiltonian matrix, if use_sparse = False
        nn_coupling_z (float) : The nearest-neighbor coupling along z
        nnn_coupling_z (float) : The next-nearest-neighbor coupling along z
        nn_coupling_x (float) : The nearest-neighbor coupling along x
        nn_coupling_y (float) : The nearest-neighbor coupling along y
        nx (int) : Number of cells along x in the Hamiltonian matrix
        ny (int) : Number of cells along y in the Hamiltonian matrix
        nz (int) : Number of cells along z in the Hamiltonian matrix
        periodic (bool) : Whether or not to use periodic BCs
        use_sparse (bool) : Whether or not to construct a sparse matrix 
    """

    ind_curr = _index_3D(i, j, k, ny, nz)

    # z couplings
    if k-1 >= 0:
        ind_z = _index_3D(i, j, k-1, ny=ny, nz=nz)
        _add_element(ind_curr, ind_z, nn_coupling_z, diag_dict, hamiltonian, use_sparse=use_sparse, tot_num_elements=nx*ny*nz)
    if k-2 >= 0:
        ind_z = _index_3D(i, j, k-2, ny=ny, nz=nz)
        _add_element(ind_curr, ind_z, nnn_coupling_z, diag_dict, hamiltonian, use_sparse=use_sparse, tot_num_elements=nx*ny*nz)
    if k+1 < nz:
        ind_z = _index_3D(i, j, k+1, ny=ny, nz=nz)
        _add_element(ind_curr,ind_z, nn_coupling_z, diag_dict, hamiltonian, use_sparse=use_sparse, tot_num_elements=nx*ny*nz)
    if k+2 < nz:
        ind_z = _index_3D(i, j, k+2, ny=ny, nz=nz)
        _add_element(ind_curr,ind_z, nnn_coupling_z, diag_dict, hamiltonian, use_sparse=use_sparse, tot_num_elements=nx*ny*nz)

    # x couplings
    if i-1 >= 0:
        ind_x = _index_3D(i-1, j, k, ny=ny, nz=nz)
        _add_element(ind_curr, ind_x, nn_coupling_x, diag_dict, hamiltonian, use_sparse=use_sparse, tot_num_elements=nx*ny*nz)
    elif periodic:
        ind_x = _index_3D(nx-1, j, k, ny=ny, nz=nz)
        _add_element(ind_curr, ind_x, nn_coupling_x, diag_dict, hamiltonian, use_sparse=use_sparse, tot_num_elements=nx*ny*nz)

    if i+1 < nx:
        ind_x = _index_3D(i+1, j, k, ny=ny, nz=nz)
        _add_element(ind_curr, ind_x, nn_coupling_x, diag_dict, hamiltonian, use_sparse=use_sparse, tot_num_elements=nx*ny*nz)
    elif periodic:
        ind_x = _index_3D(0, j, k, ny=ny, nz=nz)
        _add_element(ind_curr, ind_x, nn_coupling_x, diag_dict, hamiltonian, use_sparse=use_sparse, tot_num_elements=nx*ny*nz)
        
    # y couplings
    if j-1 >= 0:
        ind_y = _index_3D(i, j-1, k, ny=ny, nz=nz)
        _add_element(ind_curr, ind_y, nn_coupling_y, diag_dict, hamiltonian, use_sparse=use_sparse, tot_num_elements=nx*ny*nz)
    elif periodic:
        ind_y = _index_3D(i, ny-1, k, ny=ny, nz=nz)
        _add_element(ind_curr,ind_y, nn_coupling_y, diag_dict, hamiltonian, use_sparse=use_sparse, tot_num_elements=nx*ny*nz)

    if j+1 < ny:
        ind_y = _index_3D(i, j+1, k, ny=ny, nz=nz)
        _add_element(ind_curr, ind_y, nn_coupling_y, diag_dict, hamiltonian, use_sparse=use_sparse, tot_num_elements=nx*ny*nz)
    elif periodic:
        ind_y = _index_3D(i, 0, k, ny=ny, nz=nz)
        _add_element(ind_curr, ind_y, nn_coupling_y, diag_dict, hamiltonian, use_sparse=use_sparse, tot_num_elements=nx*ny*nz)




def compute_3D_H(
        nx: int, 
        ny: int,
        nz: int, 
        onsite_func: callable,
        nn_coupling_z: float, 
        nnn_coupling_z: float,
        nn_coupling_x: float,
        nn_coupling_y: float,
        use_sparse: bool,
        periodic: bool,
):
    """
    Constructs the 3D Hamiltonian.

    Inputs:
        nx (int) : Number of cells along x in the Hamiltonian matrix
        ny (int) : Number of cells along y in the Hamiltonian matrix
        nz (int) : Number of cells along z in the Hamiltonian matrix
        onsite_func (callable) : A function that takes indices (i, j, k) and returns the onsite potential
        nn_coupling_z (float) : The nearest-neighbor coupling along z
        nnn_coupling_z (float) : The next-nearest-neighbor coupling along z
        nn_coupling_x (float) : The nearest-neighbor coupling along x
        nn_coupling_y (float) : The nearest-neighbor coupling along y
        use_sparse (bool) : Whether or not to construct a sparse matrix 
        periodic (bool) : Whether or not to use periodic BCs

    Returns:
        Either the Hamiltonian matrix (3D np.ndarray) or a sparse list of diagonals, depending on the value of use_sparse
    
    """
    diag_dict = dict()
    main_diag = np.zeros(nx * ny * nz)
    tot_num_elements = nx * ny * nz
    
    if not use_sparse:
        hamiltonian = np.zeros((nx*ny*nz, nx*ny*nz), dtype=np.double)
    else:
        hamiltonian = None

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                ind_curr = _index_3D(i, j, k, ny, nz)

                # onsite terms
                on = onsite_func(i,j,k)

                if use_sparse:
                    _add_element_to_diag(0, ind_curr, on, diag_dict, tot_num_elements=tot_num_elements)
                    main_diag[ind_curr] = on
                else:
                    hamiltonian[ind_curr,ind_curr] = on


                _add_neighbor_couplings_3D(
                        i = i, 
                        j = j,
                        k = k, 
                        diag_dict = diag_dict, 
                        hamiltonian = hamiltonian,
                        nn_coupling_z = nn_coupling_z,
                        nnn_coupling_z = nnn_coupling_z,
                        nn_coupling_x = nn_coupling_x,
                        nn_coupling_y = nn_coupling_y,
                        nx = nx,
                        ny = ny,
                        nz = nz,
                        periodic = periodic,
                        use_sparse = use_sparse,
                )



    if use_sparse:
        offsets = list()
        diags = list()
        for offset in diag_dict.keys():
            offsets.append(offset)
            diags.append(diag_dict[offset])
        ham = sparse.diags(diags, offsets=offsets, shape=(nx*ny*nz, nx*ny*nz))
    else:
        ham = hamiltonian

    return ham