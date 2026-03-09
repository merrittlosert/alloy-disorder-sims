This repo contains code to model single-electron quantum dots in Si/SiGe heterostructures.

These devices can be modeled in 1, 2, or 3 dimensions, and can be solved with either a two-band tight-binding model or an effective mass model.

The two-band model accounts for the position and curvature of the conduction band minima in silicon, allowing you to compute valley splittings and valley wavefunctions.

The effective mass model accounts for the curvature of the conduction band minima but does not account for the pair of low-lying minima. With the effective mass model, you can simulate envelope functions but not valley oscillations or valley splittings.

Examples of heterostructures and system solutions in 1D, 2D, and 3D are found in the notebooks folder.

If you find this code useful and would like to cite it, please cite [Losert et al., Phys. Rev. B 108, 125405](https://doi.org/10.1103/PhysRevB.108.125405).
