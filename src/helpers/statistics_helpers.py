import numpy as np
from scipy.stats import rice, rayleigh

"""
Helpers for various statistical calculations that may be useful
"""

def valley_splitting_pdf(Ev_arr: np.ndarray | float, Ev0: float, sigma_delta: float):
    """
    Returns the Rayleigh/Rice pdfs of valley splittings for a given array of values Ev_arr (eV) and dot size (nm).

    Input:
        Ev_arr (np.ndarray or float) : The valley splitting(s) at which to evaluate the pdf 
        Ev0 (float) : The deterministic valley splitting
        sigma_delta (float) : The variance of the inter-valley coupling
    """
    s = sigma_delta * np.sqrt(2)

    eps = 1e-7
    if Ev0 < eps:
        # if the inter-valley coupling is zero, then the valley splitting is just Rayleigh distributed
        pdf_arr = rayleigh.pdf(Ev_arr, scale=s)
        return pdf_arr

    else:
        pdf_arr = rice.pdf(Ev_arr, b=Ev0/s, scale=s)

    return pdf_arr

def mean_valley_splitting_disordered(sigma_delta: np.ndarray | float):
    """
    Return the mean valley splitting in the disordered regime
    """
    return np.sqrt(np.pi) * sigma_delta