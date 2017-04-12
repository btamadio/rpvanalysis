import numba
import numpy as np

#These functions are optimized with numba.jit
@numba.jit(nopython=True)
def get_temp_bin(pt,eta,bmatch):
    pt_bins = np.array([0.2,0.221,0.244,0.270,0.293,0.329,0.364,0.402,0.445,0.492,0.544,0.6,0.644,0.733,0.811,13])
    eta_bins = np.array([0.0,0.5,1.0,1.5,2.0])
    pt_bin = -1
    eta_bin = -1
    for i in range(len(pt_bins)-1):
        if pt > pt_bins[i] and pt <= pt_bins[i+1]:
            pt_bin = i
    for i in range(len(eta_bins)-1):
        if eta > eta_bins[i] and eta <= eta_bins[i+1]:
            eta_bin = i
    if pt_bin == -1 or eta_bin == -1:
        return -1
    return 60*bmatch+4*pt_bin+eta_bin

@numba.jit(nopython=True)
def apply_get_temp_bin(col_pt,col_eta,col_bmatch):
    #Loop over series of pt, eta, and b-match and return list of template indices
    n=len(col_bmatch)
    result = np.zeros(n,dtype=np.int32)
    assert len(col_pt)==len(col_eta)==n
    for i in range(n):
        result[i] = get_temp_bin(col_pt[i],abs(col_eta[i]),col_bmatch[i])
    return result

@numba.jit(nopython=True)
def sample_from_cdf(sample,probs,bin_centers):
    thresh = 0.0
    for i in range(len(bin_centers)):
        thresh+=probs[i]
        if sample < thresh:
            return bin_centers[i] 
    return bin_centers[i]

@numba.jit(nopython=True)
def apply_get_dressed_mass(col_pt,col_temp_bin,probs_array,bin_centers_array,n_toys):
    n = len(col_pt)    
    assert n == len(col_temp_bin)
    result = np.zeros( (n,n_toys) )
    for i in range(n):
        probs = probs_array[col_temp_bin[i]]
        bin_centers = bin_centers_array[col_temp_bin[i]]
        for j in range(n_toys):
            r = sample_from_cdf(np.random.random(),probs,bin_centers)
            result[i][j] = np.exp(r)*col_pt[i]
    return result
