import numba
import numpy as np
#These functions are optimized with numba.jit
@numba.jit(nopython=True)
def get_pt_bin(pt,pt_bins):
    if pt > pt_bins[-2]:
        return(len(pt_bins)-2)
    for i in range(len(pt_bins)-1):
        if pt > pt_bins[i] and pt <= pt_bins[i+1]:
            return i
    return -1

@numba.jit(nopython=True)
def get_temp_bin(pt,eta,bmatch,pt_bins,eta_bins):
    #pt_bins = np.array([0.2,0.221,0.244,0.270,0.293,0.329,0.364,0.402,0.445,0.492,0.544,0.6,0.644,0.733,0.811,13])
    #eta_bins = np.array([0.0,0.5,1.0,1.5,2.0])
    pt_bin = -1
    eta_bin = -1
    if pt > pt_bins[-2]:
        pt_bin = len(pt_bins)-2
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
def apply_get_temp_bin(col_pt,col_eta,col_bmatch,pt_bins,eta_bins):
    #Loop over series of pt, eta, and b-match and return list of template indices
    n=len(col_bmatch)
    result = np.zeros(n,dtype=np.int32)
    assert len(col_pt)==len(col_eta)==n
    for i in range(n):
        result[i] = get_temp_bin(col_pt[i],abs(col_eta[i]),col_bmatch[i],pt_bins,eta_bins)
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

@numba.jit(nopython=True)
def apply_get_mass_response(col_pt,col_eta,col_m,col_weight,col_dressed_m,pt_bins):
    n = len(col_pt)
    assert n == len(col_eta) == len(col_m) == len(col_weight) == len(col_dressed_m)

    dressed_mean = np.zeros(len(pt_bins) - 1)
    kin_mean = np.zeros(len(pt_bins) - 1)
    kin_std = np.zeros( len(pt_bins) -1)

    sumw = np.zeros(len(pt_bins) - 1)
    sumw2 = np.zeros(len(pt_bins) -1) 
    err = np.zeros(len(pt_bins) -1) 

    #Calculate x_bar for each pt bin
    for i in range(n):
        pt_bin = get_pt_bin(col_pt[i],pt_bins)
        
        kin_mean[pt_bin] += col_m[i]*col_weight[i]
        sumw[pt_bin] += col_weight[i]
        sumw2[pt_bin] += col_weight[i]*col_weight[i]

        dressed_mean[pt_bin] += col_dressed_m[i]*col_weight[i]
    
    dressed_mean = dressed_mean / sumw
    kin_mean = kin_mean / sumw

    #Calculate std for each pt bin
    for i in range(n):
        pt_bin = get_pt_bin(col_pt[i],pt_bins)
        kin_std += col_weight[i]*(col_m[i]-kin_mean[pt_bin])*(col_m[i]-kin_mean[pt_bin])
    
    kin_std = np.sqrt(kin_std / (sumw-1))
    n_eff = sumw*sumw/sumw2
    err = kin_std/np.sqrt(n_eff)

    return(dressed_mean,kin_mean,err)
