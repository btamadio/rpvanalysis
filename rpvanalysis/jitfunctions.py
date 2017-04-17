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
def sample_from_cdf(sample,probs,bin_centers,bin_edges):
    thresh = 0.0
    for i in range(len(bin_centers)):
        thresh+=probs[i]
        if sample < thresh:
            return np.random.uniform( bin_edges[i], bin_edges[i+1] )
    return np.random.uniform(bin_edges[i],bin_edges[i+1])

@numba.jit(nopython=True)
def apply_get_dressed_mass(col_pt,col_temp_bin,templates_array,templates_neff_array,bin_centers,bin_edges,n_toys):
    n = len(col_pt)    
    n_bins = len(templates_array[0])
    assert n == len(col_temp_bin)
    result = np.zeros( (n,n_toys) )
    for i in range(n):
        probs = templates_array[col_temp_bin[i]]
        for j in range(n_toys):
            r = sample_from_cdf(np.random.random(),probs,bin_centers,bin_edges)
            result[i][j] = np.exp(r)*col_pt[i]
    return result

@numba.jit(nopython=True)
def get_uncert_bin( temp_bin ):
    eta_bin = temp_bin % 4
    bmatch = 0
    if temp_bin > 59:
        bmatch = 1
    return eta_bin + 4*bmatch

@numba.jit(nopython=True)
def apply_shift_mass(mass_matrix,col_jet_temp_bin,jet_uncert):
    n = len(col_jet_temp_bin)
    n_toys = mass_matrix.shape[1]
    assert mass_matrix.shape[0] == n
    result = np.copy(mass_matrix)#np.zeros( mass_matrix.shape )
    for i in range(n):
        uncert_bin = get_uncert_bin( col_jet_temp_bin[i] )
        for j in range(n_toys):
            result[i][j] *= (1 + jet_uncert[uncert_bin] )
    return result

@numba.jit(nopython=True)
def apply_get_mass_response(col_pt,col_eta,col_m,col_weight,col_dressed_m,pt_bins):
    n = len(col_pt)
    assert n == len(col_eta) == len(col_m) == len(col_weight) == len(col_dressed_m)
    n_toys = len(col_dressed_m[0])
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
        for j in range(n_toys):
            dressed_mean[pt_bin] += col_dressed_m[i][j]*col_weight[i]
    
    dressed_mean = dressed_mean / (sumw*n_toys)
    kin_mean = kin_mean / sumw

    #Calculate std for each pt bin
    for i in range(n):
        pt_bin = get_pt_bin(col_pt[i],pt_bins)
        kin_std[pt_bin] += col_weight[i]*(col_m[i]-kin_mean[pt_bin])*(col_m[i]-kin_mean[pt_bin])
    
    kin_std = np.sqrt(kin_std / (sumw-1))
    n_eff = sumw*sumw/sumw2
    err = kin_std/np.sqrt(n_eff)

    return(dressed_mean,kin_mean,err)
