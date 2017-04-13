import numpy as np

class template:
    def __init__(self,n_bins,df_temps,temp_bin,x_min=-7,x_max=0):
        self.sumw,self.bin_edges=np.histogram(a=[],range=(x_min,x_max),bins=n_bins)
        self.sumw2,self.bin_edges2=np.histogram(a=[],range=(x_min,x_max),bins=n_bins)

        self.sumw=self.sumw.astype(float)
        self.sumw2=self.sumw2.astype(float)
        
        for i in range(len(df_temps)):
            try:
                this_df = df_temps[i].ix[temp_bin]
            except:
                continue
            temp_vals = np.log(this_df['jet_m_'+str(i+1)]/this_df['jet_pt_'+str(i+1)])
            temp_wts = this_df.weight
            temp_sumw,temp_bin_edges = np.histogram(a=temp_vals,weights=temp_wts,range=(x_min,x_mx),bins=n_bins)
            temp_sumw2,temp_bin_edges2 = np.histogram(a=temp_vals,weights=temp_wts**2,range=(x_min,x_max),bins=n_bins)
            self.sumw += temp_sumw
            self.sumw2 += temp_sumw2
        #normalize to 1 so it can be used as PDF
        self.probs = self.sumw / self.sumw.sum()
        assert abs(self.probs.sum() - 1) < 1e-6

def get_region_index(df,region_string,eta_min=0,eta_max=2):
    #Given a region string, return a list corresponding to the index of jets for that region
    mask = None
    njet=0
    if region_string.startswith('3j'):
        mask = df['njet']==3
        njet=3
    elif region_string.startswith('4j'):
        mask = df['njet']==4
        njet=4
    elif region_string.startswith('5j'):
        mask = df['njet']>=5
        njet=4
    else:
        print('Error: region name %s is invalid. Must start with 3j,4j, or 5j'%region_string)
        return np.array([])

    if 's0' in region_string:
        mask &= df['njet_soft'] == 0
    elif 's1' in region_string:
        mask &= df['njet_soft'] >= 1

    if 'VR' in region_string:
        mask &= df['dEta'] > 1.4
    elif 'SR' in region_string:
        mask &= df['dEta'] < 1.4

    if 'b0' in region_string:
        mask &= df['nbjet'] == 0
    elif 'b1' in region_string:
        mask &= df['nbjet'] >= 1

    #If we don't select on any jet-level observables, just return the list of indices as-is
    if (not 'bU' in region_string) and (not 'bM' in region_string) and eta_min == 0 and eta_max == 2:
        return [ df[mask].index for _ in range(njet) ]

    #If we ask for eta cuts or b-matching, need to get different indices for each jet
    masks = [ mask for _ in range(njet)]
    if 'bU' in region_string:
        for i in range(njet):
            jet_i = i+1
            masks[i] &= df['jet_bmatched_'+str(jet_i)] == 0

    elif 'bM' in region_string:
        for i in range(njet):
            jet_i = i+1
            masks[i] &= df['jet_bmatched_'+str(jet_i)] == 1

    for i in range(njet):
        jet_i = i+1
        masks[i] &= df['jet_eta_'+str(jet_i)].apply(np.abs) > eta_min
    
    for i in range(njet):
        jet_i = i+1
        masks[i] &= df['jet_eta_'+str(jet_i)].apply(np.abs) < eta_max

    return [ df[mask].index for mask in masks ]
