from __future__ import print_function
import numpy as np
class template:
    def __init__(self,n_bins,df_temps,temp_bin,x_min=-7,x_max=0):
        self.sumw_neg,bin_edges_neg=np.histogram(a=[],range=(x_min,x_max),bins=n_bins)
        self.sumw,self.bin_edges=np.histogram(a=[],range=(x_min,x_max),bins=n_bins)
        self.sumw2,self.bin_edges2=np.histogram(a=[],range=(x_min,x_max),bins=n_bins)

        self.sumw_neg=self.sumw_neg.astype(float)
        self.sumw=self.sumw.astype(float)
        self.sumw2=self.sumw2.astype(float)

        self.bin_centers = np.zeros(n_bins).astype(float)
        for i in range(n_bins):
            self.bin_centers[i] = (self.bin_edges[i+1]+self.bin_edges[i])/2

        njets=0
        for i in range(len(df_temps)):
            try:
                this_df = df_temps[i].ix[temp_bin]
                njets+=len(this_df)
            except:
                continue
            temp_vals = np.log(this_df['jet_m_'+str(i+1)]/this_df['jet_pt_'+str(i+1)])
            temp_wts = this_df.weight
            temp_sumw,temp_bin_edges = np.histogram(a=temp_vals,weights=temp_wts,range=(x_min,x_max),bins=n_bins)
            temp_sumw2,temp_bin_edges2 = np.histogram(a=temp_vals,weights=temp_wts*temp_wts,range=(x_min,x_max),bins=n_bins)
            self.sumw_neg += temp_sumw
            self.sumw += temp_sumw
            self.sumw2 += temp_sumw2
        if njets==0:
            print('Warning: no jets in template bin ',temp_bin)
            self.probs=np.zeros( self.sumw.shape[0] )
            self.probs_neg =np.zeros(self.sumw.shape[0])
            self.n_eff = np.zeros(self.sumw.shape[0])
            return
        #normalize to 1 so it can be used as PDF
        for i,sumw in enumerate(self.sumw):
            if sumw<0:
                print ('Warning: setting negative template bin height to zero.')
                print ('   template %i, bin %i, value %.2f'%(temp_bin,i,sumw) )
                self.sumw[i] = 0
        self.probs = self.sumw / self.sumw.sum()
        self.probs_neg = self.sumw_neg / self.sumw.sum()
        self.n_eff = self.sumw*self.sumw/self.sumw2
        self.n_eff[ self.n_eff == np.nan ] = 0

def get_region_index(df,region_string,eta_bins):
    #Given a region string, return a list corresponding to the index of jets for that region

    if region_string.startswith('UDR1'):
        region_string = '2jLJG400'+region_string[4:]
    elif region_string.startswith('UDR2'):
        region_string = '4js1LJL400'+region_string[4:]

    mask = df['njet']>0
    njet=4

    if region_string.startswith('2j'):
        mask = df['njet']==2
        njet=2
    elif region_string.startswith('3j'):
        mask = df['njet']==3
        njet=3
    elif region_string.startswith('4j'):
        mask = df['njet']==4
        njet=4
    elif region_string.startswith('5j'):
        mask = df['njet']>=5
        njet=4

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

    if 'LJL400' in region_string:
        mask &= df['jet_pt_1'] < 0.4
    elif 'LJG400' in region_string:
        mask &= df['jet_pt_1'] > 0.4
    
    #if not jet-level observables, we are done
    if not any( substr in region_string for substr in ['e1','e2','e3','e4','bU','bM']):
        return [ df[mask].index for _ in range(njet) ]

    #b-match selection
    masks = [ mask for _ in range(njet)]

    if 'bU' in region_string:
        for i in range(njet):
            jet_i = i+1
            masks[i] &= df['jet_bmatched_'+str(jet_i)] == 0
    elif 'bM' in region_string:
        for i in range(njet):
            jet_i = i+1
            masks[i] &= df['jet_bmatched_'+str(jet_i)] == 1
    if not any (substr in region_string for substr in ['e1','e2','e3','e4']):
        return [ df[mask].index for mask in masks ]

    #eta selection
    eta_min = 0.0
    eta_max = 2.0
    if 'e1' in region_string:
        eta_max = eta_bins[1]
    if 'e2' in region_string:
        eta_min = eta_bins[1]
        eta_max = eta_bins[2]
    if 'e3' in region_string:
        eta_min = eta_bins[2]
        eta_max = eta_bins[3]
    if 'e4' in region_string:
        eta_min = eta_bins[3]
        eta_max = eta_bins[4]
    for i in range(njet):
        jet_i = i+1
        masks[i] &= df['jet_eta_'+str(jet_i)].apply(np.abs).between(eta_min,eta_max)
    return [ df[mask].index for mask in masks ]
