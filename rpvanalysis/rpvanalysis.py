from __future__ import print_function
import pandas as pd
import numpy as np
import numba

@numba.jit(nopython=True)
def get_temp_bin(pt,eta,bmatch):
    #For a given pt, eta, and b-match status of a jet, return the template index
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

class analyzer:
    def __init__(self,file_name):
        self.templates = {}
        print('Setting up analysis')
        print('reading file % s'%file_name)
        headers = ['njet','MJ','dEta']
        headers+=['jet_pt_'+str(i) for i in range(1,5)]
        headers+=['jet_eta_'+str(i) for i in range(1,5)]
        headers+=['jet_m_'+str(i) for i in range(1,5)]
        headers+=['jet_phi_'+str(i) for i in range(1,5)]
        headers+=['jet_bmatched_'+str(i) for i in range(1,5)]
        headers+=['nbjet','weight']
        self.df = pd.read_csv(file_name,delimiter=' ',header=0,names=headers,nrows=100)
        self.compute_temp_bins()
        self.create_templates()
        self.df.info()
        
    def compute_temp_bins(self):
        #Get template bin indices for all jets and add to dataframe
        print('Determining template bins for all jets')
        for i in range(1,5):
            result = apply_get_temp_bin(self.df['jet_pt_'+str(i)].values,
                                        self.df['jet_eta_'+str(i)].values,
                                        self.df['jet_bmatched_'+str(i)].values)
            self.df['jet_temp_bin_'+str(i)]=pd.Series(result,index=self.df.index,name='jet_temp_bin_'+str(i))

    def get_template_hist(self,df_temps,temp_bin):
        n_bins = 50
        hist,bin_edges = np.histogram(a=[],range=(-7,0),bins=n_bins)
        hist = hist.astype(float)
        for i in range(3):
            try:
                this_df = df_temps[i].ix[temp_bin]
            except:
                continue
            temp_vals = np.log(this_df['jet_m_'+str(i+1)]/this_df['jet_pt_'+str(i+1)])
            temp_wts = this_df.weight
            temp_hist,temp_bin_edges = np.histogram(a=temp_vals,weights=temp_wts,range=(-7,0),bins=n_bins)
            hist+=temp_hist
        #normalize to 1 so it can be used as PDF
        hist = hist/hist.sum()
        return (hist,bin_edges)

    def create_templates(self):
        #Create template histograms from CR
        print('Creating templates from control region')
        for i in range(1,5):
            self.df['jet_is_CR_'+str(i)] = ((self.df['njet']==3)&( (self.df['jet_bmatched_'+str(i)]==0) | ((self.df['jet_bmatched_'+str(i)]==1)&(self.df['dEta']>1.4)))).astype(int)
        df_temps = [None,None,None]
        for i in range(1,4):
            df_temps[i-1] = self.df[self.df['jet_is_CR_'+str(i)]==1][['jet_temp_bin_'+str(i),'jet_m_'+str(i),'jet_pt_'+str(i),'jet_is_CR_'+str(i),'weight']].set_index(keys=['jet_temp_bin_'+str(i)])

        pt_bins = np.array([0.2,0.221,0.244,0.270,0.293,0.329,0.364,0.402,0.445,0.492,0.544,0.6,0.644,0.733,0.811,13])
        eta_bins = np.array([0.0,0.5,1.0,1.5,2.0])

        for pt_bin in range(len(pt_bins)-1):
            for eta_bin in range(len(eta_bins)-1):
                for bmatch in [0,1]:
                    key = 60*bmatch+4*pt_bin+eta_bin
                    self.templates[key] = self.get_template_hist(df_temps,key)
