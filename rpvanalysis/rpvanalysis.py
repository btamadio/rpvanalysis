from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jitfunctions

class analyzer:
    def __init__(self,file_name):
        self.templates = {}
        self.probs_array = np.zeros((120,50))
        self.bin_edges_array = np.zeros((120,51))
        self.bin_centers_array = np.zeros((120,50))

        print('Setting up analysis')
        print('reading file % s'%file_name)
        self.df = pd.read_csv(file_name,delimiter=' ',index_col='event_number',na_values=[-999])#,nrows=10000)
        self.compute_temp_bins()
        self.create_templates()
        self.compute_dressed_masses()
        #self.df.info()
        
    def compute_temp_bins(self):
        #Get template bin indices for all jets and add to dataframe
        print('Determining template bins for all jets')
        for i in range(1,5):
            result = jitfunctions.apply_get_temp_bin(self.df['jet_pt_'+str(i)].values,
                                                     self.df['jet_eta_'+str(i)].values,
                                                     self.df['jet_bmatched_'+str(i)].values)
            self.df['jet_temp_bin_'+str(i)]=pd.Series(result,index=self.df.index,name='jet_temp_bin_'+str(i))

    def get_template_hist(self,df_temps,temp_bin):
        #TODO: optimize with JIT. This is really messy
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
        if hist.sum() > 0:
            hist = hist/hist.sum()
        return (hist,bin_edges)

    def create_templates(self):
        #TODO: optimize with JIT

        #Create template histograms from CR
        print('Creating templates from control region')
        #mark the control region jets
        for i in range(1,5):
            mask = self.df['njet']==3
            mask &= (self.df['jet_bmatched_'+str(i)]==0) | (self.df['jet_bmatched_'+str(i)]==1)&(self.df['dEta']>1.4)
            self.df['jet_is_CR_'+str(i)] = mask.astype(int)
        df_temps = [None,None,None]

        #create 3 reindexed data frames, each containingly only CR jets
        for i in range(1,4):
            jet_var_list = ['jet_temp_bin_'+str(i),'jet_m_'+str(i),'jet_pt_'+str(i),'jet_is_CR_'+str(i),'weight']
            df_temps[i-1] = self.df[self.df['jet_is_CR_'+str(i)]==1][jet_var_list].set_index(keys=['jet_temp_bin_'+str(i)])

        pt_bins = np.array([0.2,0.221,0.244,0.270,0.293,0.329,0.364,0.402,0.445,0.492,0.544,0.6,0.644,0.733,0.811,13])
        eta_bins = np.array([0.0,0.5,1.0,1.5,2.0])

        for pt_bin in range(len(pt_bins)-1):
            for eta_bin in range(len(eta_bins)-1):
                for bmatch in [0,1]:
                    key = 60*bmatch+4*pt_bin+eta_bin
                    self.templates[key] = self.get_template_hist(df_temps,key)
                    self.probs_array[key] = self.templates[key][0]
                    self.bin_edges_array[key] = self.templates[key][1]
                    for i in range(len(self.bin_edges_array[key])-1):
                        self.bin_centers_array[key][i] = (self.bin_edges_array[key][i+1]+self.bin_edges_array[key][i])/2

    def plot_template(self,temp_bin):
        h = self.templates[temp_bin]
        plt.figure()
        plt.bar(h[1][:-1],h[0],width=h[1][1]-h[1][0])
        plt.show()

    def compute_dressed_masses(self,n_toys=100):
        print('Generating dressed masses')
        self.dressed_mass_df = {}
        for jet_i in range(1,5):
            result = jitfunctions.apply_get_dressed_mass(self.df['jet_pt_'+str(jet_i)].values,
                                                         self.df['jet_temp_bin_'+str(jet_i)].values,
                                                         self.probs_array,
                                                         self.bin_centers_array,
                                                         n_toys)
            self.dressed_mass_df[jet_i]=pd.DataFrame(result,index=self.df.index,columns=['jet_dressed_m_'+str(j) for j in range(n_toys)])
    
