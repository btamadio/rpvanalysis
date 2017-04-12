from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jitfunctions

class analyzer:
    def __init__(self,file_name):
        self.n_toys = 100
        self.templates = {}
        self.probs_array = np.zeros((120,50))
        self.bin_edges_array = np.zeros((120,51))
        self.bin_centers_array = np.zeros((120,50))
        self.pt_bins = np.array([0.2,0.221,0.244,0.270,0.293,0.329,0.364,0.402,0.445,0.492,0.544,0.6,0.644,0.733,0.811,13])
        self.eta_bins = np.array([0.0,0.5,1.0,1.5,2.0])

        print('Setting up analysis')
        print('reading file % s'%file_name)
        self.df = pd.read_csv(file_name,delimiter=' ',index_col='event_number',na_values=[-999],nrows=100000)
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
        #TODO: optimize with JIT and make use of get_region_index function

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

        for pt_bin in range(len(self.pt_bins)-1):
            for eta_bin in range(len(self.eta_bins)-1):
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

    def compute_dressed_masses(self):
        print('Generating dressed masses')
        self.dressed_mass_df = []
        for jet_i in range(1,5):
            result = jitfunctions.apply_get_dressed_mass(self.df['jet_pt_'+str(jet_i)].values,
                                                         self.df['jet_temp_bin_'+str(jet_i)].values,
                                                         self.probs_array,
                                                         self.bin_centers_array,
                                                         self.n_toys)
            self.dressed_mass_df.append(pd.DataFrame(result,index=self.df.index,columns=['jet_dressed_m_'+str(j) for j in range(self.n_toys)]))

    def get_region_index(self,region_string):
        #Given a region string, return a list corresponding to the index of jets for that region
        mask = None
        njet=0
        if region_string.startswith('3j'):
            mask = self.df['njet']==3
            njet=3
        elif region_string.startswith('4j'):
            mask = self.df['njet']==4
            njet=4
        elif region_string.startswith('5j'):
            mask = self.df['njet']>=5
            njet=4
        else:
            print('Error: region name %s is invalid. Must start with 3j,4j, or 5j'%region_string)
            return np.array([])

        if 's0' in region_string:
            mask &= self.df['njet_soft'] == 0
        elif 's1' in region_string:
            mask &= self.df['njet_soft'] >= 1

        if 'VR' in region_string:
            mask &= self.df['dEta'] > 1.4
        elif 'SR' in region_string:
            mask &= self.df['dEta'] < 1.4

        if 'b0' in region_string:
            mask &= self.df['nbjet'] == 0
        elif 'b1' in region_string:
            mask &= self.df['nbjet'] >= 1
    
        if (not 'bU' in region_string) and (not 'bM' in region_string):
            return [ self.df[mask].index for _ in range(njet) ]
        
        masks = [ mask for _ in range(njet)]
        if 'bU' in region_string:
            for i in range(njet):
                jet_i = i+1
                masks[i] &= self.df['jet_bmatched_'+str(jet_i)] == 0

        elif 'bM' in region_string:
            for i in range(njet):
                jet_i = i+1
                masks[i] &= self.df['jet_bmatched_'+str(jet_i)] == 1
        
        return [ self.df[mask].index for mask in masks ]

    def plot_response(self,region_string):
        print('Generating response plots for region %s'%region_string)
        indices = self.get_region_index(region_string)

        jet_pt = self.df.ix[indices[0],'jet_pt_1'].values
        jet_eta = self.df.ix[indices[0],'jet_eta_1'].values
        jet_m = self.df.ix[indices[0],'jet_m_1'].values
        jet_weight = self.df.ix[indices[0],'weight'].values
        jet_dressed_m = self.dressed_mass_df[0].ix[indices[0],'jet_dressed_m_0'].values

        for i in range(1,len(indices)):
            jet_i = i+1
            jet_pt = np.append( jet_pt,self.df.ix[indices[i],'jet_pt_'+str(jet_i)].values )
            jet_eta = np.append( jet_eta,self.df.ix[indices[i],'jet_eta_'+str(jet_i)].values )
            jet_m = np.append( jet_m,self.df.ix[indices[i],'jet_m_'+str(jet_i)].values )
            jet_weight = np.append( jet_weight,self.df.ix[indices[i],'weight'].values )
            jet_dressed_m = np.append( jet_dressed_m, self.dressed_mass_df[i].ix[indices[i],'jet_dressed_m_0'].values,axis=0)
        dressed_mean,kin_mean,err=jitfunctions.apply_get_mass_response(jet_pt,jet_eta,jet_m,jet_weight,jet_dressed_m,self.pt_bins)
        print(dressed_mean,kin_mean,err)
