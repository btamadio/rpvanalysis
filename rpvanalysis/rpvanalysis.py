from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jitfunctions
import plotters
import helpers
import os
import ROOT

class analyzer:
    def __init__(self):
        self.n_toys = 1
        self.web_path = '/project/projectdirs/atlas/www/multijet/RPV/btamadio/bkgEstimation/'
        self.date = '04_13'
        self.job_name = 'signal_subtraction_403566'
        self.plot_path = self.web_path + self.date + '_' + self.job_name + '/'
        print(' Output path for plots: %s ' % self.plot_path )
        os.system(' mkdir -p %s'%self.plot_path)
        os.system(' chmod a+rx %s'%self.plot_path)
        self.mc_label = ''
        self.lumi_label = '36.5'
        self.mc_lumi = 36.45
        self.n_template_bins = 50
        self.template_min = -7
        self.template_max = 0
        self.templates = {}
        self.templates_array = np.zeros((120,50))
        self.templates_neff_array = np.zeros((120,50))
        self.pt_bins = np.array([0.2,0.221,0.244,0.270,0.293,0.329,0.364,0.402,0.445,0.492,0.544,0.6,0.644,0.733,0.811,0.896])
        self.eta_bins = np.array([0.0,0.5,1.0,1.5,2.0])
        print('Setting up analysis')

    def read_bkg_from_csv(self,file_name,is_mc=True):
        print('reading from file % s'%file_name)
        self.bkg_df = pd.read_csv(file_name,delimiter=' ',na_values=[-999])
        if is_mc:
            self.bkg_df['weight'] *= self.mc_lumi
        self.df = self.bkg_df

    def read_bkg_from_root(self,file_name,is_mc=True):
        self.bkg_df = self.root_to_df(file_name)
        if is_mc:
            self.bkg_df['weight'] *= self.mc_lumi
        self.df = self.bkg_df

    def read_sig_from_root(self,file_name):
        self.sig_df = self.root_to_df(file_name)
        self.sig_df['weight'] *= self.mc_lumi
        start_ind = self.bkg_df.index.max()+1
        len_ind = len(self.sig_df)
        self.sig_df.index = np.arange(start_ind,start_ind+len_ind)
        
    def inject_sig(self,dsid,mult=1):
        if self.sig_df is None:
            print(' No signal events have been read. Run read_sig_from_root first')
            return
        sig_to_inj = self.sig_df[self.sig_df['mcChannelNumber']==dsid]
        sig_to_inj['weight'] *= mult
        self.df=self.bkg_df.append( sig_to_inj )
        print(' injecting %i raw signal events' % len(self.sig_df[self.sig_df['mcChannelNumber']==dsid]) )
        print(' cross-section scaled by a factor of %f'%mult)

    def root_to_df(self,file_name):
        print('reading from file % s'%file_name)
        f = ROOT.TFile.Open(file_name)
        t = f.Get('miniTree')
        nEntries = t.GetEntries()
        data = []
        print('reading %i rows'%nEntries)
        for entry in range(nEntries):
            if entry%10000 == 0:
                print(' reading entry %i'%entry)
            row = {}
            t.GetEntry(entry)
            row['mcChannelNumber'] = t.mcChannelNumber
            row['event_number'] = t.eventNumber
            row['njet'] = t.njet
            row['njet_soft'] = t.njet_soft
            row['nbjet'] = t.nbjet_Fix70
            row['MJ'] = t.MJ
            row['dEta'] = t.dEta
            row['weight'] = t.weight
            row['bSF'] = t.bSF_70
            for i in range(3):
                row['jet_pt_'+str(i+1)] = t.jet_pt.at(i)
                row['jet_eta_'+str(i+1)] = t.jet_eta.at(i)
                row['jet_phi_'+str(i+1)] = t.jet_phi.at(i)
                row['jet_m_'+str(i+1)] = t.jet_m.at(i)
                row['jet_bmatched_'+str(i+1)] = t.jet_bmatched_Fix70.at(i)
            if t.njet >= 4:
                row['jet_pt_4'] = t.jet_pt.at(3)
                row['jet_eta_4'] = t.jet_eta.at(3)
                row['jet_phi_4'] = t.jet_phi.at(3)
                row['jet_m_4'] = t.jet_m.at(3)
                row['jet_bmatched_4'] = t.jet_bmatched_Fix70.at(3)            
            else:
                row['jet_pt_4'] = np.nan
                row['jet_eta_4'] = np.nan
                row['jet_phi_4'] = np.nan
                row['jet_m_4'] = np.nan
                row['jet_bmatched_4'] = np.nan
            data.append(row)
        return pd.DataFrame(data)

    def compute_temp_bins(self):
        #Get template bin indices for all jets and add to dataframe
        print('Determining template bins for all jets')
        for i in range(1,5):
            result = jitfunctions.apply_get_temp_bin(self.df['jet_pt_'+str(i)].values,
                                                     self.df['jet_eta_'+str(i)].values,
                                                     self.df['jet_bmatched_'+str(i)].values,
                                                     self.pt_bins,
                                                     self.eta_bins)
            self.df['jet_temp_bin_'+str(i)]=pd.Series(result,index=self.df.index,name='jet_temp_bin_'+str(i))

    def create_templates(self):
        print('Creating templates from control region')
        for i in range(1,5):
            mask = self.df['njet']==3
            mask &= (self.df['jet_bmatched_'+str(i)]==0) | (self.df['jet_bmatched_'+str(i)]==1)&(self.df['dEta']>1.4)
            self.df['jet_is_CR_'+str(i)] = mask.astype(int)
        df_temps = [None,None,None]

        #create 3 reindexed data frames, each containingly only CR jets
        for i in range(1,4):
            jet_var_list = ['jet_temp_bin_'+str(i),'jet_m_'+str(i),'jet_pt_'+str(i),'jet_is_CR_'+str(i),'weight']
            df_temps[i-1] = self.df[self.df['jet_is_CR_'+str(i)]==1][jet_var_list].set_index(keys=['jet_temp_bin_'+str(i)])
        for bmatch in [0,1]:
            for pt_bin in range(len(self.pt_bins)-1):
                for eta_bin in range(len(self.eta_bins)-1):
                    key = 60*bmatch+4*pt_bin+eta_bin
                    self.templates[key] = helpers.template(self.n_template_bins,df_temps,key,self.template_min,self.template_max)
                    self.templates_array[key]=self.templates[key].probs
                    self.templates_neff_array[key]=self.templates[key].n_eff

    def compute_dressed_masses(self):
        print('Generating dressed masses')
        self.dressed_mass_df = []
        for jet_i in range(1,5):
            result = jitfunctions.apply_get_dressed_mass(self.df['jet_pt_'+str(jet_i)].values,
                                                         self.df['jet_temp_bin_'+str(jet_i)].values,
                                                         self.templates_array,
                                                         self.templates_neff_array,
                                                         self.templates[0].bin_centers,
                                                         self.templates[0].bin_edges,
                                                         self.n_toys)
            self.dressed_mass_df.append(pd.DataFrame(result,index=self.df.index,columns=['jet_dressed_m_'+str(j) for j in range(self.n_toys)]))

    def get_response(self,region_string,eta_bin=-1):
        print('Generating response plots for region %s'%region_string)
        eta_min = 0
        eta_max = 2
        if eta_bin >= 0:
            eta_min = self.eta_bins[eta_bin]
            eta_max = self.eta_bins[eta_bin+1]
        indices = helpers.get_region_index(self.df,region_string,eta_min,eta_max)
        if len(indices) == 0:
            return

        os.system('mkdir -p %s/%s' % (self.plot_path,region_string))
        os.system('chmod a+rx %s/%s' % (self.plot_path,region_string))
        
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
        return jitfunctions.apply_get_mass_response(jet_pt,jet_eta,jet_m,jet_weight,jet_dressed_m,self.pt_bins)
        #plotters.plot_response(response,self.plot_path,region_string,self.pt_bins,self.lumi_label,self.mc_label,eta_bin)
    def plot_response(self,response,region_str,eta_bin):
        plotters.plot_response(response,self.plot_path,region_str,self.pt_bins,self.lumi_label,self.mc_label,eta_bin)
        os.system('chmod a+rx %s%s/*' % (self.plot_path,region_string))
    def verify_templates(self):
        print('Verifying templates')
        for key in sorted(self.templates.keys()):
            t = self.templates[key].probs
            for i,prob in enumerate(t):
                if prob < 0:
                    print (' template number %i, bin %i, prob = %f ' % (key,i,prob) )
            if abs(t.sum() -1) > 1e-6:
                print(' template number %i, probabilities sum to %f'%abs(t.sum()-1))
