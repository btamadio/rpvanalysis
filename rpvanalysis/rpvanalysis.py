from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rpvanalysis import jitfunctions
from rpvanalysis import plotters
from rpvanalysis import helpers
import os
import ROOT

class analyzer:
    def __init__(self):
        self.web_path = ''
        self.date = ''
        self.job_name = ''
        self.mc_label = ''
        self.lumi_label = '36.5'
        self.mc_lumi = 36.45
        self.mj_cut = 0.8
        self.n_template_bins = 50
        self.template_min = -7
        self.template_max = 0
        self.templates = {}
        self.norm_region = (0.2,0.4)
        self.jet_uncert = np.zeros(8)
        self.templates_array = np.zeros((120,50))
        self.templates_neff_array = np.zeros((120,50))
        self.pt_bins = np.array([0.2,0.221,0.244,0.270,0.293,0.329,0.364,0.402,0.445,0.492,0.544,0.6,0.644,0.733,0.811,0.896])
        self.eta_bins = np.array([0.0,0.5,1.0,1.5,2.0])
        self.MJ_bins =np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.40,0.45,0.5,0.6,0.8,1.0,2.0])
        print('Setting up analysis')

    def make_plot_dir(self):
        self.plot_path = self.web_path + '/'+self.date + '_' + self.job_name + '/'
        print('Creating output directory: %s ' % self.plot_path )
        os.system(' mkdir -p %s'%self.plot_path)
        os.system(' chmod a+rx %s'%self.plot_path)

    def read_bkg_from_csv(self,file_name,is_mc=True):
        print('reading from file % s'%file_name)
        self.bkg_df = pd.read_csv(file_name,delimiter=' ',na_values=[-999])#,nrows=1000000)
        if is_mc:
            self.bkg_df['weight'] *= self.mc_lumi
        self.df = self.bkg_df
        print(' Read %i raw events from file' % len(self.bkg_df) )
    
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
    def get_uncert(self,response):
        dressed_mean = response[0]
        kin_mean = response[1]
        diff = (kin_mean - dressed_mean)/dressed_mean
        result=np.sqrt(np.mean(np.square(diff)))
        print(result)
        return result
            
    def compute_uncertainties(self):
        self.jet_uncert = []
        for bmatch in ['bU','bM']:
            for eta_bin in range(0,4):
                region_str = '4js1VRb9'+bmatch
                print(region_str)
                self.jet_uncert.append( self.get_uncert( self.get_response( region_str,eta_bin )))
                
    def compute_dressed_masses(self,n_toys=100):
        print('Generating dressed masses')
        self.dressed_mass_nom = []
        self.dressed_mass_shift = []
        column_list = ['jet_dressed_m_'+str(j) for j in range(n_toys)]
        for jet_i in range(1,5):
            col_jet_pt = self.df['jet_pt_'+str(jet_i)].values
            col_jet_temp_bin = self.df['jet_temp_bin_'+str(jet_i)].values
            result = jitfunctions.apply_get_dressed_mass(col_jet_pt,
                                                         col_jet_temp_bin,
                                                         self.templates_array,
                                                         self.templates_neff_array,
                                                         self.templates[0].bin_centers,
                                                         self.templates[0].bin_edges,
                                                         n_toys)
            result_shift = jitfunctions.apply_shift_mass(result,col_jet_temp_bin,self.jet_uncert)
            self.dressed_mass_nom.append(pd.DataFrame(result,index=self.df.index,columns=column_list))
            self.dressed_mass_shift.append(pd.DataFrame(result_shift,index=self.df.index,columns=column_list))

    def get_response(self,region_str,eta_bin=-1):
        eta_min = 0
        eta_max = 2
        if eta_bin >= 0:
            eta_min = self.eta_bins[eta_bin]
            eta_max = self.eta_bins[eta_bin+1]
        indices = helpers.get_region_index(self.df,region_str,eta_min,eta_max)
        if len(indices) == 0:
            return
        jet_pt = self.df.ix[indices[0],'jet_pt_1'].values
        jet_eta = self.df.ix[indices[0],'jet_eta_1'].values
        jet_m = self.df.ix[indices[0],'jet_m_1'].values
        jet_weight = self.df.ix[indices[0],'weight'].values
        jet_dressed_m = self.dressed_mass_nom[0].ix[indices[0]].as_matrix()

        for i in range(1,len(indices)):
            jet_i = i+1
            jet_pt = np.append( jet_pt,self.df.ix[indices[i],'jet_pt_'+str(jet_i)].values )
            jet_eta = np.append( jet_eta,self.df.ix[indices[i],'jet_eta_'+str(jet_i)].values )
            jet_m = np.append( jet_m,self.df.ix[indices[i],'jet_m_'+str(jet_i)].values )
            jet_weight = np.append( jet_weight,self.df.ix[indices[i],'weight'].values )
            jet_dressed_m = np.append( jet_dressed_m, self.dressed_mass_nom[i].ix[indices[i]].as_matrix(),axis=0)
        return jitfunctions.apply_get_mass_response(jet_pt,jet_eta,jet_m,jet_weight,jet_dressed_m,self.pt_bins)

    def plot_MJ(self,region_str):
        rand_str = plotters.get_random_string()
        can_name = 'c_'+rand_str
        self.canvas = ROOT.TCanvas(can_name,can_name,800,800)
        full_path = self.plot_path + region_str
        os.system('mkdir -p %s'%full_path)#%s' % (self.plot_path,region_str))
        os.system('chmod a+rx %s' % (full_path))
        index = helpers.get_region_index(self.df,region_str)[0]
        kin_MJ = self.df.ix[index].MJ.values
        dressed_MJ_nom = self.dressed_MJ_nom.ix[index].as_matrix()
        dressed_MJ_systs = [ self.dressed_MJ_syst[i].ix[index].as_matrix() for i in range(8) ]
        weights = self.df.ix[index].weight.values
        
        MJ_hists = jitfunctions.apply_get_MJ_hists(kin_MJ,dressed_MJ_nom,dressed_MJ_systs,weights,self.MJ_bins)
#        plotters.plot_MJ(MJ_hists,self.plot_path,self.canvas,region_str,self.MJ_bins,self.lumi_label,self.mc_label)

    def plot_response(self,region_str,eta_bin=-1):
        rand_str = plotters.get_random_string()
        can_name = 'c_'+rand_str
        self.canvas = ROOT.TCanvas(can_name,can_name,800,800)
        full_path = self.plot_path + region_str
        print('Creating directory %s'%full_path)
        os.system('mkdir -p %s'%full_path)#%s' % (self.plot_path,region_str))
        os.system('chmod a+rx %s' % (full_path))
        response = self.get_response(region_str,eta_bin)
        plotters.plot_response(response,self.plot_path,self.canvas,region_str,self.pt_bins,self.lumi_label,self.mc_label,eta_bin)
    def verify_templates(self):
        print('Verifying templates')
        for key in sorted(self.templates.keys()):
            t = self.templates[key].probs
            for i,prob in enumerate(t):
                if prob < 0:
                    print (' template number %i, bin %i, prob = %f ' % (key,i,prob) )
            if abs(t.sum() -1) > 1e-6:
                print(' template number %i, probabilities sum to %f'%abs(t.sum()-1))
    
    def compute_dressed_MJ(self):
        self.dressed_MJ_nom = self.dressed_mass_nom[0]+self.dressed_mass_nom[1]+self.dressed_mass_nom[2]+self.dressed_mass_nom[3].fillna(0)
        dm_nom_list = np.array([self.dressed_mass_nom[i].fillna(0).as_matrix() for i in range(4)])
        dm_shift_list = np.array([self.dressed_mass_shift[i].fillna(0).as_matrix() for i in range(4)])
        temp_list = np.array([self.df['jet_temp_bin_'+str(i+1)].values for i in range(4)])
        self.dressed_MJ_syst = jitfunctions.apply_get_shifted_MJ(dm_nom_list,dm_shift_list,temp_list)
        #TODO: create 8 shifted MJ dataframes instead of just 1
        # For each uncertainty region, only shift the jets that belong to that region
#        

        
    
    def get_scale_factor(self,region_str):
        index = helpers.get_region_index(self.df,region_str)[0]
        kin_MJ = self.df.ix[index].MJ.values
        dressed_MJ = self.dressed_MJ_nom.ix[index].as_matrix()        
        weights = self.df.ix[index].weight.values
        return jitfunctions.apply_get_scale_factor(kin_MJ,dressed_MJ,weights,self.norm_region[0],self.norm_region[1])
