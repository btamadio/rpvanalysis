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
        self.n_systs=8
        self.web_path = ''
        self.date = ''
        self.job_name = ''
        self.canvas = None
        self.mc_label = ''
        self.lumi_label = '36.5'
        self.mc_lumi = 36.45
        self.MJ_cut = 0.8
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
        #Create directory for saving plots
        self.plot_path = self.web_path + '/'+self.date + '_' + self.job_name + '/'
        print('Creating output directory: %s ' % self.plot_path )
        os.system(' mkdir -p %s'%self.plot_path)
        os.system(' chmod a+rx %s'%self.plot_path)

    def read_bkg_from_csv(self,file_name,is_mc=True):
        print('reading from file % s'%file_name)
        self.bkg_df = pd.read_csv(file_name,delimiter=' ',na_values=[-999])#,nrows=100000)
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
        #Select signal events by DSID and scale by mult
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

    def use_lead_sublead_templates(self):
        self.df['jet_bmatched_1'] = 1
        self.df['jet_bmatched_2'] = 0
        self.df['jet_bmatched_3'] = 0
        self.df['jet_bmatched_4'] = 0

    def drop_2jet_events(self):
        index_to_drop = self.df[self.df['njet']==2].index
        self.df = self.df.drop(index_to_drop)
        for i in range(len(self.dressed_mass_nom)):
            self.dressed_mass_nom[i] = self.dressed_mass_nom[i].drop(index_to_drop)

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
        #Uncertainty of a response = RMS
        dressed_mean = response[0]
        kin_mean = response[1]
        diff = (kin_mean - dressed_mean)/dressed_mean
        result=np.sqrt(np.mean(np.square(diff)))
        return result
            
    def compute_uncertainties(self):
        #Loop over UDRs and calculate uncertainties
        self.jet_uncert = []
        for bmatch in ['bU','bM']:
            for eta_bin in range(0,4):
                region_str = '4js1VRb9'+bmatch
                self.jet_uncert.append( self.get_uncert( self.get_response( region_str )))

    def compute_dressed_masses(self,n_toys):
        self.n_toys=n_toys
        print('Generating dressed masses')
        self.dressed_mass_nom = []
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
            self.dressed_mass_nom.append(pd.DataFrame(result,index=self.df.index,columns=column_list))

    def compute_shifted_masses(self):
        print('Generated shifted jet masses')
        self.dressed_mass_shift = []
        column_list = ['jet_dressed_m_'+str(j) for j in range(self.n_toys)]
        for jet_i in range(1,5):
            col_jet_pt = self.df['jet_pt_'+str(jet_i)].values
            col_jet_temp_bin = self.df['jet_temp_bin_'+str(jet_i)].values
            result = self.dressed_mass_nom[jet_i-1].as_matrix()
            print(len(col_jet_temp_bin),result.shape)
            result_shift = jitfunctions.apply_shift_mass(result,col_jet_temp_bin,self.jet_uncert)
            self.dressed_mass_shift.append(pd.DataFrame(result_shift,index=self.df.index,columns=column_list))

    def get_response(self,region_str):
        print ('Getting response for region',region_str)
        indices = helpers.get_region_index(self.df,region_str,self.eta_bins)
        if len(indices) == 0:
            return
        jet_pt = self.df.ix[indices[0],'jet_pt_1'].values
        jet_eta = self.df.ix[indices[0],'jet_eta_1'].values
        jet_m = self.df.ix[indices[0],'jet_m_1'].values
        jet_weight = self.df.ix[indices[0],'weight'].values
        jet_dressed_m = self.dressed_mass_nom[0].ix[indices[0]].as_matrix()

        #all jets
        if not any( substr in region_str for substr in ['l1','l2'] ):
            for i in range(1,len(indices)):
                jet_i = i+1
                jet_pt = np.append( jet_pt,self.df.ix[indices[i],'jet_pt_'+str(jet_i)].values )
                jet_eta = np.append( jet_eta,self.df.ix[indices[i],'jet_eta_'+str(jet_i)].values )
                jet_m = np.append( jet_m,self.df.ix[indices[i],'jet_m_'+str(jet_i)].values )
                jet_weight = np.append( jet_weight,self.df.ix[indices[i],'weight'].values )
                jet_dressed_m = np.append( jet_dressed_m, self.dressed_mass_nom[i].ix[indices[i]].as_matrix(),axis=0)
            return jitfunctions.apply_get_mass_response(jet_pt,jet_eta,jet_m,jet_weight,jet_dressed_m,self.pt_bins)
        else:
            #just the leading jet
            if 'l1' in region_str:
                return jitfunctions.apply_get_mass_response(jet_pt,jet_eta,jet_m,jet_weight,jet_dressed_m,self.pt_bins)
            #just the subleading jets
            else:
                jet_pt = self.df.ix[indices[1],'jet_pt_2'].values
                jet_eta = self.df.ix[indices[1],'jet_eta_2'].values
                jet_m = self.df.ix[indices[1],'jet_m_2'].values
                jet_weight = self.df.ix[indices[1],'weight'].values
                jet_dressed_m = self.dressed_mass_nom[1].ix[indices[1]].as_matrix()                
                for i in range(2,len(indices)):
                    jet_i = i+1
                    jet_pt = np.append( jet_pt,self.df.ix[indices[i],'jet_pt_'+str(jet_i)].values )
                    jet_eta = np.append( jet_eta,self.df.ix[indices[i],'jet_eta_'+str(jet_i)].values )
                    jet_m = np.append( jet_m,self.df.ix[indices[i],'jet_m_'+str(jet_i)].values )
                    jet_weight = np.append( jet_weight,self.df.ix[indices[i],'weight'].values )
                    jet_dressed_m = np.append( jet_dressed_m, self.dressed_mass_nom[i].ix[indices[i]].as_matrix(),axis=0)
            return jitfunctions.apply_get_mass_response(jet_pt,jet_eta,jet_m,jet_weight,jet_dressed_m,self.pt_bins)

    def plot_MJ(self,region_str,blinded=False):
        rand_str = plotters.get_random_string()
        can_name = 'can'
        if self.canvas is None:
            self.canvas = ROOT.TCanvas(can_name,can_name,800,800)
        full_path = self.plot_path + region_str
        os.system('mkdir -p %s'%full_path)#%s' % (self.plot_path,region_str))
        os.system('chmod a+rx %s' % (full_path))
        index = helpers.get_region_index(self.df,region_str,self.eta_bins)[0]
        kin_MJ = self.df.ix[index].MJ.values
        dressed_MJ_nom = self.dressed_MJ_nom.ix[index].as_matrix()
        dressed_MJ_systs = np.array([ self.dressed_MJ_syst[i].ix[index].as_matrix() for i in range(self.n_systs) ])
        weights = self.df.ix[index].weight.values
        MJ_hists = jitfunctions.apply_get_MJ_hists(kin_MJ,dressed_MJ_nom,dressed_MJ_systs,weights,self.MJ_bins)
        scale_factor = self.get_scale_factor(index)
        sr_yields = jitfunctions.apply_get_SR_yields(kin_MJ,dressed_MJ_nom,dressed_MJ_systs,weights,self.MJ_cut)
        return plotters.plot_MJ(MJ_hists,scale_factor,sr_yields,self.plot_path,self.canvas,region_str,self.MJ_bins,self.lumi_label,self.mc_label,blinded)
        
    def plot_response(self,region_str):
        rand_str = plotters.get_random_string()
        can_name = 'can'
        if self.canvas is None:
            self.canvas = ROOT.TCanvas(can_name,can_name,800,800)
        full_path = self.plot_path + region_str
        print('Creating directory %s'%full_path)
        os.system('mkdir -p %s'%full_path)#%s' % (self.plot_path,region_str))
        os.system('chmod a+rx %s' % (full_path))
        response = self.get_response(region_str)
        print('plotting response for region',region_str)
        return plotters.plot_response(response,self.plot_path,self.canvas,region_str,self.pt_bins,self.eta_bins,self.lumi_label,self.mc_label)

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
        print('Computing systematically shifted MJ distributions')
        self.dressed_MJ_nom = self.dressed_mass_nom[0]+self.dressed_mass_nom[1]+self.dressed_mass_nom[2]+self.dressed_mass_nom[3].fillna(0)
        column_list = ['MJ_'+str(j) for j in range(self.n_toys)]
        self.dressed_MJ_nom.columns = column_list
        dm_nom_list = np.array([self.dressed_mass_nom[i].fillna(0).as_matrix() for i in range(4)])
        dm_shift_list = np.array([self.dressed_mass_shift[i].fillna(0).as_matrix() for i in range(4)])
        temp_list = np.array([self.df['jet_temp_bin_'+str(i+1)].values for i in range(4)])
        result = jitfunctions.apply_get_shifted_MJ(dm_nom_list,dm_shift_list,temp_list)
        self.dressed_MJ_syst = []
        for i in range(result.shape[0]):
            self.dressed_MJ_syst.append(pd.DataFrame(result[i],index=self.df.index,columns=column_list))

    def get_scale_factor(self,index):
        kin_MJ = self.df.ix[index].MJ.values
        dressed_MJ = self.dressed_MJ_nom.ix[index].as_matrix()        
        weights = self.df.ix[index].weight.values
        return jitfunctions.apply_get_scale_factor(kin_MJ,dressed_MJ,weights,self.norm_region[0],self.norm_region[1])
