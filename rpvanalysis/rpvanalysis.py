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
        print('Setting up analysis')
        self.template_type = 0
        self.web_path = ''
        self.date = ''
        self.job_name = ''
        self.canvas = None
        self.dressed_mass_nom = None
        self.mc_label = ''
        self.lumi_label = '36.5'
        self.mc_lumi = 36.45
        self.MJ_cut = 0.8
        self.n_template_bins = 50
        self.template_min = -7
        self.template_max = 0
        self.templates = {}
        self.norm_region = (0.2,0.4)
        self.pt_bins = np.array([0.2,0.221,0.244,0.270,0.293,0.329,0.364,0.402,0.445,0.492,0.544,0.6,0.644,0.733,0.811,0.896])
        self.eta_bins = np.array([0.0,0.5,1.0,1.5,2.0])
        self.MJ_bins =np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.40,0.45,0.5,0.6,0.8,1.0,2.0])

    def make_plot_dir(self):
        #Create directory for saving plots
        self.plot_path = self.web_path.rstrip('/') + '/'+self.date + '_' + self.job_name + '/'
        print('Creating output directory: %s ' % self.plot_path )
        if not os.path.exists(self.plot_path):
            os.mkdir(self.plot_path)
        os.system('chmod a+rx %s'%self.plot_path)

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

    def drop_2jet_events(self):
        print('Dropping 2-jet events')
        mask = self.df['njet']>2
        self.df = self.df[mask]
        self.df.index = np.arange(len(self.df))
        print(len(self.df))
        if self.dressed_mass_nom is not None:
            temp_dm = np.zeros((4,len(self.df),self.n_toys))
            for i in range(4):
                temp_dm[i] = self.dressed_mass_nom[i][mask]
            self.dressed_mass_nom = temp_dm
            print(self.dressed_mass_nom.shape,' events remaining')

    def compute_uncert_bins(self):
        print('Determining uncertainty bins for all jets')
        for i in range(1,5):
            result = jitfunctions.apply_get_uncert_bin(self.df['jet_pt_'+str(i)].values,
                                                       self.df['jet_eta_'+str(i)].values,
                                                       self.df['jet_bmatched_'+str(i)].values,
                                                       self.eta_bins,
                                                       self.template_type)
            self.df['jet_uncert_bin_'+str(i)]=pd.Series(result,index=self.df.index,name='jet_uncert_bin_'+str(i))
            
    def compute_temp_bins(self):
        #Get template bin indices for all jets and add to dataframe
        print('Determining template bins for all jets')
        if self.template_type == 0:
            #b-match vs. non-b-match jets
            for i in range(1,5):
                result = jitfunctions.apply_get_temp_bin(self.df['jet_pt_'+str(i)].values,
                                                         self.df['jet_eta_'+str(i)].values,
                                                         self.df['jet_bmatched_'+str(i)].values,
                                                         self.pt_bins,
                                                         self.eta_bins)
                self.df['jet_temp_bin_'+str(i)]=pd.Series(result,index=self.df.index,name='jet_temp_bin_'+str(i))
        elif self.template_type == 1:
            #merge b-match and non-b-match jets
            for i in range(1,5):
                result = jitfunctions.apply_get_temp_bin(self.df['jet_pt_'+str(i)].values,
                                                         self.df['jet_eta_'+str(i)].values,
                                                         np.zeros( len(self.df),dtype=np.int32 ),
                                                         self.pt_bins,
                                                         self.eta_bins)
                self.df['jet_temp_bin_'+str(i)]=pd.Series(result,index=self.df.index,name='jet_temp_bin_'+str(i))            

        elif self.template_type == 2:
            #leading vs. non-leading jets
            for i in range(1,5):
                result = jitfunctions.apply_get_temp_bin(self.df['jet_pt_'+str(i)].values,
                                                         self.df['jet_eta_'+str(i)].values,
                                                         np.repeat(int(i==1),len(self.df)),
                                                         self.pt_bins,
                                                         self.eta_bins)
                self.df['jet_temp_bin_'+str(i)]=pd.Series(result,index=self.df.index,name='jet_temp_bin_'+str(i))            

    def create_templates(self):
        print('Creating templates from control region')
        #label jets that belong to control region
        for i in range(1,5):
            mask = self.df['njet']==3
            mask &= (self.df['jet_bmatched_'+str(i)]==0) | (self.df['jet_bmatched_'+str(i)]==1)&(self.df['dEta']>1.4)
            self.df['jet_is_CR_'+str(i)] = mask.astype(int)

        print(' making reindexed data frames for CR jets')
        #create 3 new dataframes containing only control region jets, and indexed on their template bins
        df_temps = [None,None,None]
        for i in range(1,4):
            jet_var_list = ['jet_temp_bin_'+str(i),'jet_m_'+str(i),'jet_pt_'+str(i),'jet_is_CR_'+str(i),'weight']
            df_temps[i-1] = self.df[self.df['jet_is_CR_'+str(i)]==1][jet_var_list].set_index(keys=['jet_temp_bin_'+str(i)])
            
        #loop over all template bins for which CR jets exist and create the templates
        key_list = sorted( np.unique( self.df[['jet_temp_bin_1','jet_temp_bin_2','jet_temp_bin_3']].values ) )
        
        self.templates_array = np.zeros((len(key_list),50))
        self.templates_neff_array = np.zeros((len(key_list),50))

        for key in key_list:
            if key == -1:
                continue
            key = np.asscalar(key)
            self.templates[key] = helpers.template(self.n_template_bins,df_temps,key,self.template_min,self.template_max)
            self.templates_array[key]=self.templates[key].probs
            self.templates_neff_array[key]=self.templates[key].n_eff

    def get_uncert(self,region_str):
        response = self.get_response(region_str)
        #Uncertainty of a response = RMS
        #for new UDRs: don't count bins that have zero entries
        dressed_mean = response[0]
        kin_mean = response[1]
        result = 0.0
        count = 0
        for i in range(dressed_mean.shape[0]):
            if 'LJG400' in region_str and self.pt_bins[i] < 0.4:
                continue
            if 'LJL400' in region_str and self.pt_bins[i] > 0.4:
                continue
            if dressed_mean[i] > 1e-6 and kin_mean[i] > 1e-6:
                result += np.square ( (kin_mean[i] - dressed_mean[i])/dressed_mean[i] )
                count+=1
        result /= count
        result = np.sqrt(result)
        return result
            
    def compute_uncertainties(self):
        #Loop over UDRs and calculate uncertainties
        if self.template_type == 0:
            self.UDRs = ['4js1VRb9bUe1','4js1VRb9bUe2','4js1VRb9bUe3','4js1VRb9bUe4',
                         '4js1VRb9bMe1','4js1VRb9bMe2','4js1VRb9bMe3','4js1VRb9bMe4']
        elif self.template_type == 1 or self.template_type==2:
            self.UDRs = ['4js1LJL400e1','4js1LJL400e2','4js1LJL400e3','4js1LJL400e4',
                         '2jLJG400e1','2jLJG400e2','2jLJG400e3','2jLJG400e4']
        self.jet_uncert = []
        for region_str in self.UDRs:
            self.jet_uncert.append(self.get_uncert(region_str))
            print (region_str,'uncertainty = ',self.jet_uncert[-1])

    def compute_dressed_masses(self,n_toys):
        self.n_toys=n_toys
        print('Generating dressed masses')
        self.dressed_mass_nom = np.zeros( (4,len(self.df),n_toys) )
        for i in range(4):
            jet_i = i+1
            col_jet_pt = self.df['jet_pt_'+str(jet_i)].values
            col_jet_temp_bin = self.df['jet_temp_bin_'+str(jet_i)].values
            result = jitfunctions.apply_get_dressed_mass(col_jet_pt,
                                                         col_jet_temp_bin,
                                                         self.templates_array,
                                                         self.templates_neff_array,
                                                         self.templates[0].bin_centers,
                                                         self.templates[0].bin_edges,
                                                         n_toys)
            self.dressed_mass_nom[i] = np.nan_to_num(result)

    def compute_shifted_masses(self):
        print('Generating shifted jet masses')
        self.dressed_mass_shift = np.zeros( self.dressed_mass_nom.shape )
        for i in range(4):
            jet_i = i+1
            col_jet_pt = self.df['jet_pt_'+str(jet_i)].values
            col_jet_uncert_bin = self.df['jet_uncert_bin_'+str(jet_i)].values
            self.dressed_mass_shift[i] = jitfunctions.apply_shift_mass(self.dressed_mass_nom[i],col_jet_uncert_bin,self.jet_uncert)

    def get_response(self,region_str):
        #TODO: clean up this shitty code
        print ('Getting response for region',region_str)
        indices = helpers.get_region_index(self.df,region_str,self.eta_bins)
        if len(indices) == 0:
            return
        jet_pt = self.df.ix[indices[0],'jet_pt_1'].values
        jet_eta = self.df.ix[indices[0],'jet_eta_1'].values
        jet_m = self.df.ix[indices[0],'jet_m_1'].values
        jet_weight = self.df.ix[indices[0],'weight'].values
        jet_dressed_m = self.dressed_mass_nom[0][indices[0]]
        #all jets
        if not any( substr in region_str for substr in ['l1','l2'] ):
            for i in range(1,len(indices)):
                jet_i = i+1
                jet_pt = np.append( jet_pt,self.df.ix[indices[i],'jet_pt_'+str(jet_i)].values )
                jet_eta = np.append( jet_eta,self.df.ix[indices[i],'jet_eta_'+str(jet_i)].values )
                jet_m = np.append( jet_m,self.df.ix[indices[i],'jet_m_'+str(jet_i)].values )
                jet_weight = np.append( jet_weight,self.df.ix[indices[i],'weight'].values )
                jet_dressed_m = np.append( jet_dressed_m, self.dressed_mass_nom[i][indices[i]],axis=0)
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
                jet_dressed_m = self.dressed_mass_nom[1][indices[1]]
                for i in range(2,len(indices)):
                    jet_i = i+1
                    jet_pt = np.append( jet_pt,self.df.ix[indices[i],'jet_pt_'+str(jet_i)].values )
                    jet_eta = np.append( jet_eta,self.df.ix[indices[i],'jet_eta_'+str(jet_i)].values )
                    jet_m = np.append( jet_m,self.df.ix[indices[i],'jet_m_'+str(jet_i)].values )
                    jet_weight = np.append( jet_weight,self.df.ix[indices[i],'weight'].values )
                    jet_dressed_m = np.append( jet_dressed_m, self.dressed_mass_nom[i][indices[i]],axis=0)
            return jitfunctions.apply_get_mass_response(jet_pt,jet_eta,jet_m,jet_weight,jet_dressed_m,self.pt_bins)

    def plot_MJ(self,region_str,blinded=False):
        rand_str = plotters.get_random_string()
        can_name = 'can'
        if self.canvas is None:
            self.canvas = ROOT.TCanvas(can_name,can_name,800,800)
        full_path = self.plot_path + region_str
        print('Creating directory %s'%full_path)
        if not os.path.exists(full_path):
            os.mkdir(full_path)
        index = helpers.get_region_index(self.df,region_str,self.eta_bins)[0]
        kin_MJ = self.df.ix[index].MJ.values
        dressed_MJ_systs = np.array([ self.dressed_MJ_syst[i][index] for i in range(len(self.UDRs)) ])
        weights = self.df.ix[index].weight.values
        MJ_hists = jitfunctions.apply_get_MJ_hists(kin_MJ,self.dressed_MJ_nom[index],dressed_MJ_systs,weights,self.MJ_bins)
        scale_factor = self.get_scale_factor(index)
        sr_yields = jitfunctions.apply_get_SR_yields(kin_MJ,self.dressed_MJ_nom[index],dressed_MJ_systs,weights,self.MJ_cut)
        return plotters.plot_MJ(MJ_hists,scale_factor,sr_yields,self.plot_path,self.canvas,region_str,self.MJ_bins,self.lumi_label,self.mc_label,blinded,self.MJ_cut)
        
    def plot_response(self,region_str):
        rand_str = plotters.get_random_string()
        can_name = 'can'
        if self.canvas is None:
            self.canvas = ROOT.TCanvas(can_name,can_name,800,800)
        full_path = self.plot_path + region_str
        print('Creating directory %s'%full_path)
        os.system('mkdir -p %s'%full_path)
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
        self.dressed_MJ_nom = self.dressed_mass_nom[0]+self.dressed_mass_nom[1]+self.dressed_mass_nom[2]+np.nan_to_num(self.dressed_mass_nom[3])
        jet_uncert_bins = np.array([self.df['jet_uncert_bin_'+str(i+1)].values for i in range(4)])
        self.dressed_MJ_syst = jitfunctions.apply_get_shifted_MJ(self.dressed_mass_nom,self.dressed_mass_shift,jet_uncert_bins,len(self.UDRs))

    def get_scale_factor(self,index):
        kin_MJ = self.df.ix[index].MJ.values
        weights = self.df.ix[index].weight.values
        return jitfunctions.apply_get_scale_factor(kin_MJ,self.dressed_MJ_nom[index],weights,self.norm_region[0],self.norm_region[1])
