from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jitfunctions
import plotters
import helpers
import os
import ROOT
import csv

class analyzer:
    def __init__(self):
        print('Setting up analysis')
        self.template_type = 0
        self.web_path = ''
        self.hist_path = ''
        self.date = ''
        self.job_name = ''
        self.canvas = None
        self.bkg_df = None
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
        if self.web_path:
            self.plot_path = self.web_path.rstrip('/') + '/'+self.date + '_' + self.job_name + '/'
            print('Creating output directory: %s ' % self.plot_path )
            if not os.path.exists(self.plot_path):
                os.mkdir(self.plot_path)
            os.system('chmod a+rx %s'%self.plot_path)

        self.hist_path += '/'+self.date+'_'+self.job_name+'/'
        print('Creating output directory: %s'%self.hist_path)
        if not os.path.exists(self.hist_path):
            os.mkdir(self.hist_path)
        os.system('chmod a+rx %s'%self.hist_path)
        self.out_file = ROOT.TFile.Open(self.hist_path+'histograms.root','RECREATE')

    def read_bkg_from_csv(self,file_name,is_mc=True):
        print('reading from file % s'%file_name)
        self.bkg_df = pd.read_csv(file_name,delimiter=' ',na_values=[-999])
        if is_mc:
            self.bkg_df['weight'] *= self.mc_lumi
        self.df = self.bkg_df
        print(' Read %i raw events from file' % len(self.bkg_df) )
    
    def read_bkg_from_root(self,file_name,is_mc=True):
        self.bkg_df = self.root_to_df(file_name)
        if is_mc:
            self.bkg_df['weight'] *= self.mc_lumi
        self.df = self.bkg_df

    def read_sig_from_csv(self,file_name):
        print('reading from file %s'%file_name)
        self.sig_df = pd.read_csv(file_name,delimiter=' ',na_values=[-999])
        self.sig_df['weight'] *= self.mc_lumi
        if self.bkg_df is not None:
            start_ind = self.bkg_df.index.max()+1
            len_ind = len(self.sig_df)
            self.sig_df.index = np.arange(start_ind,start_ind+len_ind)
        else:
            self.sig_df.index = np.arange( len(self.sig_df) )
            self.df = self.sig_df

    def read_sig_from_root(self,file_name):
        self.sig_df = self.root_to_df(file_name)
        self.sig_df['weight'] *= self.mc_lumi
        if self.bkg_df is not None:
            start_ind = self.bkg_df.index.max()+1
            len_ind = len(self.sig_df)
            self.sig_df.index = np.arange(start_ind,start_ind+len_ind)
        else:
            self.sig_df.index = np.arange( len(self.sig_df) )
            self.df = self.sig_df

    def inject_sig(self,dsid,mult=1):
        #Select signal events by DSID and scale by mult
        if self.sig_df is None:
            print(' No signal events have been read. Run read_sig_from_root first')
            return
        sig_to_inj = self.sig_df[self.sig_df['mcChannelNumber']==dsid]
        sig_to_inj['weight'] *= mult
        self.df=self.bkg_df.append( sig_to_inj )
        self.df.index = np.arange(len(self.df))
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

    def drop_3jet_events(self):
        print('Dropping 3-jet events')
        mask = self.df['njet']>3
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
                                                       self.df['jet_eta_'+str(i)].values)
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
        elif self.template_type == 3:
            #leading 2 jets vs. 3rd jet
            for i in range(1,5):
                result = jitfunctions.apply_get_temp_bin(self.df['jet_pt_'+str(i)].values,
                                                         self.df['jet_eta_'+str(i)].values,
                                                         np.repeat(int(i<=2),len(self.df)),
                                                         self.pt_bins,
                                                         self.eta_bins)
                self.df['jet_temp_bin_'+str(i)]=pd.Series(result,index=self.df.index,name='jet_temp_bin_'+str(i))

    def read_templates(self,templates_file):
        print('Reading templates from file %s'%templates_file)
        f = ROOT.TFile.Open(templates_file)
        hist_dict = {}

        for key in f.GetListOfKeys():
            if key.GetName().startswith('temp_'):
                bin = int(key.GetName().split('_')[1])
                hist_dict[bin] = f.Get(key.GetName())
        self.templates_array = np.zeros( (len(hist_dict), 50) )
        
        for key in sorted( hist_dict.keys() ):
            self.templates[key] = helpers.template()
            self.templates[key].set_from_hist( hist_dict[key] )
            self.templates_array[key]=self.templates[key].probs

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

        key_list = [key for key in key_list if key!=-1]

        self.templates_array = np.zeros((len(key_list),50))
        self.templates_neff_array = np.zeros((len(key_list),50))

        for key in key_list:
            if key == -1:
                continue
            key = np.asscalar(key)
            self.templates[key] = helpers.template()
            self.templates[key].set_from_df(self.n_template_bins,df_temps,key,self.template_min,self.template_max)
            self.templates_array[key]=self.templates[key].probs
            self.templates_neff_array[key]=self.templates[key].n_eff

    def get_rms_err( self, kin_mean, dressed_mean ):
        result = 0.0
        count = 0
        for i in range(dressed_mean.shape[0]):
            if dressed_mean[i] > 1e-6 and kin_mean[i] > 1e-6:
                result += np.square ( (kin_mean[i] - dressed_mean[i])/dressed_mean[i] )
                count+=1
        result /= count
        result = np.sqrt(result)
        return result

    def get_max_err( self, kin_mean, dressed_mean ):
        max_err = 0.0
        for i in range(dressed_mean.shape[0]):
            this_err = abs( kin_mean[i] - dressed_mean[i] ) / dressed_mean[i]
            if this_err > max_err:
                max_err = this_err
        return max_err

    def compute_uncertainties(self):
        #Loop over UDRs and calculate uncertainties
        self.jet_uncert = []
        for eta_bin in ['e1','e2','e3','e4']:
            udr_str = '2jLJG400'+eta_bin
            kin_mean,dressed_mean,err = self.get_response(udr_str)
            self.jet_uncert.append( self.get_rms_err( kin_mean[:7], dressed_mean[:7] ))
            self.jet_uncert.append( self.get_max_err( kin_mean[7:10], dressed_mean[7:10] ))
            self.jet_uncert.append( self.get_rms_err( kin_mean[10:], dressed_mean[10:] ))
        for i,uncert in enumerate(self.jet_uncert):    
            print (' uncert bin %i, value = %.2f ' % (i,uncert) )
 
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
                                                         self.templates[0].bin_centers,
                                                         self.templates[0].bin_edges,
                                                         n_toys)
            self.dressed_mass_nom[i] = np.nan_to_num(result)

    def compute_shifted_masses(self):
        print('Generating shifted jet masses')
        self.dressed_mass_up = np.zeros( self.dressed_mass_nom.shape )
        self.dressed_mass_down = np.zeros( self.dressed_mass_nom.shape )
        for i in range(4):
            jet_i = i+1
            col_jet_pt = self.df['jet_pt_'+str(jet_i)].values
            col_jet_uncert_bin = self.df['jet_uncert_bin_'+str(jet_i)].values
            self.dressed_mass_up[i],self.dressed_mass_down[i] = jitfunctions.apply_shift_mass(self.dressed_mass_nom[i],col_jet_uncert_bin,self.jet_uncert)

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

    def plot_MJ_shifts(self,region_str,blinded=False):
        full_path = self.plot_path + region_str
        if not os.path.exists(full_path):
            print('Creating directory %s'%full_path)
            os.mkdir(full_path)

        index = helpers.get_region_index(self.df,region_str,self.eta_bins)[0]        
        kin_MJ = self.df.ix[index].MJ.values

        dressed_MJ_nom = self.dressed_MJ_nom[index]
        dressed_MJ_systs = np.array([ self.dressed_MJ_syst[i][index] for i in range(self.dressed_MJ_syst.shape[0]) ])
        weights = self.df.ix[index].weight.values

        MJ_hists = jitfunctions.apply_get_MJ_hists(kin_MJ,dressed_MJ_nom,dressed_MJ_systs,weights,self.MJ_bins)
        scale_factor = self.get_scale_factor(index)
        result = [None,None]
        if self.canvas is None:
            can_name = 'can'+plotters.get_random_string()
            self.canvas = ROOT.TCanvas(can_name,can_name,800,800)
        result[0] =  plotters.plot_MJ_shifts(MJ_hists,True,scale_factor,self.plot_path,self.canvas,region_str,self.MJ_bins,self.lumi_label,self.mc_label,blinded,self.MJ_cut)
        result[1] =  plotters.plot_MJ_shifts(MJ_hists,False,scale_factor,self.plot_path,self.canvas,region_str,self.MJ_bins,self.lumi_label,self.mc_label,blinded,self.MJ_cut)

        err_hist = result[0][2]
        dressed_hist_low_pt_up = result[0][3]
        dressed_hist_low_pt_down = result[0][4]
        dressed_hist_high_pt_up = result[1][3]
        dressed_hist_high_pt_down = result[1][4]
        self.out_file.cd()

        err_hist.SetName('err_stat_MJ_%s'%region_str)
        dressed_hist_low_pt_up.SetName('dressed_MJ_low_pt_up_%s'%region_str)
        dressed_hist_low_pt_down.SetName('dressed_MJ_low_pt_down_%s'%region_str)
        dressed_hist_high_pt_up.SetName('dressed_MJ_high_pt_up_%s'%region_str)
        dressed_hist_high_pt_down.SetName('dressed_MJ_high_pt_down_%s'%region_str)

        err_hist.Write()
        dressed_hist_low_pt_up.Write()
        dressed_hist_low_pt_down.Write()
        dressed_hist_high_pt_up.Write()
        dressed_hist_high_pt_down.Write()
        return result

    def plot_MJ(self,region_str,blinded=False):
        if self.canvas is None:
            can_name = 'can'+plotters.get_random_string()
            self.canvas = ROOT.TCanvas(can_name,can_name,800,800)

        full_path = self.plot_path + region_str

        if not os.path.exists(full_path):
            print('Creating directory %s'%full_path)
            os.mkdir(full_path)

        index = helpers.get_region_index(self.df,region_str,self.eta_bins)[0]
        kin_MJ = self.df.ix[index].MJ.values
        dressed_MJ_nom = self.dressed_MJ_nom[index]
        dressed_MJ_systs = np.array([ self.dressed_MJ_syst[i][index] for i in range(self.dressed_MJ_syst.shape[0]) ])
        weights = self.df.ix[index].weight.values
        MJ_hists = jitfunctions.apply_get_MJ_hists(kin_MJ,self.dressed_MJ_nom[index],dressed_MJ_systs,weights,self.MJ_bins)
        scale_factor = self.get_scale_factor(index)
        self.write_scale_factor( region_str, scale_factor)
        sr_yields = jitfunctions.apply_get_SR_yields(kin_MJ,dressed_MJ_nom,dressed_MJ_systs,weights,self.MJ_cut)
        self.write_sr_yields(region_str,sr_yields,blinded)
        print(sr_yields)
        result =  plotters.plot_MJ(MJ_hists,scale_factor,self.plot_path,self.canvas,region_str,self.MJ_bins,self.lumi_label,self.mc_label,blinded,self.MJ_cut)

        kin_hist = result[0]
        dressed_hist = result[1]
        err_hist = result[2]
        dressed_hist_up = result[3]
        dressed_hist_down = result[4]

        self.out_file.cd()
        dressed_hist.SetName('dressed_MJ_%s'%region_str)
        kin_hist.SetName('kin_MJ_%s'%region_str)
        err_hist.SetName('err_MJ_%s'%region_str)
        dressed_hist_up.SetName('dressed_MJ_up_%s'%region_str)
        dressed_hist_down.SetName('dressed_MJ_down_%s'%region_str)

        dressed_hist.Write()
        kin_hist.Write()
        err_hist.Write()
        dressed_hist_up.Write()
        dressed_hist_down.Write()

        return result
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
        result= plotters.plot_response(response,self.plot_path,self.canvas,region_str,self.pt_bins,self.eta_bins,self.lumi_label,self.mc_label)
        dressed_hist = result[0]
        kin_hist = result[1]
        self.out_file.cd()
        dressed_hist.SetName('dressed_mass_%s'%region_str)
        kin_hist.SetName('kin_mass_%s'%region_str)
        dressed_hist.Write()
        kin_hist.Write()
        return result

    def plot_template_stats(self):
        full_path = self.plot_path + 'templates'
        if not os.path.exists(full_path):
            print('Creating directory %s'%full_path)
            os.mkdir(full_path)
        if self.canvas is None:
            self.canvas = ROOT.TCanvas('can','can',800,600)
            self.canvas2 = ROOT.TCanvas('can2','can2',800,600)
        return plotters.plot_template_stats(self.templates,self.plot_path,self.canvas,self.canvas2,self.lumi_label,self.mc_label,self.pt_bins,self.eta_bins)

    def plot_template_compare(self):
        full_path = self.plot_path + 'templates'
        if not os.path.exists(full_path):
            print('Creating directory %s'%full_path)
            os.mkdir(full_path)
        temp_bin = 0
        for i in range(len(self.pt_bins)-1):
            for j in range(len(self.eta_bins)-1):
                pt_min = self.pt_bins[i]
                pt_max = self.pt_bins[i+1]
                eta_min = self.eta_bins[j]
                eta_max = self.eta_bins[j+1]
                
                rand_str = plotters.get_random_string()
                can_name = 'can'
                if self.canvas is None:
                    self.canvas = ROOT.TCanvas(can_name,can_name,800,800)
                temp_1 = self.templates[temp_bin+60]
                temp_2 = self.templates[temp_bin]
                plotters.plot_template_compare(temp_1,temp_2,self.template_type,self.plot_path,self.canvas,self.lumi_label,self.mc_label,pt_min,pt_max,eta_min,eta_max,temp_bin,self.out_file)
                temp_bin+=1

    def verify_templates(self):
        print('Verifying templates')
        for key in sorted(self.templates.keys()):
            t = self.templates[key].probs
            for i,prob in enumerate(t):
                if prob < 0:
                    print (' template number %i, bin %i, prob = %f ' % (key,i,prob) )
            if abs(t.sum() -1) > 1e-6:
                print(' template number %i, probabilities sum to %f'%abs(t.sum()-1))
    
    def compute_dressed_MJ_nom(self):
        print('Computing nominal MJ')
        self.dressed_MJ_nom = self.dressed_mass_nom[0]+self.dressed_mass_nom[1]+self.dressed_mass_nom[2]+np.nan_to_num(self.dressed_mass_nom[3])

    def make_dressed_MJ_df(self):
        self.dressed_MJ_df = pd.DataFrame( self.dressed_MJ_nom )
        self.dressed_MJ_df.index = self.df.index

    def compute_dressed_MJ_syst(self):
        print('Computing systematically shifted MJ')
        jet_uncert_bins = np.array([self.df['jet_uncert_bin_'+str(i+1)].values for i in range(4)])
        self.dressed_MJ_syst = jitfunctions.apply_get_shifted_MJ(self.dressed_mass_nom,self.dressed_mass_up,self.dressed_mass_down,jet_uncert_bins)

    def get_scale_factor(self,index):
        kin_MJ = self.df.ix[index].MJ.values
        weights = self.df.ix[index].weight.values
        return jitfunctions.apply_get_scale_factor(kin_MJ,self.dressed_MJ_nom[index],weights,self.norm_region[0],self.norm_region[1])

    def make_webpage(self):
        plotters.make_webpage(self.plot_path)

    def get_signal_pred(self,region_str,dsid):
        this_df = self.df[self.df.mcChannelNumber==dsid]
        index = helpers.get_region_index( this_df, region_str, self.eta_bins)[0]
        kin_MJ = this_df.ix[index].MJ.values
        dressed_MJ = self.dressed_MJ_df.ix[index].as_matrix()
        weights = this_df.ix[index].weight.values
        scale_factor = self.scale_factor_dict[region_str]
        n_pred,n_obs = jitfunctions.apply_get_signal_pred(kin_MJ,dressed_MJ,weights,scale_factor,self.MJ_cut)
        self.write_signal_pred(self.MJ_cut,region_str,dsid,n_pred,n_obs)
        print(self.MJ_cut,region_str,dsid,n_pred,n_obs)

    def write_uncertainties(self):
        file_name = self.hist_path+'uncertainties.csv'
        if not hasattr(self,'uncert_csv'):
            append_write = 'w'
            self.uncert_csv = file_name
        else:
            append_write = 'a'
        with open(self.uncert_csv,append_write) as f:
            writer = csv.writer(f)
            writer.writerow(self.jet_uncert)
    
    def write_signal_pred(self,mj_cut,region_str,dsid,n_pred,n_obs):
        file_name = self.hist_path+'signal_predictions.csv'
        if not hasattr(self,'sig_pred_csv'):
            append_write = 'w'
            self.sig_pred_csv = file_name
        else:
            append_write = 'a'
        with open(self.sig_pred_csv,append_write) as f:
            writer = csv.writer(f,delimiter=' ')
            writer.writerow([mj_cut,region_str,dsid,n_pred,n_obs])

    def write_scale_factor(self,region_str,scale_factor):
        file_name = self.hist_path+'scale_factors.csv'
        if not hasattr(self,'sf_csv'):
            append_write = 'w'
            self.sf_csv = file_name
        else:
            append_write = 'a'
        with open(self.sf_csv,append_write) as f:
            writer = csv.writer(f,delimiter = ' ')
            writer.writerow([region_str,scale_factor])

    def read_scale_factors(self,file_name):
        self.scale_factor_dict = {}
        with open( file_name, 'r' ) as f:
            reader = csv.reader(f,delimiter = ' ')
            for row in reader:
                self.scale_factor_dict[row[0]] = float(row[1])
        print(self.scale_factor_dict)

    def write_sr_yields(self,region_str,sr_yields,blinded):
        kin_yield, kin_uncert, pred_yield, pred_stat, pred_syst_1, pred_syst_2 = sr_yields
        file_name = self.hist_path + 'sr_yields.csv'
        if not hasattr(self,'sr_yields_csv'):
            self.sr_yields_csv = file_name
            with open(self.sr_yields_csv,'w') as f:
                writer = csv.writer(f,delimiter = ' ')
                writer.writerow(['MJ_cut','region','n_pred','sigma_stat','sigma_syst_1','sigma_syst_2','n_obs','sigma_n_obs'])
        with open(self.sr_yields_csv,'a') as f:
            writer = csv.writer(f,delimiter = ' ')
            if blinded:
                writer.writerow([self.MJ_cut,region_str,pred_yield,pred_stat,pred_syst_2,pred_syst_1,'NaN','NaN'])
            else:
                writer.writerow([self.MJ_cut,region_str,pred_yield,pred_stat,pred_syst_2,pred_syst_1,kin_yield,kin_uncert])
