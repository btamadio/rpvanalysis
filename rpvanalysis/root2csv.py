#!/usr/bin/env python
from __future__ import print_function
import ROOT,csv
class root2csv:
    def __init__(self,in_file_name,out_file_name,tree_name='miniTree'):
        self.in_file = ROOT.TFile.Open(in_file_name)
        self.out_file = out_file_name
        self.tree = self.in_file.Get(tree_name)
        self.n_entries = self.tree.GetEntries()
        print('Converting root file: %s'%in_file_name)
        print('Number of entries: %i'%self.n_entries)
    def loop(self):
        feature_list = ['eventNumber','njet','njet_soft','nbjet_Fix70','MJ','dEta','weight',
                        'jet_pt','jet_eta','jet_phi','jet_m','jet_bmatched_Fix70']
        with open(self.out_file,'wb') as csvfile:
            writer = csv.writer(csvfile,delimiter=' ')
            header = ['event_number','njet','njet_soft','nbjet','MJ','dEta','weight']
            header += ['jet_pt_'+str(i) for i in range(1,5)]
            header += ['jet_eta_'+str(i) for i in range(1,5)]
            header += ['jet_phi_'+str(i) for i in range(1,5)]
            header += ['jet_m_'+str(i) for i in range(1,5)]
            header += ['jet_bmatched_'+str(i) for i in range(1,5)]
            writer.writerow(header)
            for entry in range(self.n_entries):
                if entry % 1e5 == 0:
                    print ('processing entry %i' % entry)
                row = []
                self.tree.GetEntry(entry)
                for feature in feature_list:
                    branch = getattr(self.tree,feature)
                    if '.vector<' in str(type(branch)):
                        vec = getattr(self.tree,feature)
                        for value in vec[:4]:
                            row.append(value)
                        for _ in range(4-len(vec)):
                            row.append(-999)
                    else:
                        value = getattr(self.tree,feature)
                        row.append(value)
                writer.writerow(row)

r2c=root2csv('../bkgEstimation/samples/03_24_trigcompare/largeR_data_full.root','data_largeR.csv')
r2c.loop()
#r2c = root2csv('../bkgEstimation/samples/pythia_combined_mass/main_pythia_combined_mass.root','pythia_2.csv')
#r2c.loop()
        
