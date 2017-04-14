#!/usr/bin/env python
import rpvanalysis

a = rpvanalysis.analyzer()
a.date='04_13'
a.job_name='data'
a.read_bkg_from_csv('data_largeR.csv',is_mc=False)
#a.read_sig_from_root('../bkgEstimation/samples/RPV10_largeRtrig/RPV_RPV10_largeRtrig_11.root')
#a.inject_sig(403566,mult=-1)

a.compute_temp_bins()
a.create_templates()
a.verify_templates()
a.compute_dressed_masses()

#a.plot_response(a.get_response('4js1VRb9bU'),'4js1VRb9bU',-1)
