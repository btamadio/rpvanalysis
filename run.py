#!/usr/bin/env python
import rpvanalysis

a = rpvanalysis.analyzer()
a.read_bkg_from_csv('pythia_2.csv')
#a.read_sig_from_root('../bkgEstimation/samples/03_24_trigcompare/largeR_RPV10.root')
a.read_sig_from_root('../bkgEstimation/samples/RPV10_largeRtrig/RPV_RPV10_largeRtrig_0.root')
a.inject_sig(403550,mult=-1)

a.compute_temp_bins()
a.create_templates()
a.verify_templates()
a.compute_dressed_masses()

#a.make_response_plot('4js1VRb9bU')
