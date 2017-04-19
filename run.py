#!/usr/bin/env python
import rpvanalysis
import json
import sys

if len(sys.argv) < 2:
    print('Must specify configuration file.')
    sys.exit(1)
print(sys.argv[1])

with open(sys.argv[1]) as f:
    config = json.load(f)

a = rpvanalysis.analyzer()

a.date=config['date']
a.job_name=config['job_name']
a.web_path = config['web_path']
if config['is_mc']:
    a.mc_label = config['mc_label']

a.make_plot_dir()
a.read_bkg_from_csv(config['bkg_file'],is_mc=config['is_mc'])

if config['inject_sig']:
    a.read_sig_from_root(config['sig_file'])
    a.inject_sig(config['sig_dsid'],config['sig_mult'])

a.compute_temp_bins()
a.create_templates()
a.verify_templates()
a.compute_dressed_masses(config['n_toys'])
a.compute_uncertainties()
a.compute_dressed_masses(config['n_toys'])

a.compute_dressed_MJ()


#if 'response_plots' in config:
#    [a.plot_response(region_str) for region_str in config['response_plots']]
#if 'mass_plots' in config:
#    [a.plot_mass(region_str) for region_str in config['mass_plots']]
#if 'MJ_plots' in config:
#    [a.plot_MJ(region_str) for region_str in config['MJ_plots']]
