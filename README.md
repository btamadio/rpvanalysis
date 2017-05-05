# RPV SUSY Analysis Code

This project implements the 13 TeV RPV SUSY Multijet background prediction algorithm. 

## Installation instructions on PDSF
```
git clone https://github.com/btamadio/rpvanalysis.git

module load python/2.7-anaconda

conda create --name rpv_susy numpy

source activate rpv_susy

pip install -e .
```
Each time you relog, you'll have to re-run these two lines:

```
module load python/2.7-anaconda

source activate rpv_susy
```

# Running the analysis

## Configuration files

Example Configuration files are located in config/examples

Config files are in json format (double quotes for strings) and have some required entries, depending on which version of the analysis you want to run.

### baseline analysis with data

Example config file: `config_data.py`

Required entries:

`date`: Specify date of running the job with format `MM_DD`. This will end up in the name of the output directory

`job_name`: Appended to date to make the output directory name

`bkg_file`: csv file containing entire dataset you want to run on

`web_path`: Directory for the output plots

`hist_path`: Directory for the ROOT and csv output files

`is_mc`: false

`mc_label`: leave as empty quotes

`n_toys`: Number of dressed masses to generate per-jet for the background prediction

`inject_sig`: false

`MJ_plots`: List of regions for which to produce MJ plots

`MJ_cut` : Value of MJ cut that defines SR - used to calculated predicted and observed yields

`blinded`: Whether or not to blind the plots above MJ_blind in SRs

`MJ_blind` : If blinded == true, specify the MJ cut above which to blind the plots

### baseline analysis with MC

Example config file: `config_MC.py`

Required entries:

`date`: Specify date of running the job with format `MM_DD`. This will end up in the name of the output directory

`job_name`: Appended to date to make the output directory name

`bkg_file`: csv file containing entire dataset you want to run on

`web_path`: Directory for the output plots

`hist_path`: Directory for the ROOT and csv output files

`is_mc`: true

`mc_label`: Text to appear on plots, e.g. "Pythia" or "Herwig++"

`n_toys`: Number of dressed masses to generate per-jet for the background prediction

`inject_sig`: false

`MJ_plots`: List of regions for which to produce MJ plots

`MJ_cut` : Value of MJ cut that defines signal region (TeV)

### baseline analysis with signal MC injected into data

Example config file: `config_signal_injection.py`

Required entries:

`date`: Specify date of running the job with format `MM_DD`. This will end up in the name of the output directory

`job_name`: Appended to date to make the output directory name

`bkg_file`: csv file containing entire dataset you want to run on

`web_path`: Directory for the output plots

`hist_path`: Directory for the ROOT and csv output files

`is_mc`: false

`mc_label`: ""

`n_toys`: Number of dressed masses to generate per-jet for the background prediction

`inject_sig`: true

`sig_file`: location of signal MC csv file

`sig_mult`: scale factor by which to multiply signal cross-section. Generally this should be set to 1.0

`sig_dsid`: DSID of signal point to inject into data

`MJ_plots`: List of regions for which to produce MJ plots

`MJ_cut` : Value of MJ cut that defines signal region (TeV)

`blinded`: true to blind SRs above MJ_cut

### Generate predictions and observed yields from signal MC *only*

Example config file: `config_signal_only.py`

Required entries:

`date`: Specify date of running the job with format `MM_DD`. This will end up in the name of the output directory

`job_name`: Appended to date to make the output directory name

`sig_file`: location of signal MC csv file

`hist_path`: Directory for the csv output files

`n_toys`: Number of dressed masses to generate per-jet for the background prediction

`templates_file`: location of ROOT file containing templates, i.e. the histograms.root file that comes from the baseline analysis

`scale_factor_file`: location of .csv file containing the scale factors, i.e. scale_factors.csv from baseline analysis

## Running the code

### baseline analysis or signal-injected analysis:

```
bin/run-2jet-analysis <config_file>
```

### make predictions from signal MC only

```
bin/run-signal-predictions <config_file>
```

## Making input .csv files

The input files are in .csv format, and can be created from a root file using the following:

```
bin/convert <input_root_file> <output_csv_file>
```

The input root file must contain a TTree at the top level called "miniTree" with the following branches:

### scalars

`eventNumber`

`njet`

`njet_soft`

`nbjet_Fix70`

`MJ`

`dEta`

`weight`

### vectors

`jet_pt`

`jet_eta`

`jet_phi`

`jet_m`

`jet_bmatched_Fix70`

## Outputs

### Baseline analysis

`histograms.root` : templates and MJ histograms

`sr_yields` : predicted and observed (if unblinded) SR yields with uncertainties

`scale_factors.csv` : scale factors for normalizing prediction by region

`uncertainties.csv` : data-driven uncertainty in the following order: 
[ (pt_bin_1,eta_bin_1),
(pt_bin_2,eta_bin_1),
(pt_bin_3,eta_bin_1),
(pt_bin_1,eta_bin_2),
(pt_bin_2,eta_bin_2),
(pt_bin_3,eta_bin_2),
(pt_bin_1,eta_bin_3),
(pt_bin_2,eta_bin_3),
(pt_bin_3,eta_bin_3),
(pt_bin_1,eta_bin_4),
(pt_bin_2,eta_bin_4),
(pt_bin_3,eta_bin_4)]

### signal-only-predictions

`signal_predictions.csv` : columns are : MJ_cut, region, DSID, n_predicted, n_observed

## Author

* **Brian Amadio**
btamadio@gmail.com

See also the list of [contributors](https://github.com/btamadio/rpvanalysis/contributors) who participated in this project.

