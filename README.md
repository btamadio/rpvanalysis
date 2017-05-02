# RPV SUSY Analysis Code

This project implements the 13 TeV RPV SUSY Multijet background prediction algorithm. 

## Getting started

### Installation instructions on PDSF
```
git clone https://github.com/btamadio/rpvanalysis.git

module load python/2.7-anaconda

conda create --name rpv_susy numpy

source activate rpv_susy

pip install -e .
```
## Running the analysis

### Configuration files

Configuration files are located in config/

Look at the most recent date for the most up-to-date syntax. Config files from older dates may be obsolete.

Config files are in json format (double quotes for strings) and have some required settings:

`date`: Specify date of running the job `MM_DD`. This will end up in the name of the output directory
`job_name`: Appended to date to make the output directory name
`bkg_file`: Specify csv file of either data or background MC
`web_path`: Directory for the output plots
`hist_path`: Directory for the ROOT files containing output histograms
`is_mc`: set to true if the input for background is MC. 
`mc_label`: Name of MC generator for labelling plots
`n_toys`: Number of dressed masses to generate per-jet for the background prediction
`inject_sig`: To inject signal MC into either data or background MC, set this to true and specify and input file for signal
`sig_files`: List of signal files to input
`sig_mult`: Scales the signal cross-section by this value. If set to 1, the signal MC will be scaled to the nominal cross-section times 36.45/fb.
`sig_dsid`: DSID of signal to include. In case the input signal file contains multiple DSIDs.
`template_type`: 0 = baseline b-match / non-b-match template binning
`MJ_plots`: List of regions for which to produce MJ plots
`blinded`: If set to true, MJ plots for SRs will be blinded about 800 GeV

## Author

* **Brian Amadio**
btamadio@gmail.com

See also the list of [contributors](https://github.com/btamadio/rpvanalysis/contributors) who participated in this project.

