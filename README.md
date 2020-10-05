# HIV Project util functions

This python package regroups useful functions and classes, that were used for the study of resistance mutations, and the search for potentially resistance-associated mutations in HIV-1 Reverse transcriptase sequences, using machine learning methods.  
You can read about this project [here](https://research.pasteur.fr/en/project/drm-hiv/) and [here](https://research.pasteur.fr/en/project/applying-machine-learning-to-sequence-analysis-phd-luc-blassel-prairie/)

## Module description
This module is separated into 5 different submodules

### DRM utils
This submodule contains all functions to get different subsets of DRMs (ie. NRTIs, NNRTIs, accessory DRMs, SDRMs, etc...). Each of the functions returns a list of selected DRMs. 

### data utils
This submodule contains useful functions and classes to pre-process the encoded dataset before model training. You can remove features corresponding to known DRMs, remove sequences that have DRMs, balance target classes by sub-sampling or over-sampling, or creating cross-validation folds. 

### learning utils
This submodule contains useful functions and classes to use classifiers needed during the study. It also contains custom classifiers based on exact fisher tests. It contains functions to train classifiers, get predictions from these classifiers and extract coefficients / weights from these classifiers. 

### param utils
This submodule contains functions useful for the generation and selection of the best hyper-parameter set via random search. 

### metrics
This submodule contains a set of custom performance metrics that we devised in an attempt to take into account class imbalance and the differing importance given to False positives (more important) and False negatives (less important).

## independent scripts
Additionally, two useful scripts are present. 

### compute_fisher_values.py
This script allows us to compute p-values for Fisher exact tests comparing the prevalence of mutations w.r.t a binary character like RTI treatment status or presence/absence of any DRM. This outputs a table with each considered mutation in a row and the raw p-value, as well as p-values corrected for multiple testing with the Bonferroni, Benjamini-Hochberg or Benjamini-Yekutieli methods. This script was used to generate the table: `utils_hiv/data/fisher_p_values.tsv`

### data_encoder.py
This script is used to create the OneHot encoded dataset from [HIVDB](https://hivdb.stanford.edu/hivdb/by-sequences/) files and an additional metadata file.  
To run this script you need the `PrettyRTAA_naive.tsv` and `PrettyRTAA_treated.tsv` generated by submitting the `naive.fa` and `treated.fa` fasta alignments to the [HIVDB sequence program](https://hivdb.stanford.edu/hivdb/by-sequences/). This also outputs `ResistanceSummary_naive.tsv` and `ResistanceSummary_treated.tsv` which are needed for the script to run.  
This script can be used to specify starting and ending positions. 

## data files
These files are in `utils_hiv/data` and are used by submodules. 

### DRM files
`NRTI.tab` and `NNRTI.tab` are local copies of HIVDB files ([1](https://hivdb.stanford.edu/pages/SDRM.worksheet.NRTI.html), [2](https://hivdb.stanford.edu/pages/SDRM.worksheet.NNRTI.html)).  
`mutation_characteristic.tab` is used by the `DRM_utils` submodule and contains known DRMs with their type (NRTI,NNRTI,Other), their SDRM status. This was obtained through the [HIVDB program](https://hivdb.stanford.edu/hivdb) and hand-curated. The accessory/primary role of each mutation was determined by the HIVDB program comment. 