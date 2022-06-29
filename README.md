# ConfLearning


This repository provides experimental data, analysis scripts
as well as the computational models for the following
publication:

The value of confidence: Confidence prediction errors drive value-based learning in the absence of external feedback 
(https://doi.org/10.31234/osf.io/wmv89).

It is subdivided into several folders:

1. **data** contains the entire behavioral dataset *[data.pkl]* as well as an *[extraction.py]*-file
through which variable-specific .npy-arrays can be extracted. Moreover, simulated data and 
experimental protocols are saved in the **sim** and **para_experiment** folder, respectively.

2. **model** contains different versions of the computational models either
including *[rl_simple_simchoice.py]* or excluding *[rl_simple.py]* simulated choices
as well as optimization scripts *[maximum_likelihood.py]*.

3. **plot** contains visualization scripts for our publication. The resulting figures are saved
under **figures**. Please refer to the publication for further detail.

4. **revision2** contains scripts for calculating and aggregating results of 
parameter recovery, model recovery and the models' generative performance. 

5. **run_model** contains model fitting scripts with parameter bounds. Model-specific 
parameter estimates and model evidences are saved in .pkl-format under **results/fittingData/**.

6. **stats** contains data analysis scripts. Please refer to the publication for further detail.
