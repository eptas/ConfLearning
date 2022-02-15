# ConfLearning


This repository provides experimental data, analysis scripts
as well as the computational models for the following
publication:

The value of confidence: Confidence prediction errors drive value-based learning in the absence of external feedback 
(https://doi.org/10.31234/osf.io/wmv89).

It is subdivided into several folders:

1. **data** contains the entire dataset *[data.pkl]* as well as an *[extraction.py]*-file
through which variable-specific .npy-arrays can be extracted.

2. **model** contains different versions of the computational models either
including *[rl_simple_simchoice.py]* or excluding *[rl_simple.py]* simulated choices
as well as an optimization script *[maximum_likelihood.py]*.

3. **plot** contains visualization scripts for our publication. Figures are saved
under **figures** and seperately for behaviour-specific (**/behav/**) and 
modelling-specific (**/model/**) outcomes.

4. **run_model** contains model fitting scripts with parameter bounds. Model-specific 
parameter estimate and model evidences are saved in .pkl-format under **results/fittingData/**.

5. **stats** contains data analysis scripts. Please refer to the publication for further detail.
