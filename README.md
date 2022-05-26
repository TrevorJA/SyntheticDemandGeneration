# SyntheticDemandGeneration
Contains scripts to generate correlated synthetic streamflow and demand timeseries and analyze the results. 

## Purpose
Generation of synthetic timeseries data allows for the assessment of water resource policy performance under a broad range of possible scenarios. When generating timeseries for multiple different variables, it is important to consider the correlations betweeen variable timeseries. The code in this repository can be used to produce correlated synthetic timeseries for water supply and water demand, given historic data for both variables. 

## Methods
Historic streamflow into a reservoir and historic water demands are provided for one location. 
Synthetic streamflow timeseries are generated using a modified Fractional Gaussian Noise (mFGN) generation method, as described by Kirsh et al. (2013).
Synthetic water demand timeseries are then generated conditional upon the corresponding synthetic streamflow and historic correlation. Two different methods for synthetic demand generation are available in this repo:
1)
2) 

## Contents
mFGN_generator.py
mFGN_generation_main.py
synthetic_demand_generation_CEM.py
synthetic_demand_generation_JPM.py

correlation_analysis_historical.py
correkatiion_analysis_synthetic.py

visual_validation.py

my_stats_functions.py
fix_nonpositive_matrix.py
synthetic_data_transformation.py

## References
Kirsch et al. (2013)
