# SyntheticDemandGeneration
Contains scripts to generate correlated synthetic streamflow and demand timeseries, along with scripts to analyze and visualize correlation patterns in historic and synthetically generated data. 

## Purpose
Generation of synthetic timeseries data allows for the assessment of water resource policy performance under a broad range of possible scenarios. When generating timeseries for multiple different variables, it is important to consider the correlations betweeen variable timeseries. The code in this repository can be used to produce correlated synthetic timeseries for water supply and water demand, given historic data for both variables. 

## Methods
Historic streamflow into a reservoir and historic unit water demands are provided for one location. 
Synthetic streamflow timeseries are generated using a modified Fractional Gaussian Noise (mFGN) generation method, as described by Kirsh et al. (2013).
Synthetic water demand timeseries are then generated conditional upon the corresponding synthetic streamflow and historic correlation. Two different methods for synthetic demand generation are available in this repo:
1) A conditional expectation method
2) A joint PDF sampling method

For more information on these methods, see the blog post [here](https://waterprogramming.wordpress.com/).

## Contents and Order of Execution

### Analysis of historic correlation patterns
[historic_correlation_analysis.py](https://github.com/TrevorJA/SyntheticDemandGeneration/blob/main/historic_correlation_analysis.py)
> Used to quantify and visualize historic correlation patterns between reservoir inflow and water demand. 
  
### Generating synthetic inflow timeseries
[mFGN_generator.py](https://github.com/TrevorJA/SyntheticDemandGeneration/blob/main/mFGN_generator.py)
> Contains the code used to generate synthetic streamflow using the modified factional gaussian noise (mFGN) method described by Kirsch et al. (2013).
[mFGN_generation_main.py](https://github.com/TrevorJA/SyntheticDemandGeneration/blob/main/mFGN_generation_main.py)
> Runs the mFGN generator and saves synthetic streamflow data as a csv file; also plots a comparison of the historic and synethtic flow duration curves. 

### Generating synthetic demand timeseries
[synthetic_demand_generation_conditional_expectation_method.py](https://github.com/TrevorJA/SyntheticDemandGeneration/blob/main/synthetic_demand_generation_conditional_expectation_method.py)
> Generates synthetic unit demand timeseries for 50 years, using a conditional expectation method based upon historic correlation values. 
[synthetic_demand_generation_joint_PDF_method.py](https://github.com/TrevorJA/SyntheticDemandGeneration/blob/main/synthetic_demand_generation_joint_PDF_method.py)
> Generates synthetic unit demand timeseries for 50 years, by sampling from a joint PDF of the historic inflow-demand data. 

### Analyzing synthetic correlation patterns
[synthetic_correlation_analysis_conditional_expectation_method.py](https://github.com/TrevorJA/SyntheticDemandGeneration/blob/main/synthetic_correlation_analysis_conditional_expectation_method.py)
> Quantifies the correlation between synthetic inflow data generated using the mFGN method with synthetic demand using the conditional expectation method. Produces several figures to visualize these patterns and report Pearson correlation values.  
[synthetic_correlation_analysis_joint_PDF_method.py](https://github.com/TrevorJA/SyntheticDemandGeneration/blob/main/synthetic_correlation_analysis_joint_PDF_method.py)
> Quantifies the correlation between synthetic inflow data generated using the mFGN method with synthetic demand using the conditional expectation method. Produces several figures to visualize these patterns and report Pearson correlation values.

### Additional files
The following files are necessary modules used during the generation process. 
[visual_validation.py](https://github.com/TrevorJA/SyntheticDemandGeneration/blob/main/visual_validation.py)
> Contains plotting functions used to produce visuals.  

[my_stats_functions.py](https://github.com/TrevorJA/SyntheticDemandGeneration/blob/main/my_stats_functions.py)
> Contains simple standardization functions and matrix transformation functions.  

[fix_nonpositive_matrix.py](https://github.com/TrevorJA/SyntheticDemandGeneration/blob/main/fix_nonpositive_matrix.py)
> Used to fix non-positive definite matrices that result from numerical errors. Source: [PyPortfolioOpt](https://pyportfolioopt.readthedocs.io/en/latest/_modules/pypfopt/risk_models.html)

## References
Kirsch, B. R., Characklis, G. W., & Zeff, H. B. (2013). Evaluating the impact of alternative hydro-climate scenarios on transfer agreements: Practical improvement for generating synthetic streamflows. Journal of Water Resources Planning and Management, 139(4), 396-406.

Zeff, H. B., Herman, J. D., Reed, P. M., & Characklis, G. W. (2016). Cooperative drought adaptation: Integrating infrastructure development, conservation, and water transfers into adaptive policy pathways. Water Resources Research, 52(9), 7327-7346.
