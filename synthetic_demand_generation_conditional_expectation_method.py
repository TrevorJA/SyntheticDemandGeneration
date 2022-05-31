"""
Trevor Amestoy
Cornell University
Spring 2022

This script creates timeseries of synthetic demand data and attempts to
preserve correlation between streamflow and demand in synthetic timeseries.

The process generates demand based upon conditional expectations of
demand mean and variance given some synthetic inflow.

Process generally follows these steps:
1. Standardize both historic inflow and demand according to week of the year,
    to remove seasonality.
2. Calculate the correlation between the standardized values.
3. Load and standardize synthetic inflow generated using the Kirsch et al. (2013) method.
4. Generate a synthetic standard demand value for each synthetic flow values
    using the conditional expectations of mean and varaince of demand.
5. Apply weekly mean expectations and variance to demand for the final values.

"""

# Core functions
import numpy as np
from numpy import random
from scipy.stats import pearsonr

# Load other functions of interest
from my_stats_functions import standardize



################################################################################
                        ### Constants  ###
################################################################################

n_years_demand = 18
n_years_inflow = 81
n_weeks = 52

first_row_to_match = n_years_inflow - n_years_demand

# constants definition the irrigation seasons
nw = 23
nn1 = 16
nn2 = 13



################################################################################
                        ### Load historic data  ###
################################################################################

# Historic demand data
raw_unit_demands = np.loadtxt('./data/historic/historic_unit_demand.csv', delimiter = ',').transpose()

# Inflow data
raw_inflow =  np.loadtxt('./data/historic/historic_inflow.csv', delimiter = ',').transpose()


################################################################################
            ### Standardize inflow and demand data by WEEK ####
################################################################################

# Demand [18 x 52]
standard_demand, demand_means, demand_std_dev = standardize(raw_unit_demands, columns = True)

# Inflow [81 x 52]
log_inflow = np.log(raw_inflow)
standard_log_inflow, log_inflow_means, log_inflow_std_dev = standardize(log_inflow, columns = True)



################################################################################
            ### Calculate correlation of standard data ###
################################################################################

### Correlation
combined_historic = np.column_stack((standard_log_inflow.reshape((52*18)), standard_demand.reshape((52*18))))
combined_irrigated_historic = np.column_stack((standard_log_inflow[:, nn1:(nn1+nw)].reshape((nw*18)), standard_demand[:,nn1:(nn1+nw)].reshape(nw*18)))


historic_annual_correlation = pearsonr(combined_historic[:,0], combined_historic[:,1])[0]
historic_irrigated_correlation = pearsonr(combined_irrigated_historic[:,0], combined_irrigated_historic[:,1])[0]


weekly_correlation = np.zeros(n_weeks)

for week in range(n_weeks):
    weekly_combine = np.column_stack((standard_log_inflow[:,week], standard_demand[:,week]))
    weekly_correlation[week] = pearsonr(weekly_combine[:,0], weekly_combine[:,1])[0]


################################################################################
        ### Load and standardize synthetic inflow ###
################################################################################

# Load synthetic streamflow
synthetic_inflow = np.loadtxt('./data/synthetic/synthetic_inflow.csv', delimiter = ',')

# Check the size
n_years, n_weeks = np.shape(synthetic_inflow)

standard_log_synthetic_inflow = np.zeros_like(synthetic_inflow)

# Standardize synthetic data using historic mean and std. dev
for year in range(int(n_years)):
    for week in range(n_weeks):
        standard_log_synthetic_inflow[year, week] = (np.log(synthetic_inflow[year, week]) - log_inflow_means[week]) / log_inflow_std_dev[week]



################################################################################
        ### Calculate flow-conditional standard synthetic demand ###
################################################################################

# Initialize synthetic demand matrix
standard_synthetic_demand = np.zeros_like(standard_log_synthetic_inflow)

for year in range(n_years):
    for week in range(n_weeks):

        # Check if the week is in the irrigation season
        if week > nn1 and week < (nn1 +nw):
            # conditional variance
            correlation = historic_irrigated_correlation
        else:
            correlation = historic_annual_correlation

        conditional_var = (1 - correlation**2)

        # extract synthetic inflow of interest
        weekly_synthetic_inflow = standard_log_synthetic_inflow[year, week]

        conditional_expectation = correlation * weekly_synthetic_inflow

        # Sample from normal distribution with mean of conditional expectation and conditional std dev
        standard_synthetic_demand[year, week] = random.normal(loc = conditional_expectation, scale = np.sqrt(conditional_var))



################################################################################
        ### Re-scale from standard to regular using projections ###
################################################################################

# Initialize final synthetic demand Matrix
synthetic_unit_demand_realizations = np.zeros_like(synthetic_inflow)

# Build record
for year in range(n_years):
    for week in range(n_weeks):
        synthetic_unit_demand_realizations[year, week] = (standard_synthetic_demand[year, week] * demand_std_dev[week]) + demand_means[week]

# EXPORT
np.savetxt('./data/synthetic/synthetic_unit_demand_CEM.csv', synthetic_unit_demand_realizations, delimiter = ',')
