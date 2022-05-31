"""
Trevor Amestoy
Cornell University
Spring 2022

This script creates timeseries of synthetic demand data, conditional upon
historic joint distributions of inflow and demand, and dependent upon
corresponding weekly synthetic inflow.

The method follows those implemented in:
Zeff, H. B., Herman, J. D., Reed, P. M., & Characklis, G. W. (2016).
Cooperative drought adaptation: Integrating infrastructure development,
conservation, and water transfers into adaptive policy pathways.
Water Resources Research, 52(9), 7327-7346.

Process generally follows these steps:
1. Standardize historic inflow and demand data.
2. Construct a joint PDF from the historic data
3. Load synthetic inflow data generated previously
4. For each weekly synthetic inflow, sample from the joint PDF, conditional
    upon the corresponding inflow.
5. Apply some noise to the sampled standard demand.
6. Re-apply historic weekly mean and standard deviations to the new
    synthetic demands.

"""

# Core functions
import numpy as np
import matplotlib.pyplot as plt
import random
import statistics as stats
import seaborn as sns

# Load other functions of interest
from my_stats_functions import standardize
from mFGN_generator import FGN_generate



################################################################################
                        ### Load historic data  ###
################################################################################

# Historic demand data
raw_unit_demands = np.loadtxt('./data/historic/historic_unit_demand.csv', delimiter = ',').transpose()

# Inflow data
raw_inflow =  np.loadtxt('./data/historic/historic_inflow.csv', delimiter = ',').transpose()


### Constants ###
n_years_demand = 18
n_years_inflow = 81
n_weeks = 52

first_row_to_match = n_years_inflow - n_years_demand


################################################################################
            ### Normalize inflow and demand data by WEEK ####
################################################################################

# Demand [18 x 52]
standardized_demand, demand_means, demand_sds = standardize(raw_unit_demands, columns = True)

# Inflow [81 x 52]
log_inflow = np.log(raw_inflow)
standardized_log_inflow, log_inflow_means, log_inflow_sd = standardize(log_inflow, columns = True)


################################################################################
            ### Build PDF between inflow and demand ### (117)
################################################################################

# constants definition the irrigation seasons
nw = 23
nn1 = 16
nn2 = 13

# specify pdf dimensions (WHY THESE?)
pdf_rows = 16
pdf_columns = 16

# Initialize matrices
inflow_demand_pdf_irrigated = np.zeros((pdf_rows, pdf_columns))
inflow_demand_pdf_nonirrigated = np.zeros((pdf_rows, pdf_columns))

inflow_demand_cdf_irrigated = np.zeros((pdf_rows, pdf_columns))
inflow_demand_cdf_nonirrigated = np.zeros((pdf_rows, pdf_columns))

# Separate irrigated and non-irrigated weeks
irrigated_inflows = standardized_log_inflow[:, nn1:(nn1+nw)]
irrigated_demands = standardized_demand[:, nn1:(nn1+nw)]

nonirrigated_inflows = np.hstack((standardized_log_inflow[:, 0:int(nn1)], standardized_log_inflow[:, int(nn1+nw):]))
nonirrigated_demand = np.hstack((standardized_demand[:, 0:nn1], standardized_demand[:, (nn1+nw):]))

# Convert 2D inflow & demand matrices into 1D arrays: progress through year
# Only use data for overlapping years
I_irrigated = irrigated_inflows.ravel()
D_irrigated = irrigated_demands.ravel()

I_nonirrigated = nonirrigated_inflows.ravel()
D_nonirrigated = nonirrigated_demand.ravel()

# Check that selected 'overlapping' data are of equal length
if len(I_irrigated) != len(D_irrigated):
    print('Uh-Oh, looks like the selected inflow and demand data are different lengths. \n \
    Check the indexing, and make sure only overlapping years are used here.')
    exit()

## Make demand-inflow PDF ##
pdf_dim1 = 16;
pdf_dim2 = pdf_dim1

# Initialize sequences
inflow_bins = np.arange((-1 * pdf_dim1 / 4 + 0.5), (pdf_dim1 / 4 + 0.5), 0.5)
demand_bins = np.arange((-1 * pdf_dim2 / 4 + 0.5), (pdf_dim2 / 4 + 0.5), 0.5)

# Loop through irrigation season data
for i in range(len(I_irrigated)):

    # Begin inflow-PDF index (y_count)
    y_count = 0

    # Identify the bin which the inflow data belongs in
    for y in inflow_bins:
        if I_irrigated[i] < y and I_irrigated[i] >= y-0.5:

            # Begin demand-PDF index (z_count)
            z_count = 0

            # Identify the bin which the demand data belongs in
            for z in demand_bins:
                if D_irrigated[i] < z and D_irrigated[i] >= z - 0.5:

                    # If conditions are met, add one to bin counter
                    inflow_demand_pdf_irrigated[y_count, z_count] += 1

                # If conditions are not met, shift z_counter (demand-PDF index) over 1
                else:
                    z_count += 1

        else:# If conditions are not met, shift y_counter (inflow-PDF index) over 1
            y_count += 1

# Loop through NON-irrigation season data
for i in range(len(I_nonirrigated)):

    # Begin inflow-PDF index (y_count)
    y_count = 0

    # Identify the bin which the inflow data belongs in
    for y in inflow_bins:
        if I_nonirrigated[i] < y and I_nonirrigated[i] >= y-0.5:

            # Begin demand-PDF index (z_count)
            z_count = 0

            # Identify the bin which the demand data belongs in
            for z in demand_bins:
                if D_nonirrigated[i] < z and D_nonirrigated[i] >= z - 0.5:

                    # If conditions are met, add one to bin counter
                    inflow_demand_pdf_nonirrigated[y_count, z_count] += 1

                # If conditions are not met, shift z_counter (demand-PDF index) over 1
                else:
                    z_count += 1

        else:# If conditions are not met, shift y_counter (inflow-PDF index) over 1
            y_count += 1



## Calculate the CDFs (sums) of the PDFs
for col in range(1,pdf_columns):
    inflow_demand_cdf_irrigated[:, col] = np.sum(inflow_demand_pdf_irrigated[:,0:col], axis = 1)
    inflow_demand_cdf_nonirrigated[:, col] = np.sum(inflow_demand_pdf_nonirrigated[:,0:col], axis = 1)


################################################################################
        ### Apply historic record statistics to synthetic record ###
################################################################################

# Load synthetic streamflow (consist of both Michie and LittleRiver): [1,000 SOWs x M weeks]
synthetic_inflow = np.loadtxt('./data/synthetic/synthetic_inflow.csv', delimiter = ',')

# Check the size
n_years, n_weeks = np.shape(synthetic_inflow)


### Whiten synthetic record according to historic means and sds
standardized_log_inflow_synthetic = np.zeros_like(synthetic_inflow)


for year in range(n_years):
    for week in range(n_weeks):
        standardized_log_inflow_synthetic[year, week] = (np.log(synthetic_inflow[year, week]) - log_inflow_means[week]) / log_inflow_sd[week]

### Get weekly demand variation based on inflow residuals
n_synthetic_period = n_years

# initialize a matrix to store demand variation
demand_variation = np.zeros_like(synthetic_inflow)
true_pdf = np.zeros_like(inflow_demand_pdf_irrigated)

# Re-define bins from earlier:
inflow_bins = np.arange((-1 * pdf_dim1 / 4 + 0.5), (pdf_dim1 / 4 + 0.5), 0.5)
demand_bins = np.arange((-1 * pdf_dim2 / 4 + 0.5), (pdf_dim2 / 4 + 0.5), 0.5)


# loop through each synthetic year and week

for year in range(n_years):
    for week in range(n_weeks):

        ### find the index for current flow value in inflow pdf
        flow = standardized_log_inflow_synthetic[year, week]

        flow_bin = 0


        if flow < -3.5:
            flow_bin = 0
        elif flow > 4:
            flow_bin = 15
        else:
            while (inflow_bins[flow_bin] + 0.5) < flow:
                if flow_bin == 15:
                    break
                else:
                    flow_bin += 1

        # Check the number of observations in that inflow-pdf bin
        # Irrigated or non-irrigated PDF depending on week

        if (week > nn1) and (week <= (nn1 + nw)):
            obs_in_pdf_bin = inflow_demand_cdf_irrigated[flow_bin, -1]
        else:
            obs_in_pdf_bin = inflow_demand_cdf_nonirrigated[flow_bin, -1]

        # Locate
        random_draw = random.uniform(0, obs_in_pdf_bin) + 1

        demand_bin = 0

        # Locate the CDF bin with
        if (week > nn1) and (week < (nn1 + nw)):

            while inflow_demand_cdf_irrigated[flow_bin, demand_bin] < random_draw:
                demand_bin += 1
                if demand_bin >= len(demand_bins):
                    demand_bin = len(demand_bins)-1
                    break

        else:
            while inflow_demand_cdf_nonirrigated[flow_bin, demand_bin] < random_draw:
                demand_bin += 1
                if demand_bin >= len(demand_bins):
                    demand_bin = len(demand_bins)-1
                    break


        # detemine demand variance from the bin number: add some noise ([0, 0.5])
        demand_variation[year, week] = ((demand_bin - pdf_dim1/2)/2) - random.uniform(0, 501)/1000

        # Add 1 to the 'true_pdf'
        true_pdf[flow_bin, demand_bin] += 1

# Export distributions
np.savetxt('./data/historic_distributions/irrigation_season_PDF.csv', inflow_demand_pdf_irrigated, delimiter = ',')
np.savetxt('./data/historic_distributions/nonirrigation_season_PDF.csv', inflow_demand_pdf_nonirrigated, delimiter = ',')
np.savetxt('./data/historic_distributions/irrigation_season_CDF.csv', inflow_demand_cdf_irrigated, delimiter = ',')
np.savetxt('./data/historic_distributions/nonrrigation_season_CDF.csv', inflow_demand_cdf_nonirrigated, delimiter = ',')

np.savetxt('./data/historic_distributions/weekly_demand_variation.csv', demand_variation, delimiter = ',')


################################################################################
                ### Build full demand timeseries ###
################################################################################

# Initialize final synthetic demand Matrix
synthetic_unit_demand = np.zeros_like(demand_variation)

# Build record
for year in range(n_years):
    for week in range(n_weeks):
        synthetic_unit_demand[year, week] = (demand_variation[year, week] * demand_sds[week] + demand_means[week])

# EXPORT new synthetic unit demand
np.savetxt('./data/synthetic/synthetic_unit_demand_JPM.csv', synthetic_unit_demand, delimiter = ',')


################################################################################
                ### Plot empirical joing PDFs ###
################################################################################

fig = plt.figure(figsize = (5,3))
g = sns.jointplot(x = I_irrigated, y = D_irrigated, kind = 'hist', cbar = True)
g.fig.suptitle("Irrigation Season Joint PDF      " , fontsize = 16)
g.fig.subplots_adjust( top = 0.90 )
g.set_axis_labels('Standard Inflow', 'Standard Demand', fontsize=12)
# get the current positions of the joint ax and the ax for the marginal x
pos_joint_ax = g.ax_joint.get_position()
pos_marg_x_ax = g.ax_marg_x.get_position()

# reposition the joint ax so it has the same width as the marginal x ax
g.ax_joint.set_position([pos_joint_ax.x0, pos_joint_ax.y0, pos_marg_x_ax.width, pos_joint_ax.height])
# reposition the colorbar using new x positions and y positions of the joint ax
g.fig.axes[-1].set_position([1.0, pos_joint_ax.y0, .07, pos_joint_ax.height])


p2 = sns.jointplot(x = I_nonirrigated, y = D_nonirrigated, kind = 'hist', cbar = True)
plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
p2.fig.suptitle("Nonirrigation Season Joint PDF       " , fontsize = 16)
p2.fig.subplots_adjust( top = 0.90 )
p2.set_axis_labels('Standard Inflow', 'Standard Demand', fontsize=12)

# get the current positions of the joint ax and the ax for the marginal x
pos_joint_ax = p2.ax_joint.get_position()
pos_marg_x_ax = p2.ax_marg_x.get_position()

# reposition the joint ax so it has the same width as the marginal x ax
p2.ax_joint.set_position([pos_joint_ax.x0, pos_joint_ax.y0, pos_marg_x_ax.width, pos_joint_ax.height])
# reposition the colorbar using new x positions and y positions of the joint ax
p2.fig.axes[-1].set_position([.83, pos_joint_ax.y0, .07, pos_joint_ax.height])
