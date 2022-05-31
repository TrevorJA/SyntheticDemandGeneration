# -*- coding: utf-8 -*-
"""
Author: Trevor Amestoy
Cornell University
Spring 2022

Purpose:
Explores the correlation between synthetically generated inflow and demand,
for demands generated using a conditional expecation method.
Several figures are produced to visualize the correlation.

"""

import numpy as np
import pandas as pd
import seaborn as sns
import random
from scipy.stats import pearsonr

# load custom functions of interest
from my_stats_functions import standardize
from visual_validation import plot_correlation_map, plot_all_years



###############################################################################
                ### Load historic and synthetic data ###
###############################################################################

# Load historical inflow and demand
historic_inflow =  np.loadtxt('./data/historic/historic_inflow.csv', delimiter = ',').transpose()
historic_unit_demand = np.loadtxt('./data/historic/historic_unit_demand.csv', delimiter = ',').transpose()

# synthetic inflow and unit demand
synthetic_inflow = np.loadtxt('./data/synthetic/synthetic_inflow.csv', delimiter = ',')

synthetic_unit_demand = np.loadtxt('./data/synthetic/synthetic_unit_demand_CEM.csv', delimiter = ',')



###############################################################################
                ### Standardize data BY WEEK ###
###############################################################################

# standardize synthetic data by WEEK
synthetic_standard_inflow, synthetic_inflow_means, synthetic_inflow_sds = standardize(np.log(synthetic_inflow), columns = True)
synthetic_standard_demand, synthetic_demand_means, synthetic_demand_sds = standardize(synthetic_unit_demand, columns = True)

# standardize historic data by WEEK
historic_standard_inflow, historic_inflow_mean, historic_inflow_sds = standardize(np.log(historic_inflow), columns = True)
historic_standard_demand, historic_demand_mean, historic_demand_sds = standardize(historic_unit_demand, columns = True)

n_syn = synthetic_standard_demand.shape[0]

###############################################################################
            ### Calculate total standard annual pearson correlation ###
###############################################################################

# synthetic across the year
combined_std_synthetic = np.column_stack((synthetic_standard_inflow.reshape((52*n_syn)), synthetic_standard_demand.reshape(52*n_syn)))
combined_std_synthetic = pd.DataFrame(combined_std_synthetic, columns=['Synthetic Standard Log-Inflow', 'Synthetic Standard Demand'])
r_std, p_std = pearsonr(combined_std_synthetic['Synthetic Standard Log-Inflow'], combined_std_synthetic['Synthetic Standard Demand'])


###############################################################################
            ### Visualize weekly correlation - entire year ###
###############################################################################

method = 'Conditional Expectation Method'

# synthetic
p2 = sns.jointplot(data = combined_std_synthetic, x = 'Synthetic Standard Log-Inflow', y ='Synthetic Standard Demand', kind = 'reg')
p2.ax_marg_x.set_xlim(-3,3)
p2.ax_marg_y.set_ylim(-3,3)
p2.ax_joint.annotate('r = {:.2f}; '.format(r_std), xy=(0.3, 2.75), fontsize = 12)
p2.ax_joint.annotate('p = {:.2e}'.format(p_std), xy=(1.5, 2.75), fontsize = 12)
p2.set_axis_labels('Synthetic Standard Log-Inflow', 'Synthetic Standard Demand', fontsize=14)
p2.fig.suptitle(str("Synthetic Data: " + method + "\n Standardized Annual Joint Distribution"), fontsize=14)
p2.fig.tight_layout()
p2.fig.subplots_adjust(top=0.9)
p2.savefig('./figures/Synthetic_irrigation_joint_pdf_CEM.png')



###############################################################################
            ### Calculate IRRIGATED pearson correlation ###
###############################################################################

# The number of non_irrigated weeks in spring (1), fall (2), and irrigated weeks in mid-year
nw = 23 ; nn1 = 16 ; nn2= 13

# synthetic
combined_std_synthetic_irrigation = np.column_stack((synthetic_standard_inflow[:,nn1:(nn1+nw)].reshape((nw*n_syn)), synthetic_standard_demand[:,nn1:(nn1+nw)].reshape(nw*n_syn)))
combined_std_synthetic_irrigation = pd.DataFrame(combined_std_synthetic_irrigation, columns=['Synthetic Standard Log-Inflow', 'Synthetic Standard Demand'])
r_std_irr, p_std_irr = pearsonr(combined_std_synthetic_irrigation['Synthetic Standard Log-Inflow'], combined_std_synthetic_irrigation['Synthetic Standard Demand'])


###############################################################################
            ### Visualize weekly correlation - Irrigation Season ###
###############################################################################

# synthetic
p4 = sns.jointplot(data =combined_std_synthetic_irrigation, x = 'Synthetic Standard Log-Inflow', y ='Synthetic Standard Demand', kind = 'reg')
p4.ax_marg_x.set_xlim(-3,3)
p4.ax_marg_y.set_ylim(-3,3)
p4.ax_joint.annotate('r = {:.2f}; '.format(r_std_irr), xy=(0.3, 2.75), fontsize = 12)
p4.ax_joint.annotate('p = {:.2e}'.format(p_std_irr), xy=(1.5, 2.75), fontsize = 12)
p4.set_axis_labels('Synthetic Standard Log-Inflow', 'Synthetic Standard Demand', fontsize=14)
p4.fig.suptitle(str("Synthetic Data: " + method + "\n Standardized Irrigation Season Joint Distribution"), fontsize=14)
p4.fig.tight_layout()
p4.fig.subplots_adjust(top=0.90)
p4.savefig('./figures/Synthetic_irrigation_joint_pdf_CEM.png')



###############################################################################
    ### Visualize total correlation - Correlation Matrix - Annual ###
###############################################################################

# Create correlation structure matrices -- Demand on y-axis, inflow on X-axis
annual_synthetic_corr_structure = np.corrcoef(np.concatenate((synthetic_standard_inflow, synthetic_standard_demand), axis = 1), rowvar = False)[52:,0:52]

# Plot synthetic maps
plot_correlation_map(annual_synthetic_corr_structure, './figures/Synthetic_Annual_correlation_CEM', historic_data = False, annual_data = True)
plot_correlation_map(annual_synthetic_corr_structure, './figures/Synthetic_Irrigated_correlation_CEM', historic_data = False, annual_data = False)


###############################################################################
            ### Visualize non-standard joint distribution ###
###############################################################################

# synthetic across the year
combined_synthetic = np.column_stack((np.log(synthetic_inflow).reshape((52*n_syn)), synthetic_unit_demand.reshape(52*n_syn)))
combined_synthetic = pd.DataFrame(combined_synthetic, columns=['Synthetic Log-Inflow', 'Synthetic Unit Demand'])
r_syn, p_syn = pearsonr(combined_synthetic['Synthetic Log-Inflow'], combined_synthetic['Synthetic Unit Demand'])

# synthetic
p6 = sns.jointplot(data = combined_synthetic, x = 'Synthetic Log-Inflow', y ='Synthetic Unit Demand', kind = 'reg')
p6.ax_marg_x.set_xlim(0,10)
p6.ax_marg_y.set_ylim(0.4,1.5)
p6.ax_joint.annotate('r = {:.2f}; '.format(r_syn), xy=(5.5, 1.45), fontsize = 12)
p6.ax_joint.annotate('p = {:.2e}'.format(p_syn), xy=(7.5, 1.45), fontsize = 12)
p6.set_axis_labels('Synthetic Log-Inflow', 'Synthetic Unit Demand', fontsize=14)
p6.fig.suptitle(str("Synthetic Data: " + method + "\n Non-Standardized Joint Distribution"), fontsize=14)
p6.fig.tight_layout()
p6.fig.subplots_adjust(top = 0.9)
p6.savefig('./figures/Synthetic_joint_pdf_CEM.png')

###############################################################################
            ### Visualize non-standard joint distribution ###
###############################################################################

plot_all_years(historic_inflow, synthetic_inflow, 'Inflow', 'Inflow_comparisons_CEM', my_color = 'blue', highlight_mean = True, standardized = False)
plot_all_years(historic_unit_demand, synthetic_unit_demand, 'Unit Demand', 'Demand_unit_comparison_CEM', my_color = 'blue', highlight_mean = True, standardized = False)
plot_all_years(historic_standard_demand, synthetic_standard_demand, 'Weekly Standardized Demand', 'Demand_standard_comparison_CEM', my_color = 'blue', highlight_mean = True, standardized = False)
