# -*- coding: utf-8 -*-
"""
Author: Trevor Amestoy
Cornell University
Spring 2022

Purpose:
    Explores the Historic correlation between inflow and demand at one site.

"""

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr


# load custom functions of interest
from my_stats_functions import standardize
from visual_validation import plot_correlation_map



###############################################################################
                ### Load historic data ###
###############################################################################

# Durham historical inflow and demand
historic_inflow = np.loadtxt('./data/historic/historic_inflow.csv', delimiter = ',').transpose()
historic_unit_demand = np.loadtxt('./data/historic/historic_unit_demand.csv', delimiter = ',').transpose()



###############################################################################
                ### Standardize data BY WEEK ###
###############################################################################

# standardize historic data by WEEK
historic_standard_inflow, historic_inflow_mean, historic_inflow_sds = standardize(np.log(historic_inflow), columns = True)
historic_standard_demand, historic_demand_mean, historic_demand_sds = standardize(historic_unit_demand, columns = True)



###############################################################################
            ### Calculate total standard annual pearson correlation ###
###############################################################################

# Historic across the year
combined_std_historic = np.column_stack((historic_standard_inflow.reshape((52*18)), historic_standard_demand.reshape(52*18)))
combined_std_historic = pd.DataFrame(combined_std_historic, columns=['Historic Standard Log-Inflow', 'Historic Standard Demand'])
r_std, p_std = pearsonr(combined_std_historic['Historic Standard Log-Inflow'], combined_std_historic['Historic Standard Demand'])


###############################################################################
            ### Visualize weekly correlation - entire year ###
###############################################################################

# Historic
p1 = sns.jointplot(data = combined_std_historic, x = 'Historic Standard Log-Inflow', y ='Historic Standard Demand', kind = 'reg')
p1.ax_marg_x.set_xlim(-3,3)
p1.ax_marg_y.set_ylim(-3,3)
p1.ax_joint.annotate('r = {:.2f}; '.format(r_std), xy=(0.3, 2.75), fontsize = 12)
p1.ax_joint.annotate('p = {:.2e}'.format(p_std), xy=(1.5, 2.75), fontsize = 12)
p1.fig.suptitle("Historic Data: Annual Joint Distribution \n", fontsize = 12)
p1.fig.tight_layout()
p1.fig.subplots_adjust(top = 0.9)
p1.savefig('./figures/Historic_Annual_Joint_Distribution.png')



###############################################################################
            ### Calculate IRRIGATED pearson correlation ###
###############################################################################

# The number of non_irrigated weeks in spring (1), fall (2), and irrigated weeks in mid-year
nw = 23 ; nn1 = 16 ; nn2= 13

#irrigated indices:  nn1:(nn1+nw)

# Historic
combined_std_historic_irrigation = np.column_stack((historic_standard_inflow[:,nn1:(nn1+nw)].reshape((nw*18)), historic_standard_demand[:,nn1:(nn1+nw)].reshape(nw*18)))
combined_std_historic_irrigation = pd.DataFrame(combined_std_historic_irrigation, columns=['Historic Standard Log-Inflow', 'Historic Standard Demand'])
r_std_irr, p_std_irr = pearsonr(combined_std_historic_irrigation['Historic Standard Log-Inflow'], combined_std_historic_irrigation['Historic Standard Demand'])

###############################################################################
            ### Visualize weekly correlation - Irrigation Season ###
###############################################################################

# Historic
p3 = sns.jointplot(data = combined_std_historic_irrigation, x = 'Historic Standard Log-Inflow', y = 'Historic Standard Demand', kind = 'reg')
p3.ax_marg_x.set_xlim(-3,3)
p3.ax_marg_y.set_ylim(-3,3)
p3.ax_joint.annotate('r = {:.2f}; '.format(r_std_irr), xy=(0.3, 2.75), fontsize = 12)
p3.ax_joint.annotate('p = {:.2e}'.format(p_std_irr), xy=(1.5, 2.75), fontsize = 12)
p3.fig.suptitle("Historic Data: Irrigation Season Joint Distribution", fontsize = 12)
p3.fig.tight_layout()
p3.fig.subplots_adjust(top = 0.9)
p3.savefig('./figures/Historic_Irrigation_Joint_Distribution.png')



###############################################################################
    ### Visualize total correlation - Correlation Matrix - Annual ###
###############################################################################

# Create correlation structure matrices -- Demand on y-axis, inflow on X-axis
annual_historic_corr_structure = np.corrcoef(np.concatenate((historic_standard_inflow, historic_standard_demand), axis = 1), rowvar = False)[52:,0:52]

# Plot historic maps
plot_correlation_map(annual_historic_corr_structure, './figures/Historic_Annual_Correlation_Pattern', historic_data = True, annual_data = True)
plot_correlation_map(annual_historic_corr_structure, './figures/Historic_Irrgigation_Correlation_Pattern', historic_data = True, annual_data = False)



###############################################################################
            ### Visualize non-standard joint distribution ###
###############################################################################

# Historic across the year
combined_historic = np.column_stack((np.log(historic_inflow).reshape((52*18)), historic_unit_demand.reshape(52*18)))
combined_historic = pd.DataFrame(combined_historic, columns=['Historic Log-Inflow', 'Historic Unit Demand'])
r, p = pearsonr(combined_historic['Historic Log-Inflow'], combined_historic['Historic Unit Demand'])

# Historic
p5 = sns.jointplot(data = combined_historic, x = 'Historic Log-Inflow', y ='Historic Unit Demand', kind = 'reg')
p5.ax_marg_x.set_xlim(0,10)
p5.ax_marg_y.set_ylim(0.4,1.5)
p5.ax_joint.annotate('r = {:.2f}; '.format(r), xy=(5.5, 1.45), fontsize = 12)
p5.ax_joint.annotate('p = {:.2e}'.format(p), xy=(7.5, 1.45), fontsize = 12)
p5.fig.suptitle("Historic Data: Non-Standardized Joint Distribution \n", fontsize = 12)
p5.fig.tight_layout()
p5.fig.subplots_adjust(top = 0.9)
p5.savefig('./figures/Historic_Nonstandardized_Joint_Distribution.png')
