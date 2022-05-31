# -*- coding: utf-8 -*-
"""
Author: Trevor Amestoy
Cornell University
Spring 2022

Purpose:
    Produces synthetic streamflows using the modified Fractional Gaussian Noise
    method described in Kirsch et al (2013).

    Produces a figure comparison of the flow duration curves for historic and
    synthetic inflow timeseries.

Reference:
    Kirsch, B. R., G. W. Characklis, and H. B. Zeff (2013),
    Evaluating the impact of alternative hydro-climate scenarios on transfer
    agreements: A practical improvement for generating synthetic streamflows,
    J. Water Resour. Plann. Manage., 139(4), 396–406.

"""
# Core libraries
import numpy as np

# Load custom functions of interest
from mFGN_generator import FGN_generate
from visual_validation import plot_FDC_range



###############################################################################
                ### Load Historic Data ###
###############################################################################

# Historical inflow and demand (18 year shared record)
historic_inflow = np.loadtxt('./data/historic/historic_inflow.csv', delimiter = ',').transpose()
historic_demand = np.loadtxt('./data/historic/historic_unit_demand.csv', delimiter = ',').transpose()



###############################################################################
                ### Generate synthetic streamflow using mFGN ###
###############################################################################

synthetic_inflow, synthetic_standard_inflow = FGN_generate(historic_inflow, n_years= 50, standardized = False)

# Export
np.savetxt('./data/synthetic/synthetic_inflow.csv', synthetic_inflow, delimiter = ',')


###############################################################################
                            ### plot FDC ###
###############################################################################

plot_FDC_range(historic_inflow, synthetic_inflow, filename = 'Compare_flow_durations.jpeg')
