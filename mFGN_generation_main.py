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
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import random

# Load custom functions of interest
from my_stats_functions import standardize
from mFGN_generator import FGN_generate
from visual_validation import plot_FDC_range

# Set directory to current file location
# Change the current directory to the directory containing this .py
pname = os.path.abspath(sys.argv[0])
dname = os.path.dirname(pname)
os.chdir(dname)


###############################################################################
                ### Load Historic Data ###
###############################################################################

# Historical inflow and demand (18 year shared record)
historic_inflow = np.loadtxt('./data/historical/historic_inflow.csv', delimiter = ',').transpose()
historic_demand = np.loadtxt('./data/historic/historic_unit_demand.csv', delimiter = ',').transpose()



###############################################################################
                ### Generate synthetic streamflow using mFGN ###
###############################################################################

synthetic_inflow, synthetic_standard_inflow = FGN_generate(full_inflow_hist, n_years= 50, standardized = False)


###############################################################################
                            ### plot FDC ###
###############################################################################

plot_FDC_range(historic_inflow, synthetic_inflow, filename = 'Compare_flow_durations.jpeg')
